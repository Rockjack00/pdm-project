import traceback
from typing import Any

import numpy as np
import numpy.typing as npt

import pdmproject.cspace.obstacle as obs
from pdmproject.cspace.obstacle import CartesianIterator, HypercubeIterator
from pdmproject.cspace.tree import SparseOccupancyTree

from . import SamplerBase


class NullSpaceSampler(SamplerBase):
    def __init__(
        self,
        lower_bound: npt.ArrayLike = (
            -5,
            -5,
            -np.pi,
            -np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
            -np.pi,
        ),
        upper_bound: npt.ArrayLike = (
            5,
            5,
            np.pi,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi,
        ),
        dimension: int = 3,
        resolution: int = 7,
    ) -> None:
        """Create a uniform NullSpaceSampler.

        Args:
            lower_bound (npt.ArrayLike(len == 7), optional): The lower bound of this Sampler. Defaults to ( -5, -5, -np.pi, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi, ).
            upper_bound (npt.ArrayLike(len == 7), optional): The upper bound of this Sampler. Defaults to ( 5, 5, np.pi, np.pi / 2, np.pi / 2, np.pi / 2, np.pi, ).
            dimension: The dimension of the sample space (one dimension for each joint).
               | Link  | Dimension |
               |-------|-----------|
               |base(0)|     2     |
               |   1   |     4     |
               |   2   |     5     |
               |   3   |     6     |
               |   4*  |     6     |
               * link 4 is always ignored in the sample space tree.

            resolution: The resolution of the sample space, where each dimension is split into 2^resolution voxels.
        """
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        upper_bound = np.asarray(upper_bound, dtype=np.float64)

        assert len(lower_bound) == 7, "7 lower bounds must be specified"
        assert len(upper_bound) == 7, "7 upper bounds must be specified"

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.dimension = dimension
        self.sample_space = SparseOccupancyTree(
            dimension=dimension,
            resolution=resolution,
            limits=np.array([lower_bound[:dimension], upper_bound[:dimension]]),
            wraps=[False, False, True, False, False, False][:dimension],
        )
        self.debug_iter = 0

    @SamplerBase.lower_bound.getter
    def lower_bound(self):
        return self._lower_bound

    @SamplerBase.upper_bound.getter
    def upper_bound(self):
        return self._upper_bound

    def get_sample(self, sample_count=None):
        if sample_count is None:
            sample_count = 1

        cs_samples = np.zeros((sample_count, self.dimension))
        for i in range(sample_count):
            cs_samples[i, :] = self._get_cspace_sample()

        last_samples = np.random.uniform(
            self._lower_bound[self.dimension :],
            self._upper_bound[self.dimension :],
            size=(sample_count, 7 - self.dimension),
        )

        samples = np.hstack((cs_samples, last_samples))

        assert samples.shape == (sample_count, 7)
        return samples.T.squeeze(-1)

    def _get_cspace_sample(self):
        """Get a uniform sample from the sample space (first D joints), ignoring collision regions. D is the dimension of self.sample_space.

        Returns:
            A numpy array of size (1,D) containing a uniformly distributed
            random configuration of the first D joints.
        """
        node = self.sample_space._root
        node_stack = []
        depth = 1
        while node is not None:
            depth = len(node_stack) + 1
            max_content = self.sample_space.max_content(depth)
            numerators = max_content - node.get_values()
            probabilities = numerators / np.sum(numerators)

            child_idx = np.random.choice(len(probabilities), p=probabilities)
            node = node.get_child(child_idx)
            node_stack.append((child_idx, node))

        node_key = self.sample_space._assemble_voxel_key(node_stack)
        bounds = self.sample_space.get_configurations(node_key, depth)

        # print(f'Sampled ({node_key:b}, {depth}) to get ({bounds[0]}, {bounds[1]})')  # FIXME:DEBUG:REMOVE

        return np.random.uniform(bounds[0, :], bounds[1, :])

    def callback(self, poses: npt.NDArray, collision_checker: Any) -> None:
        try:
            for link, collisions in zip(*collision_checker.collision_finder()):
                # Skip unimplemented updates
                if link > 0:
                    continue
                self._update_sample_space(link, collisions)
        except Exception as e:
            print(
                "[ERROR] - While attempting to update the sample space encountered the following error:"
            )
            traceback.print_exception(e)

    def _update_sample_space(self, link, collisions):
        """A callback function to create a new cspace obstacle and insert it into the sample space.

        WARNING: Currently only implemented for link 0.

        Args:
            link: An index from 0-3 to select which link to use where the base is
                link 0. Link 4 is the robot hand but it is ignored for the
                purpose of calculating the null space because we model it as a
                cylinder.
            collisions: A numpy array of one or two workspace points (rows) where
                the link is in collision. Ignores the second collision if solving
                for link 0.
        """
        self.debug_iter += 1

        # TODO: parallelize this with multiple workers
        # may require load balancing

        midpoints = []
        node_stack = []

        # param generator generates sets of parameters [0,1] x 5 where each
        # of the 4-d "hyper-faces" is solved by fixing one of the parameters at 0 and 1
        # and getting a cartesean product of the rest.
        if link == 0:
            marcher = CartesianIterator(
                [
                    obs.dtheta_step(self.sample_space),
                    obs.voxel_step(3, self.sample_space),
                ],  # remove these if not in the tree
                # obs.voxel_step(4, self.sample_space),  # remove these if not in the tree
                # obs.voxel_step(5, self.sample_space),  # remove these if not in the tree
                # obs.voxel_step(6, self.sample_space)]  # remove these if not in the tree
                outer=0,
            )
        # TODO
        elif link >= 1:
            # cry
            marcher = []
            # marcher = HypercubeIterator()
            raise NotImplementedError

        # using the generator, march over the null space boundary
        old_content = self.sample_space._root.sum_values()
        for params in marcher:
            points = obs.calc_ns(
                collisions, link, params, self.sample_space.limits[:, 2:]
            )
            if link > 0:
                midpoints.append(points[len(points) // 2, :])

            voxels = self.sample_space.locate(points)
            for voxel in voxels:
                if self.sample_space.is_full(voxel):
                    continue
                # use a bigger brush (only in the x and y directions) to make the boundary
                node_stack = self.sample_space.paint(
                    voxel, directions=0b011011, node_stack=node_stack
                )
                # node_stack = self.sample_space.set(voxel, node_stack=node_stack)
            new_content = self.sample_space._root.sum_values()

            # we already filled this cyclinder
            if link == 0 and (new_content - old_content) == 0:
                # print('Already filled this cylinder?') # FIXME:DEBUG:REMOVE
                return

        # TODO: get a smarter interior point or multiple interior points
        # multiple interior points would require load balancing between workers
        if link == 0:
            interior_point = np.ones(self.dimension)
            interior_point[:2] = collisions[0, :2]
        else:
            interior_point = np.average(midpoints, axis=0)

        # fill the inside of the obstacle
        self.sample_space.flood_fill(interior_point)

        # debug
        if 1:  # FIXME:DEBUG:REMOVE
            new_content = self.sample_space._root.sum_values()
            print(
                f"[{self.debug_iter}] Obstacle at: {collisions[0,:]},"
                f" Set {new_content - old_content:>3} voxels. Sample space content: {new_content / self.sample_space.max_content(0):>8.4%}"
            )
            if new_content - old_content == 0:
                breakpoint()
