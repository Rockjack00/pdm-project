from typing import Any

import numpy as np
import numpy.typing as npt

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
        dimension: int = 6,
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
            wraps=[False, False, True, False, False, False],
        )

        for i in range(2**dimension):
            voxel = i
            if i in (0, 3):
                continue
            self.sample_space.set(voxel, depth=1)

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
        print(self.sample_space._root.sum_values())
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

        return np.random.uniform(bounds[0, :], bounds[1, :])

    def callback(self, poses: npt.NDArray, collision_checker: Any) -> None:
        return super().callback(poses, collision_checker)

    @SamplerBase.lower_bound.getter
    def lower_bound(self):
        return self._lower_bound

    @SamplerBase.upper_bound.getter
    def upper_bound(self):
        return self._upper_bound
