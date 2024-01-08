import numpy as np
import numpy.typing as npt
from numpy import dtype, ndarray

from .base import SamplerBase
from pdmproject.cspace.tree import SparseBinaryTetrahexacontree


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
        resolution: int = 7,
    ) -> None:
        """Create a uniform NullSpaceSampler

        Args:
            lower_bound (npt.ArrayLike(len == 7), optional): The lower bound of this Sampler. Defaults to ( -5, -5, -np.pi, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi, ).
            upper_bound (npt.ArrayLike(len == 7), optional): The upper bound of this Sampler. Defaults to ( 5, 5, np.pi, np.pi / 2, np.pi / 2, np.pi / 2, np.pi, ).
            resolution: The resolution of the sample space, where each dimension is split into 2^resolution voxels. 
        """
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        upper_bound = np.asarray(upper_bound, dtype=np.float64)

        assert len(lower_bound) == 7, "7 lower bounds must be specified"
        assert len(upper_bound) == 7, "7 upper bounds must be specified"

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.sample_space = SparseBinaryTetrahexacontree(resolution=resolution,
                                                         limits=np.array([lower_bound,
                                                                          upper_bound]),
                                                         wraps=[False,False,True,False,False,False])

    def get_sample(self, sample_count=None):
        if sample_count is None:
            sample_count = 1

        cs_sample = np.zeros(sample_count,6)
        for i in range(sample_count):
            cs_sample[i,:] = self._get_cspace_sample()

        last_sample = np.random.uniform(self._lower_bound[-1], self._upper_bound[-1], size=(sample_count,1))
        return np.hstack(cs_samples, last_sample)

    def _get_cspace_sample(self):
        '''
        Get a uniform sample from the sample space (first 6 joints), 
        ignoring collision regions.

        Returns:
            A numpy array of size (1,6) containing a uniformly distributed random
            configuration of the first 6 joints.
        '''

        node = self.sample_space._root
        node_stack = []
        while node is not None:
            depth = len(node_stack) + 1
            max_content = self.sample_space.max_content(depth)
            numerators = max_content - node.values()
            probabilities = numerators / np.sum(numerators)

            child = np.random.choice(range(node.children), p=probabilities)
            node = node.children[child_idx]
            node_stack.append((child, node))
            
        node_key = self.sample_space._assemble_voxel_key(node_stack)
        bounds = self.sample_space.get_configurations(self, node_key, depth)

        return np.random.uniform(bounds[0,:], bounds[1,:])


    @SamplerBase.lower_bound.getter
    def lower_bound(self):
        return self._lower_bound

    @SamplerBase.upper_bound.getter
    def upper_bound(self):
        return self._upper_bound
