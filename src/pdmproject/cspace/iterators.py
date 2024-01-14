from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class IteratorBase(ABC):
    """The abstact base class for Iterators."""

    @abstractmethod
    def __next__(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Get the next set of parameters to evaluate.

        Returns:
            A numpy array of shape (N,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march through the entire hypercube, in
            lexicographical order. N is dependent on the step sizes.
        """
        pass

    def __iter__(self):
        """Reset this and return it."""
        self._reset()
        return self
    
    @abstractmethod
    def _reset(self):
        """Reset the iterator from the first point."""
        pass

    @staticmethod
    def _cartesian_product(*arrays):
        """Get the cartesian product of several numpy arrays.

        Adopted from https://stackoverflow.com/a/11146645

        Returns:
            A numpy array of shape (N,D), in which each row is an element of
            the cartesian product of the inputs. N is the product the lengths of
            each input array, and D is the number of input arrays.
        """
        la = len(arrays)
        # dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=float)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)


class CartesianIterator(IteratorBase):
    """An Iterator which cycles over the Cartesian product of a set of parameters.

    This operates similarly to itertools.product() however it is specialized
    for large numpy arrays. This generates blocks of parameters to traverse
    a space lexicographically with a maximum step size at each point.
    """

    def __init__(self, deltas, outer=0, limits=None):
        """Create a CartesianIterator.

        Args:
            deltas: A list of step sizes for each dimension to march over.
            outer: The parameter index which is iterated over in the outer loop.
            limits: A numpy array of shape (2,d) which indicates minimum and
                maximum values for each dimension. Defaults to [0,1] for every
                dimension.
        """
        self.deltas = deltas
        self.dimension = len(deltas)
        self.outer = outer

        if limits is None:
            limits = np.indices((2, self.dimension))[0]
        else:
            assert np.shape(limits) == (2, self.dimension)
        self.limits = limits

        self.spaces = self._create_spaces()
        self._reset()

    def __next__(self):
        """Get the next set of parameters to evaluate.

        Returns:
            A numpy array of shape (N,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march through the entire hypercube, in
            lexicographical order. N is dependent on the step size for each
            dimension, but will return all parameters for an entire step in the
            first dimension at once.
        """
        if self.next_index >= len(self.spaces[0]):
            raise StopIteration

        self.next_index += 1

        spaces = []
        for i in range(len(self.spaces)):
            if i is not self.outer:
                spaces.append(self.spaces[i])
            else:
                spaces.append(np.array([self.spaces[i][self.next_index - 1]]))
        return self._cartesian_product(*spaces)

    def _reset(self):
        """Reset the iterator from the first point."""
        self.next_index = 0

    def _create_spaces(self):
        """Create linear spaces for each dimension with a fixed maximum step size."""
        spaces = []
        lengths = np.ceil(1 / np.array(self.deltas)).astype(int)
        for i in range(self.dimension):
            param_space = np.linspace(
                self.limits[0, i], self.limits[1, i], num=lengths[i]
            )
            spaces.append(param_space)
        return spaces

    def update(self, last_point):
        pass


class HypercubeIterator(IteratorBase):
    """An Iterator which steps over an hypercube.

    This generates parameters to traverse each sub-manifold
    lexicographically with a maximum step size at each point.
    """

    def __init__(self, step_functions, sample_space, collisions, limits=None):
        """Create a HypercubeIterator.

        Args:
            step_functions: A list of functions to calculate the step size for
                each dimension to march over.
            resolution: The highest tree depth. The number of cells along any
                given dimension will be 2^resolution.
            limits: A numpy array of shape (2,d) which indicates minimum and
                maximum values for each dimension. Defaults to [0,1] for every
                dimension.
        """
        self.dimension = len(step_functions)
        self.step_functions = step_functions

        if limits is None:
            limits = np.indices((2, self.dimension))[0]
        self.limits = limits
        self.last_point = None
        self.last_params = np.zeros(self.dimension)
        self.sample_space = sample_space
        self.collisions = collisions
        self._reset()

    def _reset(self):
        self.current_dim = 0
        self.current_face = 0

    def update(self, last_point):
        self.last_point = last_point

    def __next__(self):
        """Get the next set of parameters to evaluate.

        Returns:
            A numpy array of shape (1,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube, one
            point at a time. 
        """
        # check if we are done
        if self.current_dim >= self.dimension:
            raise StopIteration

        # get all the parameters for this face
        for params in self._step(self.limits[0,:], 0):
            if params is not None:
                return params

            # change which face to march over
            if self.current_face == 0:
                self.current_face = 1
            elif self.current_dim < self.dimensions:
                self.current_face = 0
                self.current_dim += 1
            else:
                # we are done
                raise StopIteration

    def _step(self, first_params, p):
        """Return the next set of parameters, based on the step functions.

        Returns:
            A numpy array of shape (1,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube.
        """
        # if this is the fixed dimension, keep going
        if p == self.current_dim:
            p += 1

        # base case: no more dimensions to step:
        if p == self.dimensions:
            return None

        last_params = self.limits[0,p:]
        current_param = last_params[0]

        while current_param < self.limits[1, p]:
            # recursive step: run all later dimensions
            for params in self._step(last_params[:p+1], p + 1):
                if params is None:
                    break
                yield params

            # Then step once in this dimension
            delta = self.step_functions[p](self.sample_space, self.collisions, self.last_point, self.last_params)
            current_param += delta
            if current_param >= self.limits[1, p]:
                current_param = self.limits[1, p]
            last_params[0] = current_param
            params = np.hstack(first_params,last_params)
            params[self.current_dim] = self.limits[self.current_face, self.current_dim]
            yield params

        # finish this dimension
        return None


