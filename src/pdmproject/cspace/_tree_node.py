from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, repr=False)
class SparseVoxelTreeNode(ABC):
    d: int  # dimension of the tree

    @abstractmethod
    def sum_values(self) -> int:
        """Get the sum of all values of the children."""
        pass

    @abstractmethod
    def get_values(self) -> int:
        """Get a list of all values of the children."""
        pass

    def get_child(self, index: int) -> Optional["SparseVoxelTreeNode"]:
        """Get the child at index."""
        return None


# TODO: define __get__ and __set__ for brackets to modify children
@dataclass(slots=True, repr=False)
class TopologyNode(SparseVoxelTreeNode):
    """A node in a sparse voxel tree data structure which contains topological information."""

    children: list[Optional[SparseVoxelTreeNode]] = field(init=False)
    values: npt.NDArray[np.integer] = field(init=False)

    def __post_init__(self):
        """Create a TopologyNode.

        Args:
            dimension: The dimension of this node. It will have 2^d children.
        """
        self.children = [None] * (2**self.d)  # A list of references to the children
        self.values = np.zeros(
            2**self.d, dtype=int
        )  # A list of values assiciated with each child

    def __repr__(self):
        """The string representation of a topological node."""
        return (
            f"({2 ** self.d}-Node [content={self.sum_values()}, values={self.values}])"
        )

    def sum_values(self) -> int:
        """Get the sum of all values of the children."""
        return np.sum(self.values)  # type: ignore

    def get_values(self):
        """Get a list of all values of the children."""
        return self.values

    def get_child(self, index):
        """Get the child at index."""
        return self.children[index]


# TODO: define __get__ and __set__ for brackets to modify children
@dataclass(slots=True, repr=False)
class BinaryLeafNode(SparseVoxelTreeNode):
    """A node in a sparse occupancy tree data structure which contains binary data for a collection of leaves. Each voxel is represented with one bit."""

    values: int = field(init=False, default=0)

    def __repr__(self):
        """The string representation of a leaf node."""
        return f"({2 ** self.d}-BinaryLeafNode [content={self.sum_values()}, values={self.values:b}])"

    def increment(self, index):
        """Set the value at index to 1.

        Args:
           index: The voxel index to set.

        Returns:
            The new sum of all values of children.
        """
        self.values |= 1 << index
        return self.sum_values()

    def sum_values(self) -> int:
        """Get the sum of all values of the children."""
        return self.values.bit_count()

    def get_values(self):
        """Get a list of all values of the children."""
        return (self.values & (1 << np.arange(2**self.d)) > 0) * 1
