"""This submodule contains the C-space Node class required for the RRT* graph."""
from dataclasses import InitVar, dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, eq=False)
class Node:
    """An 7-Dimensional Configuration Space (R²×S¹×R³×S¹) Node.

    This Node is used in the RRTStar graph.
    """

    array: InitVar[npt.ArrayLike]
    _array: npt.NDArray[np.float64] = field(init=False)
    parent: Optional["Node"] = field(init=False, default=None, compare=False)
    cost: float = field(init=False, default=0.0, compare=False)

    def __post_init__(self, array):
        """Convert the given ArrayLike to an numpy array and check if the sizing is correct (shape == (7,))."""
        self._array = np.asarray(array, dtype=np.float64)
        assert self._array.shape == (7,)

    @classmethod
    def from_array(cls, array) -> "Node":
        """Create node object from array.

        Args:
            array (Union[np.ndarray, List]): Pose for the node

        Raises:
            ValueError: Wrong amount of joint values given

        Returns:
            Node: Node for corresponding pose
        """
        # Ensure the array has the correct number of elements
        # if len(array) != 7:
        # raise ValueError("Array must contain 7 elements for a 7D point.")

        # Use array elements to initialize Node attributes
        return cls(array)

    def get_7d_point(self) -> npt.NDArray[np.float64]:
        """Get a Numpy Array of this 7-Dimensional Configuration Space point.

        Returns:
            npt.NDArray[np.float64]: The Numpy Array of this point.
        """
        return self._array

    def __eq__(self, value: "Node") -> bool:
        """Compare this Node to another for equality, by its coordindates.

        Args:
            value (Node): The node to compare to.

        Returns:
            bool: True if the Nodes have the same coordinates.
        """
        return (self._array == value._array).all()

    @property
    def q1(self) -> float:
        """The value 1st dimension of this C-space Node."""
        return self._array[0]

    @property
    def q2(self) -> float:
        """The value 2nd dimension of this C-space Node."""
        return self._array[1]

    @property
    def q3(self) -> float:
        """The value 3rd dimension of this C-space Node."""
        return self._array[2]

    @property
    def q4(self) -> float:
        """The value 4th dimension of this C-space Node."""
        return self._array[3]

    @property
    def q5(self) -> float:
        """The value 5th dimension of this C-space Node."""
        return self._array[4]

    @property
    def q6(self) -> float:
        """The value 6th dimension of this C-space Node."""
        return self._array[5]

    @property
    def q7(self) -> float:
        """The value 7th dimension of this C-space Node."""
        return self._array[6]
