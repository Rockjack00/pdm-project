from dataclasses import InitVar, dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, eq=False)
class Node:
    array: InitVar[npt.ArrayLike]
    _array: npt.NDArray[np.float64] = field(init=False)
    parent: Optional["Node"] = field(init=False, default=None, compare=False)
    cost: float = field(init=False, default=0.0, compare=False)

    def __post_init__(self, array):
        self._array = np.asarray(array, dtype=np.float64)
        assert self._array.shape == (7,)

    @classmethod
    def from_array(cls, array) -> "Node":
        """Create node object from array

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
        return self._array

    def __eq__(self, value: "Node") -> bool:
        return (self._array == value._array).all()

    @property
    def q1(self) -> float:
        return self._array[0]

    @property
    def q2(self) -> float:
        return self._array[1]

    @property
    def q3(self) -> float:
        return self._array[2]

    @property
    def q4(self) -> float:
        return self._array[3]

    @property
    def q5(self) -> float:
        return self._array[4]

    @property
    def q6(self) -> float:
        return self._array[5]

    @property
    def q7(self) -> float:
        return self._array[6]
