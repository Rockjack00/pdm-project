from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


class SamplerBase(ABC):
    """The abstact base class for Samplers"""

    @abstractmethod
    def get_sample(self, sample_count: int = 1) -> np.ndarray[tuple[int, Literal[7]], np.dtype[np.float64]]:
        """Get a random sample

        Args
            sample_count (int, optional): The amount of samples to generate. Defaults to 1
        
        Returns:
            np.ndarray[tuple[int, Literal[7]], np.dtype[np.float64]]: The random sample
        """
        pass

    def callback(self, poses: npt.NDArray, collision_checker: Any) -> None:
        """A callback to call when collision is found.

        Args:
            poses (npt.NDArray): The configuration state pose(s) to mark as occupied by obstacles.
            collision_checker (Any): The collision checker of the Planner.

        Returns:
            None: Nothing
        """
        return None
