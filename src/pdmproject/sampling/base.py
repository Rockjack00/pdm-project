from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ..planning.node import Node


class SamplerBase(ABC):
    """The abstact base class for Samplers"""

    @abstractmethod
    def get_sample(
        self, sample_count: Optional[int] = None
    ) -> np.ndarray[tuple[int, Literal[7]] | tuple[Literal[7]], np.dtype[np.float64]]:
        """Get a random sample

        Args
            sample_count (Optional[int], optional): The amount of samples to generate. Defaults to None (7D array)

        Returns:
            np.ndarray[tuple[int, Literal[7]] | tuple[Literal[7]], np.dtype[np.float64]]: The random sample
        """
        pass

    def get_node_sample(self) -> Node:
        """Sample a random Node

        Returns:
            Node: The newly sampled Node
        """
        # FIXME: TEMPORARY HACK FOR GOALPOINT
        if self._goal_prob is not None:
            if np.random.rand() <= self._goal_prob:
                return self._goal

        return Node.from_array(self.get_sample())

    def callback(self, poses: npt.NDArray, collision_checker: Any) -> None:
        """A callback to call when collision is found.

        Args:
            poses (npt.NDArray): The configuration state pose(s) to mark as occupied by obstacles.
            collision_checker (Any): The collision checker of the Planner.

        Returns:
            None: Nothing
        """
        return None

    @property
    @abstractmethod
    def lower_bound(self) -> Sequence:
        return NotImplemented()

    @property
    @abstractmethod
    def upper_bound(self) -> Sequence:
        return NotImplemented()

    def register_goal_hack(self, goal: Node, probability: float = 0.1):
        assert (
            0 <= probability and probability <= 1
        ), "probablity should be between [0,1]"

        self._goal_prob = probability
        self._goal = goal
