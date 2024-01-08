from dataclasses import KW_ONLY, InitVar, dataclass, field
from math import nan
from typing import Any, ClassVar, Optional

from matplotlib.axes import Axes
import numpy as np
from mpscenes.obstacles.box_obstacle import BoxObstacle
from pybullet_utils.transformations import quaternion_from_euler


@dataclass
class Wall:
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    _: KW_ONLY
    thickness: float = 0.1
    wall_height: float = 2.0
    extra_data: Optional[dict] = None
    simulation_name: InitVar[Optional[str]] = None
    _registered: bool = field(init=False, default=False)
    _name: Optional[str] = field(init=False, default=None)
    _wall_length: float = field(init=False, default=nan)
    _content_dict: Optional[dict[str, Any]] = field(init=False, default=None)
    _content_dicts: Optional[list[dict[str, Any]]] = field(init=False, default=None)

    DEFAULT_NAME_TEMPLATE: ClassVar[str] = "wall-"

    def __post_init__(self, simulation_name: Optional[str]):
        if simulation_name is not None:
            self._name = simulation_name

    def _generate_content_dicts(self, regenerate: bool = False):
        if self._content_dicts is None or regenerate:
            self._generate_content_dict()

        # Assert so type checker is aware of the current situation
        assert self._content_dict is not None
        self._content_dicts = [self._content_dict]

    def _generate_content_dict(self, regenerate: bool = False):
        if not (self._content_dict is None or regenerate):
            return None

        start_vec = np.array(self.start_point)
        end_vec = np.array(self.end_point)

        wall_vec = end_vec - start_vec
        wall_length = np.linalg.norm(wall_vec)
        wall_center = wall_vec / 2 + start_vec

        self._wall_length = wall_length  # type: ignore

        position: list[float] = wall_center.tolist()
        position.append(self.wall_height / 2.0)

        orientation = quaternion_from_euler(-np.arctan(wall_vec[1] / wall_vec[0]), 0, 0)

        content_dict = {
            "type": "box",
            "geometry": {
                "position": position,
                "width": self.thickness,
                "length": float(wall_length),
                "height": float(self.wall_height),
                "orientation": orientation.tolist(),
            },
        }

        if self.extra_data is not None:
            content_dict.update(self.extra_data)

        if "rgba" not in content_dict:
            content_dict["rgba"] = [0.5, 0.5, 0.5, 1.0]

        self._content_dict = content_dict

    def _generate_wall_segments(self, regenerate: bool = False) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )

        self._generate_content_dict(regenerate)

        return [
            BoxObstacle(
                name=self._name,
                content_dict=self._content_dict,
            )
        ]

    def has_name(self) -> bool:
        return self._name is not None

    def has_unique_name(self) -> bool:
        return self.has_name() and self._name.startswith(self.DEFAULT_NAME_TEMPLATE)  # type: ignore

    def _register(self, number: int, name_set: set[str]) -> bool:
        """Register this Wall

        Args:
            number (int): The Wall number

        Returns:
            bool: If the number has been used
        """
        assert not self.is_registered, "The wall has already been registered"

        if self._name is None:
            self._name = f"{self.DEFAULT_NAME_TEMPLATE}{number:02}"
            assert self._name not in name_set
            name_set.add(self._name)
            self._registered = True
            return True
        else:
            assert (
                self._name not in name_set
            ), "The name of this wall has already been registered!"
            name_set.add(self._name)
            self._registered = True
            return False

    @property
    def is_registered(self) -> bool:
        """If the wall has been registered yet"""
        return self._registered

    @property
    def content_dict(self) -> dict[str, Any]:
        self._generate_content_dict()

        # Assert so type checker is aware of the current situation
        assert self._content_dict is not None

        return self._content_dict

    @property
    def content_dicts(self) -> list[dict[str, Any]]:
        self._generate_content_dicts()

        # Assert so type checker is aware of the current situation
        assert self._content_dicts is not None

        return self._content_dicts

    @property
    def color(self) -> list[float]:
        self._generate_content_dict()
        return self.content_dict["rgba"]

    @property
    def wall_length(self) -> float:
        self._generate_content_dict()
        return self._wall_length

    def _plot2d(self, ax: Axes):
        ax.plot(
            [self.start_point[0], self.end_point[0]],
            [self.start_point[1], self.end_point[1]],
            color=self.color,
        )
