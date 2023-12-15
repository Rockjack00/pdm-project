from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import ClassVar, Optional

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

    DEFAULT_NAME_TEMPLATE: ClassVar[str] = "wall-"

    def __post_init__(self, simulation_name: Optional[str]):
        if simulation_name is not None:
            self._name = simulation_name

    def _generate_wall_segments(self) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )

        start_vec = np.array(self.start_point)
        end_vec = np.array(self.end_point)

        wall_vec = end_vec - start_vec
        wall_length = np.linalg.norm(wall_vec)
        wall_center = wall_vec / 2 + start_vec

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

        return [
            BoxObstacle(
                name=self._name,
                content_dict=content_dict,
            )
        ]

    def has_name(self) -> bool:
        return self._name is not None

    def has_unique_name(self) -> bool:
        return self.has_name() and self._name.startswith(self.DEFAULT_NAME_TEMPLATE)

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
