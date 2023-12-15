import copy
from typing import ClassVar, Optional

from mpscenes.obstacles.box_obstacle import BoxObstacle

from .wall import Wall


class PerimeterWall(Wall):
    DEFAULT_NAME_TEMPLATE: ClassVar[str] = "perimeterwall-"

    def __init__(
        self,
        center: tuple[float, float],
        width: float = 10.0,
        length: float = 10.0,
        thickness: float = 0.1,
        wall_height: float = 2.0,
        extra_data: Optional[dict] = None,
        simulation_name: Optional[str] = None,
    ) -> None:
        self.center = center
        self.width = width
        self.length = length
        self.thickness = thickness
        self.wall_height = wall_height
        self.extra_data = extra_data
        if simulation_name is not None:
            self._name = simulation_name

        self._content_dict = {
            "type": "box",
            "geometry": {
                "height": self.wall_height,
            },
        }

        if self.extra_data is not None:
            self._content_dict.update(self.extra_data)

        if "rgba" not in self._content_dict:
            self._content_dict["rgba"] = [1.0, 1.0, 1.0, 1.0]

    def _generate_wall_segments(self) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )
        wall_segments = []
        self._content_dicts = []

        for i in range(4):
            position = [
                self.width / 2 * (i % 2) * (2 * (i % 3) - 1) + self.center[0],
                self.length / 2 * (i % 2 == 0) * ((i % 3) - 1) + self.center[1],
                self.wall_height / 2.0,
            ]

            content_dict = copy.deepcopy(self._content_dict)
            content_dict["geometry"]["position"] = position
            content_dict["geometry"]["width"] = (i % 2 == 1) * self.length + (
                i % 2 == 0
            ) * self.thickness
            content_dict["geometry"]["length"] = (i % 2 == 1) * self.thickness + (
                i % 2 == 0
            ) * self.width

            self._content_dicts.append(content_dict)

            wall = BoxObstacle(name=f"{self._name}-{i}", content_dict=content_dict)

            wall_segments.append(wall)

        return wall_segments
