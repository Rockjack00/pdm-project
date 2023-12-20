import copy
from typing import ClassVar, Optional

from matplotlib.axes import Axes
from mpscenes.obstacles.box_obstacle import BoxObstacle
import numpy as np

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

        self._content_dict_base = {
            "type": "box",
            "geometry": {
                "height": self.wall_height,
            },
        }

        if self.extra_data is not None:
            self._content_dict_base.update(self.extra_data)

        if "rgba" not in self._content_dict_base:
            self._content_dict_base["rgba"] = [1.0, 1.0, 1.0, 1.0]

    def _generate_content_dicts(self, regenerate: bool = False):
        if not (self._content_dicts is None or regenerate):
            return None

        self._content_dicts = []

        for i in range(4):
            position = [
                self.width / 2 * (i % 2) * (2 * (i % 3) - 1) + self.center[0],
                self.length / 2 * (i % 2 == 0) * ((i % 3) - 1) + self.center[1],
                self.wall_height / 2.0,
            ]

            content_dict = copy.deepcopy(self._content_dict_base)
            content_dict["geometry"]["position"] = position
            content_dict["geometry"]["width"] = (i % 2 == 1) * self.length + (
                i % 2 == 0
            ) * self.thickness
            content_dict["geometry"]["length"] = (i % 2 == 1) * self.thickness + (
                i % 2 == 0
            ) * self.width

            self._content_dicts.append(content_dict)

        self._content_dict = self._content_dicts[0]

    def _generate_content_dict(self, regenerate: bool = False):
        self._generate_content_dicts(regenerate)

    def _generate_wall_segments(self, regenerate: bool = True) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )

        wall_segments = []
        self._generate_content_dicts(regenerate=regenerate)

        for idx, content_dict in enumerate(self._content_dicts):
            wall = BoxObstacle(name=f"{self._name}-{idx}", content_dict=content_dict)

            wall_segments.append(wall)

        return wall_segments

    def _plot2d(self, ax: Axes):
        for i in range(4):
            x = self.width / 2 * (i % 2) * (2 * (i % 3) - 1) + self.center[0]
            y = self.length / 2 * (i % 2 == 0) * ((i % 3) - 1) + self.center[1]

            dx = (i % 2 == 0) * self.width / 2
            dy = self.length * (i % 2 == 1) / 2


            color = self.color

            if (np.array(color) == 1.0).all():
                color = [0, 0, 0, 1]

            ax.plot(
                [x - dx, x + dx],
                [y - dy, y + dy],
                color=color,
            )
