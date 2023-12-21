import copy
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes
from mpscenes.obstacles.box_obstacle import BoxObstacle
from pybullet_utils.transformations import quaternion_from_euler

from .wall import Wall


def upgrade_to_gatewall(
    wall: Wall,
    gate_point_rate: float = 0.5,
    gate_width: float = 1.0,
    gate_height: float = 1.5,
) -> "GateWall":
    """Upgrade a normal Wall to a GateWall

    Args:
        wall (Wall): The original Wall to upt a gate in
        gate_point_rate (float, optional): The rate allong the wall where the gate should be placed. Defaults to 0.5.
        gate_width (float, optional): The width of the gate. Defaults to 1.0.
        gate_height (float, optional): The height of the gate, it should be smaller than the wall_height of the original Wall. Defaults to 1.5.

    Returns:
        GateWall: A gate wall placed at the original wall position
    """
    assert (
        not wall.is_registered
    ), "Trying to upgrade a Wall, which is already registered"

    start_vec = np.asarray(wall.start_point)
    gate_point = (
        (np.asarray(wall.end_point) - start_vec) * gate_point_rate + start_vec
    ).astype(float)

    new_wall = GateWall(
        wall.start_point,
        wall.end_point,
        (float(gate_point[0]), float(gate_point[1])),
        thickness=wall.thickness,
        wall_height=wall.wall_height,
        extra_data=wall.extra_data,
        gate_height=float(gate_height),
        gate_width=float(gate_width),
    )

    return new_wall


@dataclass
class GateWall(Wall):
    gate_point: tuple[float, float]
    _: KW_ONLY
    gate_height: float = 1.5
    gate_width: float = 1.0

    DEFAULT_NAME_TEMPLATE: ClassVar[str] = "gatewall-"

    def _generate_content_dicts(self, regenerate: bool = False):
        if self._content_dict is None or regenerate:
            self._generate_content_dict()

        if self._content_dicts is None or regenerate:
            gate_vec = np.array(self.gate_point)
            end_vec = np.array(self.end_point)

            position_gate_top: list[float] = list(self.gate_point)
            position_gate_top.append(
                (self.wall_height - self.gate_height) / 2.0 + self.gate_height
            )

            gate_end_vec = gate_vec - end_vec
            gate_end_dist = np.linalg.norm(gate_end_vec)

            length_end_gate = gate_end_dist - self.gate_width / 2
            wall_center_end_gate_vec = (
                length_end_gate * (gate_end_vec) / gate_end_dist / 2 + end_vec
            )

            position_end_gate: list[float] = wall_center_end_gate_vec.tolist()
            position_end_gate.append(self.wall_height / 2.0)

            content_dict_start_gate = self._content_dict

            content_dict_top_gate = copy.deepcopy(content_dict_start_gate)
            content_dict_top_gate["geometry"]["position"] = position_gate_top
            content_dict_top_gate["geometry"]["length"] = self.gate_width
            content_dict_top_gate["geometry"]["height"] = (
                self.wall_height - self.gate_height
            )

            content_dict_end_gate = copy.deepcopy(content_dict_start_gate)
            content_dict_end_gate["geometry"]["position"] = position_end_gate
            content_dict_end_gate["geometry"]["length"] = float(length_end_gate)

            self._content_dicts = [
                content_dict_start_gate,
                content_dict_top_gate,
                content_dict_end_gate,
            ]

    def _generate_content_dict(self, regenerate: bool = False):
        if not (self._content_dict is None or regenerate):
            return None
        start_vec = np.array(self.start_point)
        gate_vec = np.array(self.gate_point)
        end_vec = np.array(self.end_point)

        wall_vec = end_vec - start_vec
        total_wall_length = np.linalg.norm(wall_vec)

        self._wall_length = total_wall_length

        gate_start_vec = gate_vec - start_vec
        gate_start_dist = np.linalg.norm(gate_start_vec)

        assert (
            np.abs(
                (gate_start_vec.dot(wall_vec)) / (gate_start_dist * total_wall_length)
            )
            >= 0.99999
        ), "The gate should be between the edges of the wall"

        assert (
            total_wall_length > self.gate_width
        ), "The wall length should be larger than the gate width"

        length_start_gate = gate_start_dist - self.gate_width / 2

        wall_center_start_gate_vec = (
            length_start_gate * gate_start_vec / gate_start_dist / 2 + start_vec
        )

        position_start_gate: list[float] = wall_center_start_gate_vec.tolist()
        position_start_gate.append(self.wall_height / 2.0)

        position_gate_top: list[float] = list(self.gate_point)
        position_gate_top.append(
            (self.wall_height - self.gate_height) / 2.0 + self.gate_height
        )

        gate_end_vec = gate_vec - end_vec
        gate_end_dist = np.linalg.norm(gate_end_vec)

        length_end_gate = gate_end_dist - self.gate_width / 2
        wall_center_end_gate_vec = (
            length_end_gate * (gate_end_vec) / gate_end_dist / 2 + end_vec
        )

        position_end_gate: list[float] = wall_center_end_gate_vec.tolist()
        position_end_gate.append(self.wall_height / 2.0)

        orientation = quaternion_from_euler(-np.arctan(wall_vec[1] / wall_vec[0]), 0, 0)

        content_dict_start_gate = {
            "type": "box",
            "geometry": {
                "position": position_start_gate,
                "width": self.thickness,
                "length": float(length_start_gate),
                "height": float(self.wall_height),
                "orientation": orientation.tolist(),
            },
        }

        if self.extra_data is not None:
            content_dict_start_gate.update(self.extra_data)

        if "rgba" not in content_dict_start_gate:
            content_dict_start_gate["rgba"] = [0.5, 0.5, 0.5, 1.0]

        self._content_dict = content_dict_start_gate

    def _generate_wall_segments(self, regenerate: bool = False) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )

        self._generate_content_dicts(regenerate=regenerate)

        content_dict_start_gate = self._content_dicts[0]
        content_dict_top_gate = self._content_dicts[1]
        content_dict_end_gate = self._content_dicts[2]

        return [
            BoxObstacle(
                name=f"{self._name}-start",
                content_dict=content_dict_start_gate,
            ),
            BoxObstacle(
                name=f"{self._name}-top",
                content_dict=content_dict_top_gate,
            ),
            BoxObstacle(
                name=f"{self._name}-end",
                content_dict=content_dict_end_gate,
            ),
        ]

    def _plot2d(self, ax: Axes):
        start_vec = np.array(self.start_point)
        end_vec = np.array(self.end_point)
        wall_vec = end_vec - start_vec
        total_wall_length = np.linalg.norm(wall_vec)

        unit_wall_vec = wall_vec / total_wall_length

        gate_correction = unit_wall_vec / 2 * self.gate_width

        # START
        ax.plot(
            [self.start_point[0], self.gate_point[0] - gate_correction[0]],
            [self.start_point[1], self.gate_point[1] - gate_correction[1]],
            color=self.color,
        )

        # Gate
        ax.plot(
            [
                self.gate_point[0] - gate_correction[0],
                self.gate_point[0] + gate_correction[0],
            ],
            [
                self.gate_point[1] - gate_correction[1],
                self.gate_point[1] + gate_correction[1],
            ],
            color=self.color,
            linestyle=":",
        )

        # END
        ax.plot(
            [self.gate_point[0] + gate_correction[0], self.end_point[0]],
            [self.gate_point[1] + gate_correction[1], self.end_point[1]],
            color=self.color,
        )
