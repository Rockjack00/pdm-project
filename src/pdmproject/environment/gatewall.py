import copy
from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar, Optional

import numpy as np
from mpscenes.obstacles.box_obstacle import BoxObstacle
from pybullet_utils.transformations import quaternion_from_euler

from .wall import Wall


@dataclass
class GateWall(Wall):
    gate_point: tuple[float, float]
    _: KW_ONLY
    gate_height: float = 1.5
    gate_width: float = 1.0
    _content_dicts: Optional[list[dict]] = field(init=False, default=None)

    DEFAULT_NAME_TEMPLATE: ClassVar[str] = "gatewall-"

    def _generate_wall_segments(self) -> list[BoxObstacle]:
        if not self.is_registered:
            raise RuntimeWarning(
                "The wall is not registered yet, if no simulation_name was provided the name will be None"
            )

        start_vec = np.array(self.start_point)
        gate_vec = np.array(self.gate_point)
        end_vec = np.array(self.end_point)

        wall_vec = end_vec - start_vec
        total_wall_length = np.linalg.norm(wall_vec)
        # wall_center = wall_vec / 2 + start_vec

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
