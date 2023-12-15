from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from mpscenes.obstacles.box_obstacle import BoxObstacle

from .wall import Wall


@dataclass
class GateWall(Wall):
    door_point: Tuple[float, float]
    gate_height: float = 1.5
    gate_width: float = 1.0

    def _generate_wall_segments(self) -> List[BoxObstacle]:
        pass
