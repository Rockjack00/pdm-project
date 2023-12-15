from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from mpscenes.obstacles.box_obstacle import BoxObstacle
from pybullet_utils.transformations import quaternion_from_euler
from urdfenvs.urdf_common.urdf_env import UrdfEnv


@dataclass
class Wall:
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    thickness: float = 0.1
    wall_height: float = 2.0
    extra_data: Optional[dict] = None
    # TODO: name_template must be different
    # TODO: Maybe use init var
    name_template: str = "wall-"

    def _generate_wall_segments(self) -> List[BoxObstacle]:
        start_vec = np.array(self.start_point)
        end_vec = np.array(self.end_point)

        wall_vec = end_vec - start_vec
        wall_lenght = np.linalg.norm(wall_vec)
        wall_center = wall_vec / 2 + start_vec

        position = wall_center.copy()
        position[2] = self.wall_height / 2.0

        orientation = quaternion_from_euler(-np.arctan(wall_vec[1] / wall_vec[0]), 0, 0)

        content_dict = {
            "type": "box",
            "geometry": {
                "position": position.tolist,
                "width": self.thickness,
                "length": wall_lenght,
                "height": self.wall_height,
                "orientation": orientation.tolist(),
            },
        }

        content_dict.update(self.extra_data)

        self._content_dict = content_dict

        return [
            BoxObstacle(
                # FIXME: BUILD NAME
                name=self.name_template,
                content_dict=content_dict,
            )
        ]
