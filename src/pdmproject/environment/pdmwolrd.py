from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from typing import Iterable, List, Optional
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from pdmproject.environment.wall import Wall

@dataclass
class PDMWorldCreator:
    walls: InitVar[Optional[Iterable[Wall]]] = None
    _walls: list[Wall] = field(init=False, default_factory=list)
    _wall_type_counts: defaultdict[type(Wall), int] = field(init=False),
    _wall_name_set: set[str] = field(init=False, default_factory=set)

    def __post_init__(self, walls: Optional[Iterable[Wall]]) -> None:
        self._wall_type_counts = defaultdict(int)
        if walls is not None:
            for wall in walls:
                self.register(wall)
    
    def register(self, wall: Wall) -> None:
        assert (not wall.is_registered), "This wall is already registered"

        number = self._wall_type_counts[type(wall)]
        if wall._register(number, self._wall_name_set):
            self._wall_type_counts[type(wall)] += 1
        
        self._walls.append(wall)

        assert wall._name in self._wall_name_set

    def insert_into(self, env: UrdfEnv) -> None:
        for wall in self._walls:
            for wall_segment in wall._generate_wall_segments():
                env.add_obstacle(wall_segment)