import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from scipy.spatial import HalfspaceIntersection, Voronoi, voronoi_plot_2d

from pdmproject.environment.bounded_voronoi import voronoi
from pdmproject.environment.wall import Wall

from .pdmworld import PDMWorldCreator
from .perimeterwall import PerimeterWall


# TODO: ADD ARGUMENTS
def generate_environment(
    n_rooms: int = 10,
    width: float = 10,
    length: float = 10,
    inspect: bool = False,
    minimum_gate_width: float = 0.75,
) -> PDMWorldCreator:
    # Generate random points
    points = np.random.rand(n_rooms, 2)

    # Rescale the points to the world size
    points[:, 0] *= width
    points[:, 1] *= length

    points -= np.array((width, length)) / 2

    world = PDMWorldCreator(
        (
            # TODO: EXTRA PARAMS
            PerimeterWall((0, 0), width, length),
        )
    )

    vor = voronoi(points, [-width / 2, width / 2, -length / 2, length / 2])

    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]

        # Check if doors are possible
        door_possible = False
        for idx in range(vertices.shape[0] - 1):
            door_possible = door_possible or (
                (
                    not (
                        on_border(vertices[idx], width, length)
                        and on_border(vertices[idx + 1], width, length)
                        and np.count_nonzero(vertices[idx] - vertices[idx + 1]) == 1
                    )
                )
                and np.linalg.norm(vertices[idx] - vertices[idx + 1])
                >= minimum_gate_width
            )

        if not door_possible:
            continue

        for idx in range(vertices.shape[0] - 1):
            if (
                on_border(vertices[idx], width, length)
                and on_border(vertices[idx + 1], width, length)
                and np.count_nonzero(np.round(vertices[idx] - vertices[idx + 1])) == 1
            ):
                continue

            print(vertices[idx], vertices[idx+1], vertices[idx]- vertices[idx+1])
            wall = Wall(vertices[idx], vertices[idx + 1])
            print(wall.wall_length)
            # TODO: UPGRADE WALL IF HOLE CAN BE MADE.
            world.register(wall)

    if inspect:
        fig = plt.figure()
        ax = plt.subplot()
        
        ax.scatter(points[:, 0], points[:, 1])
        
        world.plot2d(ax, fig)
        plt.show()

    return world


def on_border(point: npt.ArrayLike, width: float, length: float) -> bool:
    return (np.abs(point) == (np.array((width, length)) / 2)).any()
