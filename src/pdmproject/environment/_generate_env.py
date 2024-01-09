from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pdmproject.environment._bounded_voronoi import voronoi
from pdmproject.environment.gatewall import upgrade_to_gatewall
from pdmproject.environment.wall import Wall

from .pdmworld import PDMWorldCreator
from .perimeterwall import PerimeterWall


@overload
def generate_environment(
    n_rooms: int = 10,
    width: float = 10,
    length: float = 10,
    thickness: float = 0.1,
    wall_height: float = 3.5,
    minimum_gate_width: float = 0.75,
    maximum_gate_width_ratio: float = 0.8,
    gate_height_bounds: tuple[float, float] = (0.6, 2.5),
    *,
    get_room_centers: Literal[True],
    inspect: bool = False,
) -> tuple[PDMWorldCreator, npt.NDArray[np.float64]]:
    ...


@overload
def generate_environment(
    n_rooms: int = 10,
    width: float = 10,
    length: float = 10,
    thickness: float = 0.1,
    wall_height: float = 3.5,
    minimum_gate_width: float = 0.75,
    maximum_gate_width_ratio: float = 0.8,
    gate_height_bounds: tuple[float, float] = (0.6, 2.5),
    *,
    get_room_centers: Literal[False],
    inspect: bool = False,
) -> PDMWorldCreator:
    ...


@overload
def generate_environment(
    n_rooms: int = 10,
    width: float = 10,
    length: float = 10,
    thickness: float = 0.1,
    wall_height: float = 3.5,
    minimum_gate_width: float = 0.75,
    maximum_gate_width_ratio: float = 0.8,
    gate_height_bounds: tuple[float, float] = (0.6, 2.5),
    *,
    get_room_centers: bool = False,
    inspect: bool = False,
) -> PDMWorldCreator:
    ...


# TODO: ADD ARGUMENTS
def generate_environment(
    n_rooms: int = 10,
    width: float = 10,
    length: float = 10,
    thickness: float = 0.1,
    wall_height: float = 3.5,
    minimum_gate_width: float = 0.75,
    maximum_gate_width_ratio: float = 0.8,
    gate_height_bounds: tuple[float, float] = (0.6, 2.5),
    *,
    get_room_centers: bool = False,
    inspect: bool = False,
) -> PDMWorldCreator | tuple[PDMWorldCreator, npt.NDArray[np.float64]]:
    """Generate a random environment.

    Args:
        n_rooms (int, optional): The amount of points the voronoid is based on. This roughly corresponds with the amount of rooms. Defaults to 10.
        width (float, optional): The width of the world. Defaults to 10.
        length (float, optional): The length of the world. Defaults to 10.
        thickness (float, optional): The thickness of the walls in the world. Defaults to 0.1.
        wall_height (float, optional): The heigth of the walls in the world. Defaults to 3.5.
        minimum_gate_width (float, optional): The minimum gate width that can be generated. Defaults to 0.75.
        maximum_gate_width_ratio (float, optional): The maximum fraction of gate width to total wall length. Defaults to 0.8.
        gate_height_bounds (tuple[float, float], optional): The lower and upper bounds of the gate height. Defaults to (0.6, 2.5).
        get_room_centers (bool, optional): If true the random points on which the voronoid is based are returned. Defaults to False.
        inspect (bool, optional): Enables plotting of a 2D representation of the generated world. Defaults to False.

    Returns:
        PDMWorldCreator | tuple[PDMWorldCreator, npt.NDArray[np.float64]]: The generated world and if get_room_centers is enabled the points with which the voronoid is generated.
    """
    assert (
        gate_height_bounds[0] > 0
    ), "The minimum gate height should be larger than 0.0"
    assert (
        gate_height_bounds[1] < wall_height
    ), "The upper bound for the gate height should be smaller than the wall height"

    # Generate random points
    points = np.random.rand(n_rooms, 2)

    # Rescale the points to the world size
    points[:, 0] *= width
    points[:, 1] *= length

    points -= np.array((width, length)) / 2

    world = PDMWorldCreator(
        (
            # TODO: EXTRA PARAMS
            PerimeterWall(
                (0, 0), width, length, wall_height=wall_height, thickness=thickness
            ),
        )
    )

    vor = voronoi(points, [-width / 2, width / 2, -length / 2, length / 2])

    # We need to keep track of the added walls, since regions can share walls.
    wall_set: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]

        # Check if doors are possible
        door_possible = False
        for idx in range(vertices.shape[0] - 1):
            wall_length: float = np.linalg.norm(vertices[idx] - vertices[idx + 1])  # type: ignore
            door_possible = door_possible or (
                (
                    not (
                        on_border(vertices[idx], width, length)
                        and on_border(vertices[idx + 1], width, length)
                        and np.count_nonzero(vertices[idx] - vertices[idx + 1]) == 1
                    )
                )
                # Check if a gate will fit
                and wall_length >= minimum_gate_width
                and (maximum_gate_width_ratio * (wall_length - thickness * 2))
                >= minimum_gate_width
            )

        # Skip this region if a gate cannot be made to enter it
        if not door_possible:
            continue

        for idx in range(vertices.shape[0] - 1):
            # Check if the wall is on top of the perimeter and skip it if it is.
            if (
                on_border(vertices[idx], width, length)
                and on_border(vertices[idx + 1], width, length)
                and np.count_nonzero(np.round(vertices[idx] - vertices[idx + 1])) == 1
            ):
                continue

            if (tuple(vertices[idx]), tuple(vertices[idx + 1])) in wall_set:
                continue

            wall = Wall(
                vertices[idx],
                vertices[idx + 1],
                wall_height=wall_height,
                thickness=thickness,
            )

            wall_set.add((tuple(vertices[idx]), tuple(vertices[idx + 1])))  # type: ignore
            wall_set.add((tuple(vertices[idx + 1]), tuple(vertices[idx])))  # type: ignore

            if (  # Only add a gate if it fits on the wall
                wall.wall_length >= minimum_gate_width
                and (maximum_gate_width_ratio * (wall.wall_length - thickness * 2))
                >= minimum_gate_width
            ):
                gate_width = np.random.uniform(
                    minimum_gate_width,
                    maximum_gate_width_ratio * (wall.wall_length - thickness * 2),
                )

                gate_height = np.random.uniform(*gate_height_bounds)
                gate_position_rate = np.random.uniform(
                    (gate_width / 2 + thickness) / wall.wall_length,
                    (wall.wall_length - gate_width / 2 - thickness) / wall.wall_length,
                )

                wall = upgrade_to_gatewall(
                    wall,
                    gate_point_rate=gate_position_rate,
                    gate_width=gate_width,
                    gate_height=gate_height,
                )
            world.register(wall)

    if inspect:
        fig = plt.figure()
        ax = plt.subplot()

        ax.scatter(points[:, 0], points[:, 1])

        world.plot2d(ax, fig)
        plt.show()

    if get_room_centers:
        return world, points

    return world


def on_border(point: npt.ArrayLike, width: float, length: float) -> bool:
    """Check if a point is on the border of an origin centered perimeter.

    Args:
        point (npt.ArrayLike): A 2D point to check. (dtype=float)
        width (float): The width of the perimeter.
        length (float): The length of the perimeter.

    Returns:
        bool: if the point is on the border or not.
    """
    return (np.abs(point) == (np.array((width, length)) / 2)).any()
