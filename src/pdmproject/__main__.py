#!/usr/bin/env python
import argparse
import time
from itertools import pairwise
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import numpy.typing as npt

from urdfenvs.urdf_common import UrdfEnv

from pdmproject.collision_checking import CollisionCheckRobot
from pdmproject.environment import PDMWorldCreator, generate_environment
from pdmproject.planning.rrtstar import RRTStar
from pdmproject.sampling import SamplerBase, SimpleSampler


def create_std_sampler(lower_boud, upper_bound) -> SamplerBase:
    # TODO: ADD ARGS
    return SimpleSampler(lower_boud, upper_bound)


def create_ns_sampler() -> SamplerBase:
    raise NotImplementedError()


def add_rrt_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("RRT* options")
    group.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set the random seed for this test to a fixed value for reproducability. (default: %(default)s)",
    )
    group.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        metavar="ITER",
        default=1000,
        help="The (maximum) amount of iterations the RRT* algorithm will run for. (default: %(default)s)",
    )
    group.add_argument(
        "-NS",
        "--Nullspace-Sampler",
        action="store_const",
        const=create_ns_sampler,
        default=create_std_sampler,
        dest="sampler",
        help="Use the Nullspace Sampler, which excludes the Nullspace of collisions from future samples. (default: Simple Sampler)",
    )
    group.add_argument(
        "-r",
        "--radius",
        type=float,
        default=5.0,
        help="The connecting radius for nodes. Used as the initial radius at n=2 if shrinking-radius is enabled. This value must be positive and of a large enough size (atleast max(width,length)/4). (default: %(default)s)",
    )
    group.add_argument(
        "-sr",
        "--shrinking-radius",
        action="store_true",
        help="Enable to let the connection radius shrink with the number of sampled nodes. (default: %(default)s)",
    )
    return group


def add_world_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("World options")
    group.add_argument(
        "-nr",
        "--num-rooms",
        type=int,
        default=6,
        dest="n_rooms",
        help="The amount of room points to generate (Roughly corresponds to the amount of rooms), atleast 2 (default: %(default)s)",
    )
    group.add_argument(
        "-w",
        "--width",
        type=float,
        default=10.0,
        help="The width of the world. It must be larger or equal to 1.0 (default: %(default)s)",
    )
    group.add_argument(
        "-l",
        "--length",
        type=float,
        default=6.0,
        help="The length of the world. It must be larger or equal to 1.0 (default: %(default)s)",
    )
    return group


def add_visualization_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Visualization options")
    group.add_argument(
        "-vw", "--visualize-world", action="store_true", help="Show the 2D world plan."
    )
    group.add_argument(
        "-vp", "--visualize-path", action="store_true", help="Show a 2D planned path."
    )
    group.add_argument(
        "-vs",
        "--visualize-sim",
        action="store_true",
        help="Show the planned path being executed in the PyBullet window.",
    )
    group.add_argument(
        "--matplotlib-backend",
        metavar="MPL_BACKEND",
        type=str,
        dest="mpl_backend",
        default="TkAgg",
        help="The matploblib backend for plotting the path and world. (default: %(default)s)",
    )
    return group


def create_world_and_bounds(
    width: float, length: float, n_rooms: int, visualize_world: bool
) -> tuple[
    PDMWorldCreator,
    npt.NDArray[np.float64],
    tuple[float, float, float, float, float, float, float],
    tuple[float, float, float, float, float, float, float],
]:
    assert (
        length >= 1.0
    ), "The world should be larger than 1.0 meters in the length direction."
    assert (
        width >= 1.0
    ), "The world should be larger than 1.0 meters in the width direction."

    world, centers = generate_environment(
        n_rooms=n_rooms,
        width=width,
        length=length,
        wall_height=2.0,
        gate_height_bounds=(0.9, 1.75),
        minimum_gate_width=0.6,  # TEMP FOR VARIFICATION
        get_room_centers=True,
        inspect=visualize_world,
    )

    lower_bound = (
        -width / 2,
        -length / 2,
        -np.pi,
        -np.pi / 2,
        -2 / 3 * np.pi,
        -2 / 3 * np.pi,
        0.0,
    )

    upper_bound = (
        width / 2,
        length / 2,
        np.pi,
        np.pi / 2,
        2 / 3 * np.pi,
        2 / 3 * np.pi,
        2 * np.pi,
    )

    return world, centers, lower_bound, upper_bound


def get_urdf_path(urdf: Optional[Path]) -> Path:
    # Select URDF based on argument or from Repo Location
    repo_path = Path(__file__).parents[2]

    if urdf is not None:
        if urdf.exists():
            return urdf
        else:
            urdf_path = repo_path / urdf
            assert urdf_path.exists(), "The URDF file could not be found."
            return urdf_path
    else:
        return repo_path / "demo" / "urdf" / "mobileManipulator.urdf"


def plan_rrt_star(
    robots: list[CollisionCheckRobot],
    world: PDMWorldCreator,
    sampler: SamplerBase,
    start_pose: npt.NDArray[np.float64],
    goal_pose: npt.NDArray[np.float64],
    max_iterations: int,
    radius: float,
    shrinking_radius: bool,
) -> RRTStar:
    assert (
        radius >= max(sampler.upper_bound[0], sampler.upper_bound[1]) / 2
    ), "The radius must be larger than or equal to 1/4 the largest world size (width or length)"

    # Environment initialisation
    env = UrdfEnv(
        dt=0.01,
        robots=robots,  # type: ignore
        render=False,
    )

    world.insert_into(env)

    _ = env.reset(pos=start_pose, vel=np.zeros_like(start_pose))

    rrt_star = RRTStar(
        robot=robots[0],
        start=start_pose,
        goal=goal_pose,
        sampler=sampler,
        step_size=0.1,
        max_iter=max_iterations,
        radius=radius,
        shrinking_radius=shrinking_radius,
    )

    rrt_star.plan()
    env.close()

    return rrt_star


def visualize_rrt_path(
    robots: list[CollisionCheckRobot],
    world: PDMWorldCreator,
    rrt_star: RRTStar,
    start_pose: npt.NDArray[np.float64],
):
    env = UrdfEnv(
        dt=0.01,
        robots=robots,  # type: ignore
        render=True,
    )

    world.insert_into(env)
    env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))
    _ = env.reset(pos=start_pose, vel=np.zeros_like(start_pose))

    smooth_path = rrt_star.get_smoother_path(path_step_size=0.01)

    while True:
        for pose in smooth_path:
            robots[0].set_pose(pose=pose)
            time.sleep(0.01)
    env.close()


def visualize_rrt_paths(
    robots: list[CollisionCheckRobot],
    world: PDMWorldCreator,
    rrt_stars: list[RRTStar],
    poses: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
):
    env = UrdfEnv(
        dt=0.01,
        robots=robots,  # type: ignore
        render=True,
    )

    world.insert_into(env)
    env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))
    _ = env.reset(pos=poses[0][0], vel=np.zeros_like(poses[0][0]))

    smooth_path = np.concatenate(
        [rrt_star.get_smoother_path(path_step_size=0.01) for rrt_star in rrt_stars]
    )

    while True:
        for pose in smooth_path:
            robots[0].set_pose(pose=pose)
            time.sleep(0.01)
    env.close()


def main_single_run():
    parser = argparse.ArgumentParser("rrt-star-bench-single-run")
    # RRT* Options
    add_rrt_arguments(parser)

    # World options
    add_world_arguments(parser)

    # Visualization Options
    add_visualization_arguments(parser)

    # Simulation Options
    parser.add_argument(
        "--URDF", type=Path, help="An alternative path to the URDF file for the robot."
    )

    args = parser.parse_args()

    matplotlib.use(args.mpl_backend)

    np.random.seed(args.seed)

    assert args.n_rooms >= 2, "There should atleast be 2 rooms."

    world, centers, lower_bound, upper_bound = create_world_and_bounds(
        args.width, args.length, args.n_rooms, args.visualize_world
    )

    sampler = args.sampler(lower_bound, upper_bound)

    # Select start and endpoint
    point_idxs = np.random.choice(range(centers.shape[0]), 2, replace=False)  # type: ignore

    start_pose = np.zeros(7)
    start_pose[:2] = centers[point_idxs[0], :]

    goal_pose = np.random.uniform(lower_bound, upper_bound)  # type: ignore
    goal_pose[:2] = centers[point_idxs[1], :]

    urdf_path = get_urdf_path(args.URDF)

    robots = [
        CollisionCheckRobot(urdf=str(urdf_path)),
    ]

    # Plan the path using RRT*
    rrt_star = plan_rrt_star(
        robots,
        world,
        sampler,
        start_pose,
        goal_pose,
        args.max_iterations,
        args.radius,
        args.shrinking_radius,
    )

    print(rrt_star.has_found_path())
    print("# iterations @ Path 1:  ", rrt_star.num_iter_till_first_path)
    print("explored nodes @ Path 1:", rrt_star.explored_nodes_till_first_path)
    print("# iterations @ Path X:", rrt_star.num_iter)
    print("explored nodes @ Path X:", rrt_star.explored_nodes)

    if args.visualize_path:
        rrt_star.plot_path()

    if args.visualize_sim:
        visualize_rrt_path(robots, world, rrt_star, start_pose)


def main_multi_run():
    parser = argparse.ArgumentParser("rrt-star-bench-multi-run")
    # RRT* Options
    add_rrt_arguments(parser)

    # World options
    add_world_arguments(parser).add_argument(
        "-N",
        "--n-goals",
        type=int,
        default=3,
        help="The amount of goal point the robot must visit. This must be atleast 2. (default: %(default)s)",
    )

    # Visualization Options
    add_visualization_arguments(parser)

    # Simulation Options
    parser.add_argument(
        "--URDF", type=Path, help="An alternative path to the URDF file for the robot."
    )

    args = parser.parse_args()

    matplotlib.use(args.mpl_backend)

    np.random.seed(args.seed)

    assert args.n_rooms >= 2, "There should atleast be 2 rooms."
    assert args.n_rooms >= args.n_goals, "There should more rooms than goals"

    world, centers, lower_bound, upper_bound = create_world_and_bounds(
        args.width, args.length, args.n_rooms, args.visualize_world
    )

    sampler = args.sampler(lower_bound, upper_bound)

    # Select start and endpoint
    point_idxs = np.random.choice(range(centers.shape[0]), args.n_goals, replace=False)  # type: ignore

    goal_sets = []

    prev_pose = np.zeros(7)
    prev_pose[:2] = centers[point_idxs[0], :]

    for idx_s, idx_g in pairwise(point_idxs):
        assert (prev_pose[:2] == centers[idx_s, :]).all()

        goal_pose = np.random.uniform(lower_bound, upper_bound)  # type: ignore
        goal_pose[:2] = centers[idx_g, :]

        goal_sets.append((prev_pose, goal_pose))
        prev_pose = goal_pose

    urdf_path = get_urdf_path(args.URDF)

    robots = [
        CollisionCheckRobot(urdf=str(urdf_path)),
    ]

    # Plan the path using RRT*
    rrt_stars = [
        plan_rrt_star(
            robots,
            world,
            sampler,
            start_pose,
            goal_pose,
            args.max_iterations,
            args.radius,
            args.shrinking_radius,
        )
        for start_pose, goal_pose in goal_sets
    ]

    if args.visualize_path:
        for idx, rrt_star in enumerate(rrt_stars):
            rrt_star.plot_path(block=(idx == (args.n_goals - 2)))

    if args.visualize_sim:
        visualize_rrt_paths(robots, world, rrt_stars, goal_sets)


if __name__ == "__main__":
    main_single_run()
