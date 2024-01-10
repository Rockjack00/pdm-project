#!/usr/bin/env -S sh -c '"`dirname $0`/../venv/bin/python3" "$0" "$@"'
import argparse
from pathlib import Path
import time

import numpy as np

from urdfenvs.urdf_common import UrdfEnv

from pdmproject.collision_checking import CollisionCheckRobot
from pdmproject.environment import generate_environment
from pdmproject.planning.rrtstar import RRTStar
from pdmproject.sampling import SamplerBase, SimpleSampler


def create_std_sampler(lower_boud, upper_bound) -> SamplerBase:
    # TODO: ADD ARGS
    return SimpleSampler(lower_boud, upper_bound)


def create_ns_sampler() -> SamplerBase:
    raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rrt-star-bench-single-run")
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=1000,
        # TODO: HELP
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set the random seed for this test to a fixed value for reproducability.",
    )
    parser.add_argument(
        "-NS",
        "--Nullspace-Sampler",
        action="store_const",
        const=create_ns_sampler,
        default=create_std_sampler,
        dest="sampler",
        help="Use the Nullspace Sampler, which excludes the Nullspace of collisions from future samples.",
    )
    parser.add_argument("-vw", "--visualize-world", action="store_true")
    parser.add_argument("-vp", "--visualize-path", action="store_true")
    parser.add_argument("-vs", "--visualize-sim", action="store_true")

    parser.add_argument("--URDF", type=Path)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # TODO: ADD PARAMETER:
    length = 6
    width = 10


    world, centers = generate_environment(
        n_rooms=6,
        width=width,
        length=length,
        gate_height_bounds=(0.9, 1.75),
        minimum_gate_width=0.6,  # TEMP FOR VARIFICATION
        get_room_centers=True,
        inspect=args.visualize_world,
    )

    lower_bound = (
        -width/2,
        -length/2,
        -np.pi,
        -np.pi / 2,
        -2 / 3 * np.pi,
        -2 / 3 * np.pi,
        0.0,
    )

    upper_bound = (
        width/2,
        length/2,
        np.pi,
        np.pi / 2,
        2 / 3 * np.pi,
        2 / 3 * np.pi,
        2 * np.pi,
    )

    sampler = args.sampler(lower_bound, upper_bound)

    # Select start and endpoint
    point_idxs = np.random.choice(range(centers.shape[0]), 2, replace=False)  # type: ignore

    start_pose = np.zeros(7)
    start_pose[:2] = centers[point_idxs[0], :]

    goal_pose = np.random.uniform(lower_bound, upper_bound) # type: ignore
    goal_pose[:2] = centers[point_idxs[1], :]

    # Select URDF based on argument or from Repo Location
    package_path = Path(__file__).parents[1]

    if args.URDF is not None:
        urdf_arg: Path = args.urdf
        if urdf_arg.exists():
            urdf_path = urdf_arg
        else:
            urdf_path = package_path / urdf_arg
            assert urdf_path.exists(), "The URDF file could not be found."
    else:
        urdf_path = package_path / "demo" / "urdf" / "mobileManipulator.urdf"

    robots = [
        CollisionCheckRobot(urdf=str(urdf_path)),
    ]

    # Environment initialisation

    env = UrdfEnv(
        dt=0.01,
        robots=robots,  # type: ignore
        render=False,
    )

    world.insert_into(env)
    env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))

    ob = env.reset(pos=start_pose, vel=np.zeros_like(start_pose))


    rrt_star = RRTStar(
        robot=robots[0],
        start=start_pose,
        goal=goal_pose,
        sampler=sampler,
        step_size=0.1,
        max_iter=args.max_iterations,
        radius=5,
    )

    rrt_star.plan()
    env.close()
    
    if args.visualize_path:
        rrt_star.plot_path()

    if args.visualize_sim:
        # Path in simulation

        env = UrdfEnv(
            dt=0.01,
            robots=robots,  # type: ignore
            render=True,
        )

        world.insert_into(env)
        env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))
        ob = env.reset(pos=start_pose, vel=np.zeros_like(start_pose))

        smooth_path = rrt_star.get_smoother_path(path_step_size=0.01)

        while True:
            for pose in smooth_path:
                robots[0].set_pose(pose=pose)
                time.sleep(0.01)

        env.close()
