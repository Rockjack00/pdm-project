import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv

import time

from pdmproject.environment import GateWall, PDMWorldCreator, PerimeterWall, Wall
from pdmproject.collision_checking import CollisionCheckRobot
from pdmproject.planning import RRTStar
from pdmproject.sampling import SimpleSampler

# World Building

world_plan = PDMWorldCreator()
world_plan.register(PerimeterWall(center=(0, 0), width=10, length=6))
world_plan.register(GateWall(start_point=(2.0, 3.0), end_point=(2.0, -3.0), gate_point=(2.0, 1.5), gate_height=1.5, gate_width=0.6))
world_plan.register(GateWall(start_point=(-2.0, 3.0), end_point=(-2.0, -3.0), gate_point=(-2.0, -1.5), gate_height=1.5, gate_width=0.6))


# Robot initialisation of type CollisionCheckRobot -> allows for collision checking

robots = [
    CollisionCheckRobot(urdf="../demo/urdf/mobileManipulator.urdf"), #Fix Relative paths to urdf models
]


# Environment initialisation

env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=False,
)

world_plan.insert_into(env)
env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))

pos0 = np.array([-4.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.1])
vel0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

ob = env.reset(pos=pos0, vel=vel0)

# Planning

start_point = [-4.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.1]
goal_point = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

search_area = (-5, 5, 
            -3, 3, 
            -np.pi, np.pi, 
            -1/2 * np.pi, 1/2 * np.pi, 
            -2/3 * np.pi, 2/3 * np.pi, 
            -2/3 * np.pi, 2/3 * np.pi, 
            0, 2 * np.pi)

sampler = SimpleSampler(
    lower_bound = tuple(i for i in search_area[::2]),
    upper_bound = tuple(i for i in search_area[1::2])
)

sampler.register_goal_hack(goal_point)

rrt_star = RRTStar(robot=robots[0], start=start_point, goal=goal_point, sampler=sampler, step_size=0.1, max_iter=1000, radius=5)
rrt_star.plan()
env.close()
rrt_star.plot_path()


# Path in simulation 

env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=True,
)

world_plan.insert_into(env)
env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))
ob = env.reset(pos=pos0, vel=vel0)

smooth_path = rrt_star.get_smoother_path(path_step_size=0.01)

while True:
    for pose in smooth_path:
        robots[0].set_pose(pose=pose)
        time.sleep(0.01)
        
env.close()