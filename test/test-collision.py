import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv

import time

from pdmproject.environment import GateWall, PDMWorldCreator, PerimeterWall, Wall
from pdmproject.collision_checking import CollisionCheckRobot

world_plan = PDMWorldCreator()
world_plan.register(PerimeterWall((0, 0), 5, 7))
world_plan.register(GateWall((-0.25, 0), (2.5, 0), (1.25, 0)))
world_plan.register(Wall((-1, 1), (2, -3)))
world_plan.register(
    GateWall((-1, 1), (-2.5, 1), (-1.75, 1), extra_data={"rgba": [0.9, 0.4, 0.4, 1.0]})
)

N_STEPS = 1000

# Create Robot of type CollisionCheckRobot to allow for collision checking
robots = [
    CollisionCheckRobot(urdf="../demo/urdf/mobileManipulator.urdf"), #Fix Relative paths to urdf models
]
env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=True,
)

world_plan.insert_into(env)
env.reconfigure_camera(5.0, 0.0, -90.01, (0, 0, 0))

pos0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
vel0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

ob = env.reset(pos=pos0, vel=vel0)

time_1 = time.time()

for _ in range(N_STEPS):
    x = np.random.randint(-25, 25) / 10
    y = np.random.randint(-35, 35) / 10
    robots[0].check_if_colliding(np.array([x, y, 0.0, 0.0, 0.0, 0.0, 0.0]))
    time.sleep(.01)

time_2 = time.time()

print("Checking time: ", time_2 - time_1)

env.close()
