import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.scene_examples.goal import goal1
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common.reward import Reward
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from pdmproject.environment import generate_environment

np.random.seed(99000)

world_plan = generate_environment(width=10, length=10)

N_STEPS = 1000


class InverseDistanceDenseReward(Reward):
    def calculate_reward(self, observation: dict) -> float:
        goal = observation["robot_0"]["FullSensor"]["goals"][1]["position"]
        position = observation["robot_0"]["joint_state"]["position"]
        return 1.0 / np.linalg.norm(goal - position)


robots = [
    GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
]
env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=True,
)

# Set it slightly below 90 to prevent rendering issues
env.reconfigure_camera(5, 0, -89.9, (0, 0, 0))

env.add_goal(goal1)

world_plan.insert_into(env)

# %%
sensor = FullSensor(["position"], ["position", "size"], variance=0.0)
env.add_sensor(sensor, [0])
env.set_spaces()
env.set_reward_calculator(InverseDistanceDenseReward())
defaultAction = np.array([0.5, -0.0, 0.0])
pos0 = np.array([0.0, 1.0, 0.0])
vel0 = np.array([1.0, 0.0, 0.0])

# %%
ob = env.reset(pos=pos0, vel=vel0)
env.shuffle_goals()
print(f"Initial observation : {ob}")
gain_action = 1

history = []
ob = ob[0]
for _ in range(N_STEPS):
    # Simple p-controller for goal reaching
    goal = ob["robot_0"]["FullSensor"]["goals"][1]["position"]
    robot_position = ob["robot_0"]["joint_state"]["position"]
    action = gain_action * (goal - robot_position)
    ob, reward, done, _, info = env.step(action)
    print(f"Reward : {reward}")
    # In observations, information about obstacles is stored in ob['obstacleSensor']
    history.append(ob)

env.close()
