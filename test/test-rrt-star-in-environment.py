import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv

import time

from pdmproject.environment import GateWall, PDMWorldCreator, PerimeterWall, Wall
from pdmproject.collision_checking import CollisionCheckRobot
from pdmproject.planning import Node, RRTStar

import matplotlib.pyplot as plt

def plot_rrt_star(rrt_star, path=[]):
    plt.figure(figsize=(10, 10))
    
    # Plot obstacles
    for obstacle in rrt_star.obstacle_list:
        plt.plot(obstacle[0], obstacle[1], 'sk', markersize=30)

    # Plot search area
    plt.plot([rrt_star.search_area[0], rrt_star.search_area[1], rrt_star.search_area[1], rrt_star.search_area[0], rrt_star.search_area[0]],
             [rrt_star.search_area[2], rrt_star.search_area[2], rrt_star.search_area[3], rrt_star.search_area[3], rrt_star.search_area[2]], 'k--')

    # Plot nodes and edges
    for node in rrt_star.node_list:
        if node.parent is not None:
            plt.plot([node.q1, node.parent.q1], [node.q2, node.parent.q2], 'bo-')

    #Plot path
    if path:
        path_q1, path_q2, _, _, _, _, _ = path
        plt.plot(path_q1, path_q2, 'r-', linewidth=2)

    plt.plot(rrt_star.start.q1, rrt_star.start.q2, 'go', markersize=10)
    plt.plot(rrt_star.goal.q1, rrt_star.goal.q2, 'ro', markersize=10)

    plt.title('RRT* Path Planning')
    plt.xlabel('q1-axis')
    plt.ylabel('q2-axis')
    plt.grid(True)
    plt.show()

world_plan = PDMWorldCreator()
world_plan.register(PerimeterWall(center=(0, 0), width=10, length=6))
world_plan.register(GateWall(start_point=(2.0, 3.0), end_point=(2.0, -3.0), gate_point=(2.0, 1.5), gate_height=1.5))
world_plan.register(GateWall(start_point=(-2.0, 3.0), end_point=(-2.0, -3.0), gate_point=(-2.0, -1.5), gate_height=1.5))



# Create Robot of type CollisionCheckRobot to allow for collision checking
robots = [
    CollisionCheckRobot(urdf="../demo/urdf/mobileManipulator.urdf"), #Fix Relative paths to urdf models
]
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
time_1 = time.time()

### PLANNING STUFF ###

start_point = (-4.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.1)
goal_point = (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

obstacles = []  # Coordinates of obstacles

search_area = (-5, 5, 
            -3, 3, 
            0, 2 * np.pi, 
            -1/2 * np.pi, 1/2 * np.pi, 
            -2/3 * np.pi, 2/3 * np.pi, 
            -2/3 * np.pi, 2/3 * np.pi, 
            0, 2 * np.pi)

rrt_star = RRTStar(start=start_point, goal=goal_point, search_area=search_area, step_size=0.1, path_step_size=0.01, max_iter=500, radius=5)
rrt_star.assign_robot(robot=robots[0])
path = rrt_star.plan()
#plot_rrt_star(rrt_star, path) Uncomment this to plot the path for q1 and q2. Need to close the plot window to continue
smooth_path = rrt_star.get_smoother_path()
### END PLANNING STUFF ###

time_2 = time.time()
print("Checking time: ", time_2 - time_1)

env.close()



env = UrdfEnv(
    dt=0.01,
    robots=robots,
    render=True,
)

world_plan.insert_into(env)
env.reconfigure_camera(5.0, 0.0, -89.99, (0, 0, 0))
ob = env.reset(pos=pos0, vel=vel0)

while True:
    for pose in smooth_path:
        robots[0].set_pose(pose=pose)
        time.sleep(0.01)
        



env.close()
