import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from tqdm import tqdm

from ..sampling import SamplerBase
from ..planning import Node
from . import metric

class RRTStar:
    def __init__(
        self,
        robot,
        start,
        goal,
        sampler: SamplerBase,
        max_iter=1000,
        step_size=0.1,
        radius=1.0,
    ):
        """Initialise RRTStar planner class

        Args:
            robot (CollisionCheckRobot): Robot object for collision checking
            start (Node): Node for starting position
            goal (Node): Node for goal position
            sampler (SamplerBase): The sampler for the states
            max_iter (int, optional): Amount of itertaions in RRTStar planner. Defaults to 1000.
            step_size (float, optional): Step size for collision checking between nodes. Defaults to 0.1.
            radius (float, optional): Radius for connecting new node to existing node. Defaults to 1.0.

            #TODO
            sample_function (_type_, optional): Sampler for configurations from c-space. Defaults to None.
        """
        self.robot = robot
        self.start = Node(start)
        self.goal = Node(goal)
        # self.search_area = search_area #search_area shape is (min_q1, max_q1, min_q2, max_q2, ..., max_q7)
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
        self.node_list = [self.start]
        self.sampler = sampler


    def collision_checker(self, pose):
        return self.robot.check_if_colliding(pose)

    @staticmethod
    def angle_difference_rad(from_angle, to_angle):
        """Calculates shortest distance between two angles

        Args:
            from_angle (double): angle1 in rad
            to_angle (double): angle2 in rad

        Returns:
            double: angle difference in rad
        """
        return metric.angle_metric(from_angle, to_angle)
        # return (to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi
        # Source book planning-algs page 205 equ 5.7
        # diff = np.abs(from_angle-to_angle)
        # return np.minimum(diff, 2*np.pi-diff)

    @staticmethod
    def calculate_distance(to_node: Node, from_node: Node):
        """Calculate distance between two nodes

        Args:
            to_node (Node): point in C-space
            from_node (Node): poin in C-space

        Returns:
            double: distance
        """
        # diff = to_node.get_7d_point() - from_node.get_7d_point()
        # diff[2::4] = (-diff[2::4]+np.pi)%(2*np.pi) - np.pi # This is wrong
        # squared_distance = diff**2
        # # abs_ang = np.absolute(diff[2::4])
        # # diff[2::4]= np.minimum(abs_ang, 2*np.pi - abs_ang)

        # return np.sqrt(np.sum(squared_distance))
        return metric.distance_metric(to_arr=to_node.get_7d_point(), from_arr=from_node.get_7d_point())

    @staticmethod
    def unit_vector(from_node: Node, to_node: Node):
        """Calculates a unit vector that goes from one node to another

        Args:
            from_node (Node): point in C-space
            to_node (Node): point in C-space

        Returns:
            np.array: unit vector
        """
        return metric.unit_vector(from_arr=from_node.get_7d_point(), to_arr=to_node.get_7d_point())

    def get_nearest_node(self, new_node):
        """Find the closest node to the new node

        Args:
            new_node (Node): newly sampled node

        Returns:
            Node: Node closest to the newly sampled node
        """
        distances = [
            RRTStar.calculate_distance(node, new_node) for node in self.node_list
        ]
        min_index = np.argmin(distances)

        nearest_node = self.node_list[min_index]

        return nearest_node

    def check_collisions_between_nodes(self, from_node, to_node):
        """Check for collision between two nodes using step size from RRTStar class instance

        Args:
            from_node (Node): starting node
            to_node (Node): end node

        Returns:
            Bool: True for collision, False otherwise
        """

        unit_vector = self.unit_vector(from_node=from_node, to_node=to_node)

        distance = RRTStar.calculate_distance(from_node, to_node)
        number_of_checks = int(distance // self.step_size)

        from_pose = from_node.get_7d_point()

        for i in range(number_of_checks):
            node = Node.from_array(from_pose + unit_vector * (i + 1) * self.step_size)
            if self.collision_checker(node.get_7d_point()):
                return True  # Collision on path

        return False

    def rewire(self, new_node):
        """Rewrite trees in the planner to minimize cost to newly added node

        Args:
            new_node (Node): newly added node
        """

        for node in self.node_list[1:]:
            if node != new_node.parent:
                cost = node.cost + RRTStar.calculate_distance(new_node, node)

                if cost < new_node.cost:
                    if self.check_collisions_between_nodes(
                        from_node=node, to_node=new_node
                    ):
                        continue
                    new_node.parent = node
                    new_node.cost = cost

    def plan(self):
        """Perform steps for RRTStar algorithm:
        Sample new node
        Collision checking
        Find near nodes in certain radius
        Assign parent node
        Rewire trees
        """

        # FIXME: TEMPORARY HACK FOR GOALPOINT
        self.sampler.register_goal_hack(self.goal, probability=0.05)

        for _ in tqdm(range(self.max_iter)):
            new_node = self.sampler.get_node_sample()

            if self.collision_checker(new_node.get_7d_point()):
                self.sampler.callback(new_node.get_7d_point(), self.robot)
                continue

            near_nodes = [
                node
                for node in self.node_list
                if RRTStar.calculate_distance(node, new_node) < self.radius
            ]
            if not near_nodes:
                continue

            distances = [
                RRTStar.calculate_distance(node, new_node) for node in near_nodes
            ]
            sorted_indices = np.argsort(distances)

            min_cost_node = near_nodes[sorted_indices[0]]

            if min_cost_node == self.goal:
                if len(near_nodes) < 1:
                    continue
                min_cost_node = near_nodes[sorted_indices[1]]

            # TODO: We could check multiple points if fail
            if self.check_collisions_between_nodes(
                from_node=min_cost_node, to_node=new_node
            ):
                # TODO: Add callback
                continue

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + RRTStar.calculate_distance(
                new_node, min_cost_node
            )

            if new_node == self.goal:
                self.rewire(new_node)
                continue

            self.node_list.append(new_node)
            self.rewire(new_node)

    def _generate_path(self):
        """Generate the path from start to goal node using nodes

        Returns:
            tuple(list, ...): List of node values for each DOF
        """
        path = []
        current_node = self.goal

        while current_node.parent is not None:
            path.append(
                (
                    *current_node.get_7d_point(),
                )
            )
            current_node = current_node.parent

        path.append(
            (
                *self.start.get_7d_point(),
            )
        )
        path.reverse()

        # Unpack the path into separate lists for q1, q2, and joint angles
        path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7 = zip(*path)

        return path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7

    def get_smoother_path(self, path_step_size=0.01):
        """Generate smooth path from start to goal node using nodes and intermediate steps. Used for robot visualisation

        Args:
            path_step_size (float, optional): Intermediary step size. Defaults to 0.01.

        Returns:
            np.ndarray(): 7 by n_steps numpy array. Each row contains values for corresponding DoF
        """
        current_node = self.goal
        path = []

        while current_node.parent is not None:
            unit_vector = self.unit_vector(
                from_node=current_node, to_node=current_node.parent
            )
            distance = RRTStar.calculate_distance(current_node, current_node.parent)
            n_steps = int(distance // path_step_size)

            from_node_array = current_node.get_7d_point()

            for i in range(n_steps):
                path.append(from_node_array + unit_vector * (i) * path_step_size)

            current_node = current_node.parent

        path.reverse()

        return np.array(path)

    def plot_path(self):
        """Plot the path from start to goal node"""

        path = self._generate_path()

        plt.figure(figsize=(10, 10))

        # Plot search area
        plt.plot(
            [
                self.sampler.lower_bound[0],
                self.sampler.upper_bound[0],
                self.sampler.upper_bound[0],
                self.sampler.lower_bound[0],
                self.sampler.lower_bound[0],
            ],
            [
                self.sampler.lower_bound[1],
                self.sampler.lower_bound[1],
                self.sampler.upper_bound[1],
                self.sampler.upper_bound[1],
                self.sampler.lower_bound[1],
            ],
            "k--",
        )

        # Plot nodes and edges
        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.q1, node.parent.q1], [node.q2, node.parent.q2], "bo-")

        # Plot path
        if path:
            path_q1, path_q2, _, _, _, _, _ = path
            plt.plot(path_q1, path_q2, "r-", linewidth=2)

        plt.plot(self.start.q1, self.start.q2, "go", markersize=10)
        plt.plot(self.goal.q1, self.goal.q2, "ro", markersize=10)

        plt.title("RRT* Path Planning")
        plt.xlabel("q1-axis")
        plt.ylabel("q2-axis")
        plt.grid(True)
        plt.show()
