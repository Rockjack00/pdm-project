"""The submodule which contains the RRTStar class which implements a version of the RRT* algorithm."""
import functools
from itertools import pairwise
from operator import itemgetter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..planning import Node
from ..sampling import SamplerBase
from . import metric

from tqdm import tqdm


class RRTStar:
    """An implmentation of the RRT* algrithm in Configuration space (C-space).

    This implementation is generalized to allow for nullspace objects in C-space, which get discovered when collision checking.
    """

    def __init__(
        self,
        robot,
        start,
        goal,
        sampler: SamplerBase,
        max_iter=1000,
        step_size=0.1,
        radius=1.0,
        shrinking_radius: bool = False,
    ):
        """Initialise RRTStar planner class.

        Args:
            robot (CollisionCheckRobot): Robot object for collision checking
            start (Node): Node for starting position
            goal (Node): Node for goal position
            sampler (SamplerBase): The sampler for the states
            max_iter (int, optional): Amount of itertaions in RRTStar planner. Defaults to 1000.
            step_size (float, optional): Step size for collision checking between nodes. Defaults to 0.1.
            radius (float, optional): Radius for connecting new node to existing node. Used as the initial radius at n=2 if shrinking_radius is True. Defaults to 1.0.
            shrinking_radius (bool, optional): Shrink the radius based on the number of samples taken. According to the following equation: r = r_init * (log(n)/n)**(1/7)$. Defaults to False.
        """
        self.robot = robot
        self.start = Node(start)
        self.goal = Node(goal)
        # self.search_area = search_area #search_area shape is (min_q1, max_q1, min_q2, max_q2, ..., max_q7)
        self.max_iter = max_iter
        self.step_size = step_size
        self._radius = radius
        self.shrink_radius = shrinking_radius
        self.node_list = [self.start]
        self.sampler = sampler

        if self.shrink_radius:
            self._radius /= (np.log(2) / 2) ** (1 / 7)

        # Variables for keeping track of the metrics
        self._num_iter_till_first_path: Optional[int] = None
        self._explored_nodes_till_first_path: Optional[int] = None
        self._collision_count_till_first_path: Optional[int] = None
        self._collision_count: int = 0
        self._rejected_nodes: int = 0
        self._planned: int = (
            0  # The amount of time plan has been used on this RRTStar instance.
        )

    def collision_checker(
        self,
        pose: npt.NDArray[np.float64],
        perform_callback: bool = True,
        count_reject: bool = False,
    ) -> bool:
        """Check if the specified pose is colliding.

        Args:
            pose (np.ndarray[np.float64]): The pose of the robot in C-space, which will be checked.
            perform_callback (bool, optional): If the sampler callback should be called. Defaults to True.
            count_reject (bool, optional): If the amount of rejected nodes should be added up. Defaults to False.

        Returns:
            bool: True for collision, False otherwise
        """
        is_colliding = self.robot.check_if_colliding(pose)

        if perform_callback and is_colliding:
            self._collision_count += 1
            self.sampler.callback(pose, self.robot)

        if count_reject and is_colliding:
            self._rejected_nodes += 1

        return is_colliding

    @staticmethod
    def angle_difference_rad(from_angle: float, to_angle: float) -> float:
        """Calculates shortest distance between two angles.

        Args:
            from_angle (float): angle1 in rad
            to_angle (float): angle2 in rad

        Returns:
            float: angle difference in rad
        """
        return metric.angle_metric(from_angle, to_angle)

    @staticmethod
    def calculate_distance(to_node: Node, from_node: Node) -> float:
        """Calculate distance between two nodes.

        Args:
            to_node (Node): point in C-space
            from_node (Node): poin in C-space

        Returns:
            double: distance
        """
        return metric.distance_metric(
            to_arr=to_node.get_7d_point(), from_arr=from_node.get_7d_point()
        )  # type: ignore

    @staticmethod
    def unit_vector(from_node: Node, to_node: Node) -> npt.NDArray[np.float64]:
        """Calculates a unit vector that goes from one node to another.

        Args:
            from_node (Node): point in C-space
            to_node (Node): point in C-space

        Returns:
            np.array: unit vector
        """
        return metric.unit_vector(
            from_arr=from_node.get_7d_point(), to_arr=to_node.get_7d_point()
        )

    def get_nearest_node(self, new_node):
        """Find the closest node to the new node.

        Args:
            new_node (Node): newly sampled node

        Returns:
            Node: Node closest to the newly sampled node
        """
        distances = [
            RRTStar.calculate_distance(node, new_node) for node in self.node_list
        ]
        min_index = np.argmin(distances)  # type: ignore

        nearest_node = self.node_list[min_index]

        return nearest_node

    def check_collisions_between_nodes(
        self,
        from_node: Node,
        to_node: Node,
        perform_callback: bool = True,
        count_reject: bool = False,
    ) -> bool:
        """Check for collision between two nodes using step size from RRTStar class instance.

        Args:
            from_node (Node): starting node
            to_node (Node): end node
            perform_callback (bool, optional): If the sampler callback should be called. Defaults to None
            count_reject (bool, optional): If the amount of rejected nodes should be added up. Defaults to False.

        Returns:
            bool: True for collision, False otherwise
        """
        unit_vector = self.unit_vector(from_node=from_node, to_node=to_node)

        distance = RRTStar.calculate_distance(from_node, to_node)
        number_of_checks = int(distance // self.step_size)

        from_pose = from_node.get_7d_point()

        vector = distance * unit_vector

        for i in range(number_of_checks):
            # Check the path using the van der Corput sequence
            node = Node(from_pose + vector * van_der_corput(i + 1))
            if self.collision_checker(
                node.get_7d_point(),
                perform_callback=perform_callback,
                count_reject=count_reject,
            ):
                return True  # Collision on path

        return False

    def rewire(self, new_node):
        """Rewrite trees in the planner to minimize cost to newly added node.

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
        """Perform steps for RRT* algorithm.

        The following steps are performed according to the RRT* algorithm:
         - Sample new node
         - Collision checking
         - Find near nodes in certain radius
         - Assign parent node
         - Rewire trees
        """
        self._planned += 1

        # FIXME: TEMPORARY HACK FOR GOALPOINT
        self.sampler.register_goal_hack(self.goal, probability=0.05)

        for i in tqdm(range(self.max_iter)):
            new_node = self.sampler.get_node_sample()

            if self.collision_checker(
                new_node.get_7d_point(), perform_callback=True, count_reject=True
            ):
                # The callback gets called by the collision checking function.
                continue

            near_nodes: list[tuple[float, Node]] = sorted(
                (
                    (d, node)
                    for node in self.node_list
                    if (d := RRTStar.calculate_distance(node, new_node)) < self.radius
                ),
                key=itemgetter(0),
            )
            if not near_nodes:
                continue

            min_cost_dist, min_cost_node = near_nodes[0]

            if min_cost_node == self.goal:
                if len(near_nodes) < 1:
                    continue
                min_cost_dist, min_cost_node = near_nodes[1]

            # TODO: We could check multiple points if fail
            if self.check_collisions_between_nodes(
                from_node=min_cost_node,
                to_node=new_node,
                perform_callback=True,
                count_reject=True,
            ):
                # The callback gets called by the collision checking function.
                continue

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + min_cost_dist

            if new_node == self.goal:
                if self._num_iter_till_first_path is None:
                    self._num_iter_till_first_path = (
                        i + 1 + self.max_iter * (self._planned - 1)
                    )
                    self._explored_nodes_till_first_path = len(self.node_list)
                    self._collision_count_till_first_path = self._collision_count
                self.rewire(new_node)
                continue

            self.node_list.append(new_node)
            self.rewire(new_node)

    def _generate_path(self):
        """Generate the path from start to goal node using nodes.

        Returns:
            tuple[list, ...]: List of node values for each DOF
        """
        path = []
        current_node = self.goal

        while current_node.parent is not None:
            path.append((*current_node.get_7d_point(),))
            current_node = current_node.parent

        path.append((*self.start.get_7d_point(),))
        path.reverse()

        # Unpack the path into separate lists for q1, q2, and joint angles
        path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7 = zip(*path)

        return path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7

    def get_smoother_path(self, path_step_size=0.01):
        """Generate smooth path from start to goal node using nodes and intermediate steps. Used for robot visualisation.

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

    def plot_path(self, block: bool = True):
        """Plot the path from start to goal node."""
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
                plt.plot([node.q1, node.parent.q1], [node.q2, node.parent.q2], "bo-")  # type: ignore

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
        plt.show(block=block)

    def _node_path(self) -> Optional[list[Node]]:
        """Get the path interms of Nodes if any.

        Returns:
            Optional[list[Node]]: Get a path of nodes from start till end, if a path has been found. Otherwise None is returned.
        """
        if self.has_found_path():
            path = []
            current_node = self.goal

            while current_node.parent is not None:
                path.append(current_node)
                current_node = current_node.parent

            path.append(self.start)
            path.reverse()
            return path

    @property
    def num_iter_till_first_path(self) -> Optional[int]:
        """The number of planning iterations at the time first valid path was found.

        Returns:
            Optional[int]: If a path has been found, the number of iterations at the time first valid path was found. Otherwise, return None.
        """
        return self._num_iter_till_first_path

    @property
    def num_iter(self) -> int:
        """The total amount of planning iterations done until now.

        Returns:
            int: The amount of planning iterations.
        """
        return self.max_iter * self._planned

    @property
    def explored_nodes_till_first_path(self) -> Optional[int]:
        """The number of explored Nodes at the time first valid path was found.

        Returns:
            Optional[int]: If a path has been found, the number of explored nodes at the time first valid path was found. Otherwise, return None.
        """
        return self._explored_nodes_till_first_path

    @property
    def explored_nodes(self) -> int:
        """The number of explored Nodes until now.

        Returns:
            int: The number of explored nodes.
        """
        return len(self.node_list)

    def has_found_path(self) -> bool:
        """Check if the RRT* has found a valid path yet.

        Returns:
            bool: True if a path has been found. False otherwise.
        """
        return self.goal.parent is not None

    @property
    def rejected_nodes(self) -> int:
        """The amount rejected nodes found during the adding of new Nodes.

        Returns:
            int: The rejected Node count.
        """
        return self._rejected_nodes

    @property
    def collision_count(self) -> int:
        """The amount of collision found during planning.

        Returns:
            int: The collision count.
        """
        return self._collision_count

    @property
    def collision_count_till_first_path(self) -> Optional[int]:
        """The amount of collision found during planning at the time first valid path was found.

        Returns:
            Optional[int]: The collision count at the time first valid path was found, otherwise None.
        """
        return self._collision_count_till_first_path

    @property
    def path_length(self) -> Optional[float]:
        """Get the length of the current found path in Configuration-space.

        Returns:
            Optional[float]: The cost of the found path provided that a path has been found.
        """
        if self.has_found_path():
            return sum(
                RRTStar.calculate_distance(from_node=from_node, to_node=to_node)
                for from_node, to_node in pairwise(self._node_path())  # type: ignore
            )

    @property
    def radius(self) -> float:
        """Get the current connecting radius.

        If shrink_radius was set to true at initialization, than the radius will shrink with the number of explored nodes.
        Otherwise it stays constant.

        Returns:
            float: The connecting radius
        """
        if self.shrink_radius:
            n = self.explored_nodes
            return self._radius * (np.log(n) / n) ** (1 / 7) if n > 1 else self._radius
        else:
            return self._radius


@functools.cache
def van_der_corput(n: int, base=2) -> float:
    """Calculate the nth element in the van der Corput sequence.

    The implmentation is based on the Wikipedia description and [this code example](https://rosettacode.org/wiki/Van_der_Corput_sequence#Python).

    Args:
        n (int): The nth element of the sequence.
        base (int, optional): The base in which to calculate the sequence. Defaults to 2.

    Returns:
        float: The nth element of the van der Corput sequence in the given base.
    """
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc
