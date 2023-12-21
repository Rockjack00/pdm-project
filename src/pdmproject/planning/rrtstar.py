import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class Node:
    def __init__(self, q1, q2, q3, q4, q5, q6, q7):
        """ Initialise node with joint values
        """
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7

        self.parent = None
        self.cost = 0.0
    
    @classmethod
    def from_array(cls, array):
        """ Create node object from array

        Args:
            array (Union[np.ndarray, List]): Pose for the node

        Raises:
            ValueError: Wrong amount of joint values given

        Returns:
            Node: Node for corresponding pose
        """
        # Ensure the array has the correct number of elements
        if len(array) != 7:
            raise ValueError("Array must contain 7 elements for a 7D point.")

        # Use array elements to initialize Node attributes
        return cls(*array)

    def get_7d_point(self):
        return np.array([self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7])


class RRTStar:
    def __init__(self, robot, start, goal, search_area, max_iter=1000, step_size=0.1, radius=1.0, sample_function= None):
        """ Initialise RRTStar planner class

        Args:
            robot (CollisionCheckRobot): Robot object for collision checking
            start (Node): Node for starting position
            goal (Node): Node for goal position
            search_area (): Bounds of search are for robot degreed of freedom
            max_iter (int, optional): Amount of itertaions in RRTStar planner. Defaults to 1000.
            step_size (float, optional): Step size for collision checking between nodes. Defaults to 0.1.
            radius (float, optional): Radius for connecting new node to existing node. Defaults to 1.0.

            #TODO
            sample_function (_type_, optional): Sampler for configurations from c-space. Defaults to None.
        """
        self.robot = robot
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.search_area = search_area #search_area shape is (min_q1, max_q1, min_q2, max_q2, ..., max_q7)
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
        self.node_list = [self.start]
        self.sample_function = sample_function or self._default_sample_function
        


    def _default_sample_function(self):
        """ Default sampling function that samples a random 7d node based on the bounds given by: search_area

        Returns:
            Node: randomly sampled node. Either new node or the goal node
        """

        if np.random.rand() < 0.9: # 1 - probability of sampling goal node
            
            return Node(np.random.uniform(self.search_area[0], self.search_area[1]), 
                        np.random.uniform(self.search_area[2], self.search_area[3]),
                        np.random.uniform(self.search_area[4], self.search_area[5]),
                        np.random.uniform(self.search_area[6], self.search_area[7]),
                        np.random.uniform(self.search_area[8], self.search_area[9]),
                        np.random.uniform(self.search_area[10], self.search_area[11]),
                        np.random.uniform(self.search_area[12], self.search_area[13]))

        else:
            return self.goal


    def collision_checker(self, pose):
        return self.robot.check_if_colliding(pose)


    @staticmethod
    def angle_difference_rad(from_angle, to_angle):
        """ Calculates shortest distance between two angles

        Args:
            from_angle (double): angle1 in rad
            to_angle (double): angle2 in rad

        Returns:
            double: angle difference in rad
        """

        return (to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi
    

    @staticmethod
    def calculate_distance(node1, node2):
        """ Calculate distance between two nodes

        Args:
            node1 (Node): point in C-space
            node2 (Node): poin in C-space

        Returns:
            double: distance
        """

        squared_distance = (node1.q1 - node2.q1) ** 2 + \
                            (node1.q2 - node2.q2) ** 2 + \
                            RRTStar.angle_difference_rad(from_angle=node1.q3, to_angle=node2.q3) ** 2 + \
                            (node1.q4 - node2.q4) ** 2 + \
                            (node1.q5 - node2.q5) ** 2 + \
                            (node1.q6 - node2.q6) ** 2 + \
                            RRTStar.angle_difference_rad(from_angle=node1.q7, to_angle=node2.q7) ** 2

        return np.sqrt(squared_distance)
    

    @staticmethod
    def unit_vector(from_node, to_node):
        """Calculates a unit vector that goes from one node to another

        Args:
            node1 (Node): point in C-space
            node2 (Node): poin in C-space

        Returns:
            np.array: unit vector
        """
        distance = RRTStar.calculate_distance(from_node, to_node)

        unit_vector = np.array([to_node.q1 - from_node.q1, 
                                to_node.q2 - from_node.q2, 
                                RRTStar.angle_difference_rad(from_angle=from_node.q3, to_angle=to_node.q3),
                                to_node.q4 - from_node.q4,
                                to_node.q5 - from_node.q5,
                                to_node.q6 - from_node.q6, 
                                RRTStar.angle_difference_rad(from_angle=from_node.q7, to_angle=to_node.q7)])
        
        unit_vector /= distance

        return unit_vector


    def get_nearest_node(self, new_node):
        """ Find the closest node to the new node

        Args:
            new_node (Node): newly sampled node

        Returns:
            Node: Node closest to the newly sampled node
        """
        distances = [RRTStar.calculate_distance(node, new_node) for node in self.node_list]
        min_index = np.argmin(distances)

        nearest_node = self.node_list[min_index]

        return nearest_node
    

    def check_collisions_between_nodes(self, from_node, to_node):
        """ Check for collision between two nodes using step size from RRTStar class instance

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
            node = Node.from_array(from_pose + unit_vector * (i+1) * self.step_size)
            if self.collision_checker(node.get_7d_point()):
                return True # Collision on path
            
        return False
    

    def rewire(self, new_node):
        """ Rewrite trees in the planner to minimize cost to newly added node

        Args:
            new_node (Node): newly added node
        """

        for node in self.node_list[1:]:
            if node != new_node.parent:

                cost = node.cost + RRTStar.calculate_distance(new_node, node)

                if cost < new_node.cost:
                    if self.check_collisions_between_nodes(from_node=node, to_node=new_node):
                        continue
                    new_node.parent = node
                    new_node.cost = cost


    def plan(self):
        """ Perform steps for RRTStar algorithm:
                Sample new node
                Collision checking
                Find near nodes in certain radius
                Assign parent node
                Rewire trees
        """

        for _ in tqdm(range(self.max_iter)):
                
            new_node = self.sample_function()

            if self.collision_checker(new_node.get_7d_point()):
                continue

            near_nodes = [node for node in self.node_list if RRTStar.calculate_distance(node, new_node) < self.radius]
            if not near_nodes:
                continue

            distances = [RRTStar.calculate_distance(node, new_node) for node in near_nodes]
            sorted_indices = np.argsort(distances)

            min_cost_node = near_nodes[sorted_indices[0]]

            if min_cost_node == self.goal:
                if len(near_nodes) < 1:
                    continue
                min_cost_node = near_nodes[sorted_indices[1]]

            if self.check_collisions_between_nodes(from_node=min_cost_node, to_node=new_node):
                continue

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + RRTStar.calculate_distance(new_node, min_cost_node)

            if new_node == self.goal:
                self.rewire(new_node)
                continue

            self.node_list.append(new_node)
            self.rewire(new_node)


    def _generate_path(self):
        """ Generate the path from start to goal node using nodes

        Returns:
            tuple(list, ...): List of node values for each DOF
        """
        path = []
        current_node = self.goal

        while current_node.parent is not None:
            path.append((current_node.q1, current_node.q2, current_node.q3, current_node.q4, current_node.q5, current_node.q6, current_node.q7))
            current_node = current_node.parent

        path.append((self.start.q1, self.start.q2, self.start.q3, self.start.q4, self.start.q5, self.start.q6, self.start.q7))
        path.reverse()

        # Unpack the path into separate lists for q1, q2, and joint angles
        path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7 = zip(*path)

        return path_q1, path_q2, path_q3, path_q4, path_q5, path_q6, path_q7
    
    
    def get_smoother_path(self, path_step_size=0.01):
        """ Generate smooth path from start to goal node using nodes and intermediate steps. Used for robot visualisation

        Args:
            path_step_size (float, optional): Intermediary step size. Defaults to 0.01.

        Returns:
            np.ndarray(): 7 by n_steps numpy array. Each row contains values for corresponding DoF
        """
        current_node = self.goal
        path = []

        while current_node.parent is not None:
            unit_vector = self.unit_vector(from_node=current_node, to_node=current_node.parent)
            distance = RRTStar.calculate_distance(current_node, current_node.parent)
            n_steps = int(distance // path_step_size)

            from_node_array = current_node.get_7d_point()

            for i in range(n_steps):
                path.append(from_node_array + unit_vector * (i) * path_step_size)

            current_node = current_node.parent

        path.reverse()

        return np.array(path)
    

    def plot_path(self):
        """ Plot the path from start to goal node 
        """

        path = self._generate_path()

        plt.figure(figsize=(10, 10))

        # Plot search area
        plt.plot([self.search_area[0], self.search_area[1], self.search_area[1], self.search_area[0], self.search_area[0]],
                [self.search_area[2], self.search_area[2], self.search_area[3], self.search_area[3], self.search_area[2]], 'k--')

        # Plot nodes and edges
        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.q1, node.parent.q1], [node.q2, node.parent.q2], 'bo-')

        #Plot path
        if path:
            path_q1, path_q2, _, _, _, _, _ = path
            plt.plot(path_q1, path_q2, 'r-', linewidth=2)

        plt.plot(self.start.q1, self.start.q2, 'go', markersize=10)
        plt.plot(self.goal.q1, self.goal.q2, 'ro', markersize=10)

        plt.title('RRT* Path Planning')
        plt.xlabel('q1-axis')
        plt.ylabel('q2-axis')
        plt.grid(True)
        plt.show()  