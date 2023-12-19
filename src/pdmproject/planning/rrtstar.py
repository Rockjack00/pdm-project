import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pdmproject.collision_checking import CollisionCheckRobot


class Node:
    def __init__(self, q1, q2, q3, q4, q5, q6, q7):
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
        # Ensure the array has the correct number of elements
        if len(array) != 7:
            raise ValueError("Array must contain 7 elements for a 7D point.")

        # Use array elements to initialize Node attributes
        return cls(*array)

    def get_7d_point(self):
        return np.array([self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7])


class RRTStar:
    def __init__(self, start, goal, search_area, max_iter=1000, step_size=5.0, path_step_size=0.1, radius=100.0, obstacle_list= None, sample_function= None, collision_checker= None):
        self.start = Node(*start)
        self.goal = Node(*goal)

        self.obstacle_list = obstacle_list or []
        #search_area shape is (min_x, max_x, min_y, max_y, ....., max_theta5)
        self.search_area = search_area
        self.max_iter = max_iter
        self.step_size = step_size
        self.path_step_size = path_step_size
        self.radius = radius
        self.node_list = [self.start]
        self.sample_function = sample_function or self._default_sample_function
        self.collision_checker = collision_checker or self._new_collision_checker


    def _default_sample_function(self):
        """Sampling function that samples a random 7d node based on the min and max values given by the search_area

        Returns:
            Node: randomly sampled node
        """
        #set to how many times we want to sample goal node
        if np.random.rand() < 0.85:
            
            return Node(np.random.uniform(self.search_area[0], self.search_area[1]), 
                        np.random.uniform(self.search_area[2], self.search_area[3]),
                        np.random.uniform(self.search_area[4], self.search_area[5]),
                        np.random.uniform(self.search_area[6], self.search_area[7]),
                        np.random.uniform(self.search_area[8], self.search_area[9]),
                        np.random.uniform(self.search_area[10], self.search_area[11]),
                        np.random.uniform(self.search_area[12], self.search_area[13]))

        else:
            return self.goal
        

    def _default_collision_checker(self, node):
        """This function is a collision checker for very simple obstacles.

        Args:
            node (Node): 

        Returns:
            bool: collision
        """
        for obstacle in self.obstacle_list:
            distance = np.sqrt((node.q1 - obstacle[0]) ** 2 + (node.q2 - obstacle[1]) ** 2)
            if distance < self.radius:
                return True  # Collision
        return False
     
    
    def assign_robot(self, robot):
        self.robot = robot


    def _new_collision_checker(self, node):
        return self.robot.check_if_colliding(node.get_7d_point(), verbose=False)



    @staticmethod
    def angle_difference_rad(angle1, angle2):
        """This function calculates the difference from two angles

        Args:
            angle1 (double): angle1 in rad
            angle2 (double): angle2 in rad

        Returns:
            double: angle difference in rad
        """
        return (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
    

    @staticmethod
    def calculate_distance(node1, node2):
        """This function calculates the distances between two nodes

        Args:
            node1 (Node): point in C-space
            node2 (Node): poin in C-space

        Returns:
            double: distance
        """
        squared_distance = (node1.q1 - node2.q1) ** 2 + \
                            (node1.q2 - node2.q2) ** 2 + \
                            RRTStar.angle_difference_rad(node1.q3, node2.q3) ** 2 + \
                            (node1.q4 - node2.q4) ** 2 + \
                            (node1.q5 - node2.q5) ** 2 + \
                            (node1.q6 - node2.q6) ** 2 + \
                            RRTStar.angle_difference_rad(node1.q7, node2.q7) ** 2

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
                                RRTStar.angle_difference_rad(to_node.q3, from_node.q3),
                                to_node.q4 - from_node.q4,
                                to_node.q5 - from_node.q5,
                                to_node.q6 - from_node.q6, 
                                RRTStar.angle_difference_rad(to_node.q7, from_node.q7)])
        unit_vector /= distance

        return unit_vector


    def get_nearest_node(self, new_node):
        """This function finds the closest node to the random point node

        Args:
            random_point (_type_): _description_

        Returns:
            _type_: _description_
        """
        distances = [RRTStar.calculate_distance(node, new_node) for node in self.node_list]
        min_index = np.argmin(distances)

        nearest_node = self.node_list[min_index]

        return nearest_node
    

    def check_to_node(self, from_node, to_node):
        unit_vector = self.unit_vector(from_node=from_node, to_node=to_node)
        distance = RRTStar.calculate_distance(from_node, to_node)
        number_of_checks = int(distance // self.step_size)
        from_node_array = from_node.get_7d_point()

        for i in range(number_of_checks):
            node = Node.from_array(from_node_array + unit_vector * (i+1) * self.step_size)
            if self.collision_checker(node):
                return True # Collision on path
        return False
    

    def step(self, from_node, to_node):
        """This function is often used when you want to take a step in the direction of the new node instead of using the new node and checking if this node fits in the tree.
           is not used right now but might be used in the future

        Args:
            from_node (Node): point in C-space
            to_node (Node): poin in C-space

        Returns:
            Node: the new node
        """
        direction = self.unit_vector(to_node, from_node)

        new_q1 = from_node.q1 + self.step_size * direction[0]
        new_q2 = from_node.q2 + self.step_size * direction[1]

        new_q3 = from_node.q3 + self.step_size * direction[2]
        new_q4 = from_node.q4 + self.step_size * direction[3]
        new_q5 = from_node.q5 + self.step_size * direction[4]
        new_q6 = from_node.q6 + self.step_size * direction[5]
        new_q7 = from_node.q7 + self.step_size * direction[6]

        new_node = Node(new_q1, new_q2, new_q3, new_q4, new_q5, new_q6, new_q7)

        new_node.parent = from_node
        new_node.cost = from_node.cost + self.step_size

        return new_node
    

    def rewire(self, new_node):
        """This is used to find the new shortest distances between nodes after a new node is added

        Args:
            new_node (_type_): _description_
        """
        for node in self.node_list[1:]:
            if node != new_node.parent:
                if self.check_to_node(from_node=node, to_node=new_node):
                    continue

                cost = node.cost + RRTStar.calculate_distance(new_node, node)

                if cost < new_node.cost:
                    new_node.parent = node
                    new_node.cost = cost

    def plan(self):
        for i in range(self.max_iter):
            print(i)
            new_node = self.sample_function()
            
            if self.collision_checker(new_node):
                continue

            # nearest_node = self.get_nearest_node(new_node)

            # if RRTStar.calculate_distance(nearest_node, new_node) > self.radius:
            #     continue

            # if self.check_to_node(from_node=nearest_node, to_node=new_node):
            #     continue  

            # new_node.parent = nearest_node
            # new_node.cost = new_node.parent.cost + RRTStar.calculate_distance(new_node, new_node.parent)

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

            if self.check_to_node(from_node=min_cost_node, to_node=new_node):
                continue

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + RRTStar.calculate_distance(new_node, min_cost_node)

            if new_node == self.goal:
                continue

            self.node_list.append(new_node)
            self.rewire(new_node)


            ### Uncomment to stop when a path is found (Not optimal path but cuts down on time)
            #if self.goal.parent is not None:
            #    break

        path = self.generate_path()
        self.path = path
        return path
    

    def generate_path(self):
        #path = [(self.goal.q1, self.goal.q2, self.goal.q3, self.goal.q4, self.goal.q5, self.goal.q6, self.goal.q7)]
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
    
    
    def get_smoother_path(self):
        current_node = self.goal
        path = []

        while current_node.parent is not None:
            unit_vector = self.unit_vector(from_node=current_node, to_node=current_node.parent)
            distance = RRTStar.calculate_distance(current_node, current_node.parent)
            n_steps = int(distance // self.path_step_size)

            from_node_array = current_node.get_7d_point()

            for i in range(n_steps):
                path.append(from_node_array + unit_vector * (i) * self.path_step_size)

            current_node = current_node.parent

        path.reverse()

        return np.array(path)
    

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



if __name__ == "__main__":
    start_point = (-4, 0, 0, 0, 0, 0, 0)
    goal_point = (4, 0, 0, 0, 0, 0 ,0)

    obstacles = []  # Coordinates of obstacles

    search_area = (-5, 5, 
                -3, 3, 
                0, 2 * np.pi, 
                -1/2 * np.pi, 1/2 * np.pi, 
                -2/3 * np.pi, 2/3 * np.pi, 
                -2/3 * np.pi, 2/3 * np.pi, 
                0, 2 * np.pi)

    rrt_star = RRTStar(start=start_point, goal=goal_point, search_area=search_area, step_size=0.1, max_iter=500, radius=1)
    path = rrt_star.plan()
    plot_rrt_star(rrt_star, path)