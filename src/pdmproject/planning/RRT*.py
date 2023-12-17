import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, theta1, theta2, theta3, theta4, theta5):
        self.x = x
        self.y = y
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4
        self.theta5 = theta5
        self.parent = None
        self.cost = 0.0
    
    def from_array(cls, array):
        # Ensure the array has the correct number of elements
        if len(array) != 7:
            raise ValueError("Array must contain 7 elements for a 7D point.")

        # Use array elements to initialize Node attributes
        return cls(*array)

    def get_7d_point(self):
        return np.array([self.x, self.y, self.theta1, self.theta2, self.theta3, self.theta4, self.theta5])

class RRTStar:
    def __init__(self, start, goal, search_area, max_iter=1000, step_size=5.0, radius=100.0, obstacle_list= None, sample_function= None, collision_checker= None):
        self.start = Node(*start)
        self.goal = Node(*goal)


        self.obstacle_list = obstacle_list or []
        #search_area shape is (min_x, max_x, min_y, max_y, ....., max_theta5)
        self.search_area = search_area
        self.max_iter = max_iter
        self.step_size = step_size
        self.radius = radius
        self.node_list = [self.start]
        self.sample_function = sample_function or self._default_sample_function
        self.collision_checker = collision_checker or self._default_collision_checker

    def _default_sample_function(self):
        """Sampling function that samples a random 7d node baised on the min and max values given by the search_area

        Returns:
            Node: randomly sampled node
        """
        #set to how many times we want to sample goal node
        if np.random.rand() < 0.1:
            return Node(np.random.uniform(self.search_area[0], self.search_area[1]), np.random.uniform(self.search_area[2], self.search_area[3]),
                         np.random.uniform(self.search_area[4], self.search_area[5]),np.random.uniform(self.search_area[6], self.search_area[7]),
                         np.random.uniform(self.search_area[8], self.search_area[9]),np.random.uniform(self.search_area[10], self.search_area[11]),
                         np.random.uniform(self.search_area[12], self.search_area[13]))
        else:
            return Node(self.goal.x, self.goal.y,self.goal.theta1,self.goal.theta2,self.goal.theta3,self.goal.theta4,self.goal.theta5)
        
    def _default_collision_checker(self, node):
        """This function is a collision checker for very simple obstacles.

        Args:
            node (Node): 

        Returns:
            bool: collision
        """
        for obstacle in self.obstacle_list:
            distance = np.sqrt((node.x - obstacle[0]) ** 2 + (node.y - obstacle[1]) ** 2)
            if distance < self.radius:
                return False  # Collision
        return True

    def angle_difference_rad(self, angle1, angle2):
        """This function calculates the difference from two angles

        Args:
            angle1 (double): angle1 in rad
            angle2 (double): angle2 in rad

        Returns:
            double: angle difference in rad
        """
        return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi

    def calculate_distance(self, node1, node2):
        """This function calculates the distances between two nodes

        Args:
            node1 (Node): point in C-space
            node2 (Node): poin in C-space

        Returns:
            double: distance
        """
        distance = np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 )
        + self.angle_difference_rad(node1.theta1, node2.theta1) + self.angle_difference_rad(node1.theta2, node2.theta2)
        + self.angle_difference_rad(node1.theta3, node2.theta3) + self.angle_difference_rad(node1.theta4, node2.theta4)
        + self.angle_difference_rad(node1.theta5, node2.theta5)

        return distance
    
    def unit_vector(self, node1, node2):
        """Calculates a unit vector that goes from one node to another

        Args:
            node1 (Node): point in C-space
            node2 (Node): poin in C-space

        Returns:
            np.array: unit vector
        """
        distance = self.calculate_distance(node1,node2)
        unit_vector = np.array([node1.x - node2.x, node1.y - node2.y, self.angle_difference_rad(node1.theta1, node2.theta1), self.angle_difference_rad(node1.theta2, node2.theta2),
                                self.angle_difference_rad(node1.theta3, node2.theta3), self.angle_difference_rad(node1.theta4, node2.theta4), self.angle_difference_rad(node1.theta5, node2.theta5)])
        unit_vector /= distance
        return unit_vector

    def get_nearest_node(self, random_point):
        """This function finds the closest node to the random point node

        Args:
            random_point (_type_): _description_

        Returns:
            _type_: _description_
        """
        distances = [self.calculate_distance(n, random_point)  for n in self.node_list]
        min_index = np.argmin(distances)
        return self.node_list[min_index]
    
    def checking_between_nodes(self, from_node, to_node):
        unit_vector = self.unit_vector(to_node, from_node)
        distance = self.calculate_distance(to_node, from_node)
        number_of_checks = distance // self.step_size
        from_node_array = from_node.get_7d_point()
        for i in range(number_of_checks):
            node = Node.from_array(from_node_array + (i+1)* self.step_size)
            if self.collision_checker(node):
                return True       
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

        new_x = from_node.x + self.step_size * direction[0]
        new_y = from_node.y + self.step_size * direction[1]

        new_theta1 = from_node.theta1 + self.step_size * direction[2]
        new_theta2 = from_node.theta2 + self.step_size * direction[3]
        new_theta3 = from_node.theta3 + self.step_size * direction[4]
        new_theta4 = from_node.theta4 + self.step_size * direction[5]
        new_theta5 = from_node.theta5 + self.step_size * direction[6]

        new_node = Node(new_x, new_y, new_theta1, new_theta2, new_theta3, new_theta4, new_theta5)
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.step_size

        return new_node
    
    def rewire(self, new_node):
        """This is used to find the new shortest distances between nodes after a new node is added

        Args:
            new_node (_type_): _description_
        """
        for node in self.node_list:
            if node != new_node.parent and self.collision_checker(node):
                cost = node.cost + self.calculate_distance(new_node, node)
                if cost < new_node.cost:
                    new_node.parent = node
                    new_node.cost = cost

    def plan(self):
        for _ in range(self.max_iter):
            new_node = self.sample_function()
            if not self.collision_checker(new_node):
                continue

            nearest_node = self.get_nearest_node(new_node)

            near_nodes = [node for node in self.node_list if self.calculate_distance(node,new_node) < self.radius]
            if not near_nodes:
                continue

            min_cost_node = min(near_nodes, key=lambda node: node.cost + self.calculate_distance(node,new_node))

            self.node_list.append(new_node)
            self.rewire(new_node)

        path = self.generate_path()
        return path
    
    def generate_path(self):
        path = [(self.goal.x, self.goal.y, self.goal.theta1, self.goal.theta2, self.goal.theta3, self.goal.theta4, self.goal.theta5)]
        current_node = self.node_list[-1]

        while current_node.parent is not None:
            path.append((current_node.x, current_node.y, current_node.theta1, current_node.theta2, current_node.theta3, current_node.theta4, current_node.theta5))
            current_node = current_node.parent

        path.append((self.start.x, self.start.y, self.start.theta1, self.start.theta2, self.start.theta3, self.start.theta4, self.start.theta5))
        path.reverse()

        # Unpack the path into separate lists for x, y, and joint angles
        path_x, path_y, path_theta1, path_theta2, path_theta3, path_theta4, path_theta5 = zip(*path)

        return path_x, path_y, path_theta1, path_theta2, path_theta3, path_theta4, path_theta5
    

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
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'bo-')

    # Plot path
    if path:
        path_x, path_y, _, _, _, _, _ = path
        plt.plot(path_x, path_y, 'r-', linewidth=2)

    plt.plot(rrt_star.start.x, rrt_star.start.y, 'go', markersize=10)
    plt.plot(rrt_star.goal.x, rrt_star.goal.y, 'ro', markersize=10)

    plt.title('RRT* Path Planning')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

start_point = (1, 1,0, np.pi/2, 0, 0, 0)
goal_point = (50, 50, 0, np.pi/2, 0, 0 ,0)
obstacles = []  # Coordinates of obstacles
search_area = (0, 100, 0, 100,0,2* np.pi, 0,2* np.pi,0,2* np.pi,0,2* np.pi,0,2* np.pi)  # Define the search area
rrt_star = RRTStar(start_point, goal_point, search_area)
path = rrt_star.plan()
plot_rrt_star(rrt_star, path)


