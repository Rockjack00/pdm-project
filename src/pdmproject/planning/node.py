import numpy as np

class Node:
    def __init__(self, q1, q2, q3, q4, q5, q6, q7):
        """Initialise node with joint values"""
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
        """Create node object from array

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
