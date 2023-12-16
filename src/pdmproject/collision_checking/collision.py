import pybullet as p
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher


class CollisionCheckRobot(GenericUrdfReacher):
    """ CollisionCheckRobot inherist from GenericUrdfReacher
        Adds collision checking functionality
    """

    def __init__(self, urdf):
        mode = "vel"
        super().__init__(urdf, mode)

    def check_if_colliding(self, pose, verbose=False)-> bool: 
        """ Check if given pose is colliding with the environment.

        Args:
            pose (Union[np.ndarray, List]): The pose to check for collisions.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        Returns:
            bool: True if collision is detected, False otherwise.
        """
            
        for i in range(self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pose[i],
                targetVelocity=0,
            )

        p.performCollisionDetection()
        contacts = p.getContactPoints(self._robot)

        if len(contacts) > 1:
            if verbose:
                print("COLLISION DETECTED")
            return True
        
        if verbose:
            print("NO COLLISION DETECTED")
        return False
        