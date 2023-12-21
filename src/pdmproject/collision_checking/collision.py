import pybullet as p
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher


class CollisionCheckRobot(GenericUrdfReacher):
    """ CollisionCheckRobot inherits from GenericUrdfReacher
        Adds collision checking functionality
    """

    def __init__(self, urdf) -> None:
        """ Initialise robot class

        Args:
            urdf (string): Path to robot urdf file
        """
        mode = "vel"
        super().__init__(urdf, mode)


    def set_pose(self, pose) -> None:
        """ Set robot pose in simulation

        Args:
            pose (Union[np.ndarray, List]): robot pose to set
        """
        for i in range(self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pose[i],
                targetVelocity=0,
            )

        p.performCollisionDetection()


    def check_if_colliding(self, pose)-> bool: 
        """ Check if given pose is colliding with the environment.

        Args:
            pose (Union[np.ndarray, List]): The pose to check for collisions.

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
            return True
        
        return False
        