from typing import Collection

import numpy as np

import pybullet as p
from urdfenvs.robots.generic_urdf import GenericUrdfReacher


class CollisionCheckRobot(GenericUrdfReacher):
    """CollisionCheckRobot inherits from GenericUrdfReacher.

    Adds collision checking functionality.
    """

    def __init__(self, urdf) -> None:
        """Initialise robot class.

        Args:
            urdf (string): Path to robot urdf file
        """
        mode = "vel"
        super().__init__(urdf, mode)

    def reset(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mount_position: np.ndarray,
        mount_orientation: np.ndarray,
    ) -> None:
        if hasattr(self, "_robot"):
            p.removeBody(self._robot)
        self._robot = p.loadURDF(
            fileName=self._urdf_file,
            basePosition=mount_position.tolist(),
            baseOrientation=mount_orientation.tolist(),
            flags=p.URDF_USE_SELF_COLLISION,
            useFixedBase=True,
        )
        self.set_joint_names()
        self.extract_joint_ids()
        for i in range(self._n):
            p.resetJointState(
                self._robot,
                self._robot_joints[i],
                pos[i],
                targetVelocity=vel[i],
            )
        self.update_state()
        self._integrated_velocities = vel

    def set_pose(self, pose: Collection) -> None:
        """Set robot pose in simulation.

        Args:
            pose (Collection): robot pose to set. There should be self._n items of type float.
        """
        assert len(pose) == self._n

        for robot_joint, joint_pose in zip(self._robot_joints, pose):
            p.resetJointState(
                self._robot,
                robot_joint,
                joint_pose,
                targetVelocity=0,
            )

        p.performCollisionDetection()

    def check_if_colliding(self, pose) -> bool:
        """Check if given pose is colliding with the environment.

        Args:
            pose (Union[np.ndarray, List]): The pose to check for collisions.

        Returns:
            bool: True if collision is detected, False otherwise.
        """
        self.set_pose(pose)

        contacts = p.getContactPoints(self._robot)
        self_contacts = p.getContactPoints(self._robot, self._robot)

        if len(contacts) - len(self_contacts) > 1:
            return True

        valid_self_contacts = [
            tpl
            for tpl in self_contacts
            if tpl[3] - tpl[4] != 2 and tpl[3] - tpl[4] != -2
        ]

        if len(valid_self_contacts) > 0:
            return True

        return False

    def get_links_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get point data for collisions.

        Returns:
            tuple[np.ndarray, np.ndarray]: first np.array returns link indices of links which are colliding
            second np.ndarray returns x y z position of collision points on the obstacle in world coordinates
        """
        contacts = p.getContactPoints(self._robot)

        obstacle_contacts = [
            tpl for tpl in contacts if tpl[1] != tpl[2] and tpl[2] != 0
        ]

        n_collisions = len(obstacle_contacts)
        contact_links = np.zeros(n_collisions, dtype=np.int8)
        contact_links_poses = np.zeros((n_collisions, 3))

        for i, contact in enumerate(obstacle_contacts):
            contact_links[i] = contact[3]
            contact_links_poses[i, 0:] = contact[6]

        return contact_links, contact_links_poses
