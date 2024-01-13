from typing import Collection
import time

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

        self.pose = np.zeros(7)
        self.limits = np.array([[-np.pi, np.pi],[-np.pi / 2, np.pi / 2], [-np.pi, np.pi], [-np.pi, np.pi]])

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

        self.pose = np.array(pose)

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
    
    def collision_finder(self):
        """ Generate collision points based on a given collision configuration. 
        Run this method if method check_if_colliding returns true

        Returns:
            tuple[list, list]:  First array contais link ids with values 0, 1, 2 or 3
                                Second array contains collision points pairs. If link 0: collision_pose, zeros(3)
                                If lists are empty -> Self Collision
        """
        contact_links, contact_links_poses = self.get_links_data()
        pose = np.copy(self.pose)

        links = []
        all_colls = []

        for contact_link in set(contact_links):
            
            if contact_link == 2:
                links.append(0)
                current_contact_links, current_contact_links_poses = self.get_links_data()
                current_contact_links_poses = current_contact_links_poses[current_contact_links == 2]
                current_contact_links = current_contact_links[current_contact_links == 2]
                
                colls_l2_q12 = np.array([current_contact_links_poses[0], np.zeros(3)])
                all_colls.append(colls_l2_q12)
                continue

            if contact_link == 4:
                links.append(1)

                check_pose = np.copy(pose)
                q3s = np.linspace(self.limits[0][0], self.limits[0][1], 10)
                colls_l4_q3 = []
                for q3 in q3s:
                    check_pose[2] = q3
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[current_contact_links == 4]
                        current_contact_links = current_contact_links[current_contact_links == 4]

                        if len(current_contact_links) > 0:
                            colls_l4_q3.append([*current_contact_links_poses[0]])

                if colls_l4_q3:
                    all_colls.append([colls_l4_q3[0], colls_l4_q3[-1]])
                    continue


                check_pose = np.copy(pose)
                q4s = np.linspace(self.limits[1][0], self.limits[1][1], 10)
                colls_l4_q4 = []
                for q4 in q4s:
                    check_pose[3] = q4
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[current_contact_links == 4]
                        current_contact_links = current_contact_links[current_contact_links == 4]

                        if len(current_contact_links) > 0:
                            colls_l4_q4.append([*current_contact_links_poses[0]])
                
                if colls_l4_q4:
                    all_colls.append([colls_l4_q4[0], colls_l4_q4[-1]])
                    continue


            if contact_link == 6:
                links.append(2)

                check_pose = np.copy(pose)
                q3s = np.linspace(self.limits[0][0], self.limits[0][1], 10)
                colls_l6_q3 = []
                for q3 in q3s:
                    check_pose[2] = q3
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[current_contact_links == 6]
                        current_contact_links = current_contact_links[current_contact_links == 6]

                        if len(current_contact_links) > 0:
                            colls_l6_q3.append([*current_contact_links_poses[0]])

                if colls_l6_q3:
                    all_colls.append([colls_l6_q3[0], colls_l6_q3[-1]])
                    continue


                check_pose = np.copy(pose)
                q4s = np.linspace(self.limits[1][0], self.limits[1][1], 10)
                colls_l6_q4 = []
                for q4 in q4s:
                    check_pose[3] = q4
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[current_contact_links == 6]
                        current_contact_links = current_contact_links[current_contact_links == 6]

                        if len(current_contact_links) > 0:
                            colls_l6_q4.append([*current_contact_links_poses[0]])

                if colls_l6_q4:
                    all_colls.append([colls_l6_q4[0], colls_l6_q4[-1]])
                    continue


                check_pose = np.copy(pose)
                q5s = np.linspace(self.limits[2][0], self.limits[2][1], 10)
                colls_l6_q5 = []
                for q5 in q5s:
                    check_pose[4] = q5

                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[current_contact_links == 6]
                        current_contact_links = current_contact_links[current_contact_links == 6]

                        if len(current_contact_links) > 0:
                            colls_l6_q5.append([*current_contact_links_poses[0]])

                if colls_l6_q5:
                    all_colls.append([colls_l6_q5[0], colls_l6_q5[-1]])
                    continue



            if contact_link == 8 or contact_link == 9:
                links.append(3)

                check_pose = np.copy(pose)
                q3s = np.linspace(self.limits[0][0], self.limits[0][1], 10)
                colls_l89_q3 = []
                for q3 in q3s:
                    check_pose[2] = q3
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[(current_contact_links == 8) | (current_contact_links == 9)]
                        current_contact_links = current_contact_links[(current_contact_links == 8) | (current_contact_links == 9)]

                        if len(current_contact_links) > 0:
                            colls_l89_q3.append([*current_contact_links_poses[0]])

                if colls_l89_q3:
                    all_colls.append([colls_l89_q3[0], colls_l89_q3[-1]])
                    continue
                

                check_pose = np.copy(pose)
                q4s = np.linspace(self.limits[1][0], self.limits[1][1], 10)
                colls_l89_q4 = []
                for q4 in q4s:
                    check_pose[3] = q4
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[(current_contact_links == 8) | (current_contact_links == 9)]
                        current_contact_links = current_contact_links[(current_contact_links == 8) | (current_contact_links == 9)]

                        if len(current_contact_links) > 0:
                            colls_l89_q4.append([*current_contact_links_poses[0]])

                if colls_l89_q4:
                    all_colls.append([colls_l89_q4[0], colls_l89_q4[-1]])
                    continue


                check_pose = np.copy(pose)
                q5s = np.linspace(self.limits[2][0], self.limits[2][1], 10)
                colls_l89_q5 = []
                for q5 in q5s:
                    check_pose[4] = q5
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[(current_contact_links == 8) | (current_contact_links == 9)]
                        current_contact_links = current_contact_links[(current_contact_links == 8) | (current_contact_links == 9)]

                        if len(current_contact_links) > 0:
                            colls_l89_q5.append([*current_contact_links_poses[0]])

                if colls_l89_q5:
                    all_colls.append([colls_l89_q5[0], colls_l89_q5[-1]])
                    continue

                
                check_pose = np.copy(pose)
                q6s = np.linspace(self.limits[3][0], self.limits[3][1], 10)
                colls_l89_q6 = []
                for q6 in q6s:
                    check_pose[5] = q6
                    if self.check_if_colliding(check_pose):
                        current_contact_links, current_contact_links_poses = self.get_links_data()
                        current_contact_links_poses = current_contact_links_poses[(current_contact_links == 8) | (current_contact_links == 9)]
                        current_contact_links = current_contact_links[(current_contact_links == 8) | (current_contact_links == 9)]

                        if len(current_contact_links) > 0:
                            colls_l89_q6.append([*current_contact_links_poses[0]])

                if colls_l89_q6:
                    all_colls.append([colls_l89_q6[0], colls_l89_q6[-1]])
                    continue

        return links, all_colls
