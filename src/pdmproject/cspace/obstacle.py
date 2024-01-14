"""This submodule contains the calculation functions for the Configuration Space obstacle representation."""
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pdmproject.cspace import SparseOccupancyTree

# TODO: put this somewhere better
R = np.array([0.2, 0.05, 0.05, 0.05, 0.05])  # Link radii
L = np.array([0.5, 0.025, 0.5, 0.25, 0.125, 0.1])  # Robot lengths
#             | h_base  | L2 | L3  |    L4    |
h_base = L[0] + L[1]  # base height

CSPACE_DIM = 3
MIN_VOXEL_STEP = 0.5


def calc_ns(collisions, link, params, limits):
    """Calculate the null space for a given link and collision point.

    Select the solution(s) from the 5D boundary manifold given by the input
    parameters.

    Args:
        collisions: A numpy array of one or two workspace points (rows) where
            the link is in collision. Ignores the second collision if solving
            for link 0.
        link: An index from 0-3 to select which link to use where the base is
            link 0. Link 4 is the robot hand but it is ignored for the
            purpose of calculating the null space because we model it as a
            cylinder.
        params: A numpy array of shape (N,5) of normalized parameters used to
            select the point(s) on the solution manifold.
        limits: A numpy array of shape (2,4) which indicates the ranges
            (minimum and maximum values) for joints q3 through q6.

    Returns:
        A numpy array of size (N,6) where configuration n corresponds to
        solution n from the input parameters.
    """
    N = np.shape(params)[0]  # number of data points

    # ensure input shapes are valid
    assert np.shape(params)[1] == CSPACE_DIM - 1
    assert np.shape(limits) == (2, CSPACE_DIM - 2)
    assert np.shape(collisions)[-1] == 3
    if link > 0:
        assert np.shape(collisions) == (2, 3)

    q = np.zeros((N, CSPACE_DIM))

    # get q3, q5, and q6 from parameters

    # TODO: restore this for higher dimensions
    # q_idx = np.array((3, 5, 6))
    q_idx = 3
    t_q = params[:, q_idx - 2]
    # TODO: restore this for higher dimensions
    # t_q4 = params[:, 2]
    t_q4 = 0
    q[:, q_idx - 1] = t_q * limits[1, q_idx - 3] + (1 - t_q) * limits[0, q_idx - 3]

    # calculate q
    if link > 0:
        # interpolate to get the collision
        t = params[:, 0]
        collision = t * collisions[1, :] + (1 - t) * collisions[0, :]
        h = collision[2] - h_base

        # calculate valid values for q_4
        r2_max = np.sum(L[2 : 2 + link])
        limits_q4 = np.pi / 2 + np.arccos(h / r2_max) * np.array([-1, 1])
        q[:, 3] = t_q4 * limits_q4[1] + (1 - t_q) * limits_q4[0]

        # calculate theta depending on the link
        phi = np.pi / 2 - q[:, 3]
        r_1 = h * np.tan(phi)
        theta = q[:, 2]
        if link > 1:
            r_2 = np.sqrt(r_1**2 + h**2)
            alpha_1 = q[:, 4] + np.pi
            if link > 2:
                r_3 = np.sqrt(L[2] ** 2 + r_2**2)
                alpha_2 = q[:, 5] + np.pi
                beta_2 = np.pi - alpha_2 - np.arcsin(L[3] * np.sin(alpha_2) / r_3)
                alpha_1 += beta_2

            beta_1 = np.pi - alpha_1 - np.arcsin(L[2] * np.sin(alpha_1) / r_2)
            theta -= beta_1
    else:
        # base
        r_1 = R[0]
        theta = params[:, 0] * 2 * np.pi
        collision = collisions[0, :]

        # use all valid values for q_4
        # TODO: restore this for higher dimensions
        # q[:, 3] = t_q4 * limits[1, 1] + (1 - t_q4) * limits[0, 1]

    q[:, 0] = collision[0] - r_1 * np.cos(theta)
    q[:, 1] = collision[1] - r_1 * np.sin(theta)
    return q


def dtheta_step(sample_space: "SparseOccupancyTree") -> float:
    """Calculate a fixed step size for theta (used only for link 0).

    Args:
        sample_space: A SparseOccupanyTree containing the voxels to iterate over.

    Returns:
        (float): The fixed theta step size.
    """
    voxel_size = (
        min(
            (sample_space.limits[1, :2] - sample_space.limits[0, :2])
            / (2**sample_space.res)
        )
        * MIN_VOXEL_STEP
    )
    return np.arcsin(voxel_size / R[0]) / (2 * np.pi)


def dt_step_l1(
    point: npt.NDArray,
    params: npt.NDArray,
    sample_space: "SparseOccupancyTree",
    collisions: npt.NDArray,
) -> np.float64:
    """Calculate the stepsize with respect to t for Link 1 collisions.

    Guarantees a maximum step size of one voxel in each dimension.

    Args:
        point: The last point which was evaluated.
        params: The last params which were evaluated.
        sample_space (SparseOccupanyTree): A SparseOccupanyTree containing the voxels to iterate over.
        collisions: The two collisions used for this update.

    Returns:
        float: The stepsize in t.
    """
    voxel_size = (
        (sample_space.limits[1, :2] - sample_space.limits[0, :2])
        / (2**sample_space.res)
    ) * MIN_VOXEL_STEP

    dq1 = voxel_size[0]
    dq2 = voxel_size[1]

    return np.min(np.array(dq1, dq2) / (collisions[1, :] - collisions[0, :]))


def dq3_step_l1(
    point: npt.NDArray,
    params: npt.NDArray,
    sample_space: "SparseOccupancyTree",
    collisions: npt.NDArray,
) -> float:
    """Calculate the stepsize in the q3 axis for Link 1 collisions.

    Guarantees a maximum step size of one voxel in each dimension.

    Args:
        point: The last point which was evaluated.
        params: The last params which were evaluated.
        sample_space (SparseOccupanyTree): A SparseOccupanyTree containing the voxels to iterate over.
        collisions: The two collisions used for this update.

    Returns:
        float: The stepsize in q3.
    """
    # interpolate to get the collision
    t = params[:, 0]
    collision = t * collisions[1, :] + (1 - t) * collisions[0, :]
    h = collision[2] - h_base

    voxel_size = (
        (sample_space.limits[1, :2] - sample_space.limits[0, :2])
        / (2**sample_space.res)
    ) * MIN_VOXEL_STEP

    dq1 = voxel_size[0]
    dq2 = voxel_size[1]

    return np.min(
        (
            np.abs(dq1 * np.tan(point[3]) / (h * np.sin(point[2]))),
            np.abs(-dq2 * np.tan(point[3]) / (h * np.cos(point[2]))),
        ),
        axis=0,
    )


def dq4_step_l1(
    point: npt.NDArray,
    params: npt.NDArray,
    sample_space: "SparseOccupancyTree",
    collisions: npt.NDArray,
) -> float:
    """Calculate the stepsize in the q4 axis for Link 1 collisions.

    Guarantees a maximum step size of one voxel in each dimension.

    Args:
        point: The last point which was evaluated.
        params: The last params which were evaluated.
        sample_space (SparseOccupanyTree): A SparseOccupanyTree containing the voxels to iterate over.
        collisions: The two collisions used for this update.

    Returns:
        float: The stepsize in q4.
    """
    # interpolate to get the collision
    t = params[:, 0]
    collision = t * collisions[1, :] + (1 - t) * collisions[0, :]
    h = collision[2] - h_base

    voxel_size = (
        (sample_space.limits[1, :2] - sample_space.limits[0, :2])
        / (2**sample_space.res)
    ) * MIN_VOXEL_STEP

    dq1 = voxel_size[0]
    dq2 = voxel_size[1]

    return np.min(
        (
            np.abs(dq1 * np.sin(point[3]) ** 2 / (h * np.cos(point[2]))),
            np.abs(dq2 * np.sin(point[3]) ** 2 / (h * np.sin(point[2]))),
        ),
        axis=0,
    )


def voxel_step(q, sample_space: "SparseOccupancyTree") -> float:
    """Calculate a fixed step size for one voxel in joint q.

    Args:
        q: The joint number.
        sample_space (SparseOccupancyTree): A SparseOccupanyTree containing the voxels to iterate over.

    Returns:
        float: the fixed stepsize fod one voxel
    """
    return 1 / (2**sample_space.res) * MIN_VOXEL_STEP


def dtheta_fixed(limits, resolution) -> float:
    """Calculate a fixed step size for theta (used only for link 0).

    Args:
        limits: A numpy array of shape (2,d) which indicates minimum and
            maximum values for each dimension. Defaults to [0,1] for every
            dimension.
        resolution: The highest tree depth. The number of cells along any
            given dimension will be 2^resolution.

    Returns:
        float: the fixed theta step size for link 0 iteration
    """
    voxel_size = min((limits[1, :2] - limits[0, :2]) / (2**resolution))
    return np.arcsin(voxel_size / R[0]) / (2 * np.pi * R[0])


def make_voxel_fixed(q):
    """Create a step function that moves one voxel at a time.

    Arguments:
        q: joint number

    Returns:
        A function which can be used by HypercubeIterator to get steps the size
        of one voxel for the q dimension.
    """

    def voxel_fixed(limits, resolution):
        return (limits[1, q - 1] - limits[0, q - 1]) / (2**resolution)

    return voxel_fixed


if __name__ == "__main__":
    pass
