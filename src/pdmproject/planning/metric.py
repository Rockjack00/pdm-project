"""This submodule contains metric calculations, which are JIT compiled using [numba](https://numba.pydata.org/)."""
import numpy as np
import numpy.typing as npt

from numba import float64, jit


@jit(float64(float64, float64), nopython=True)
def angle_metric(from_angle: float, to_angle: float) -> float:
    """Calculates shortest distance between two angles.

    Args:
        from_angle (float): angle1 in rad
        to_angle (float): angle2 in rad

    Returns:
        float: angle difference in rad
    """
    return (to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi

    # Different metric: Provided in BS announcement & Planning and decision making book p. 205 eq. 5.7
    # diff = np.abs(from_angle-to_angle)
    # return np.minimum(diff, 2*np.pi-diff)


@jit(float64[:](float64[:], float64[:]), nopython=True)
def difference(
    to_arr: npt.NDArray[np.float64], from_arr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates a vector between the given poses in C-space.

    Args:
        to_arr (NDArray[float64]): The target or to pose
        from_arr (NDArray[float64]): The from pose

    Returns:
        NDArray[float64]: The vector from the from to the to pose
    """
    diff = to_arr - from_arr

    # Doing it wrong (inverted diff[2::4]) makes it faster, however correcting for this loses the speed up???)
    diff[2::4] = (+diff[2::4] + np.pi) % (2 * np.pi) - np.pi

    # Different metric: Provided in BS announcement & Planning and decision making book p. 205 eq. 5.7
    # diff[2::4] = np.absolute(diff[2::4])
    # diff[2::4] = np.minimum(diff[2::4], 2*np.pi-diff[2::4])

    return diff


@jit(float64(float64[:], float64[:]), nopython=True)
def distance_metric(
    to_arr: npt.NDArray[np.float64], from_arr: npt.NDArray[np.float64]
) -> np.float64:
    """The distance between 2 poses in C-space.

    Args:
        to_arr (NDArray[float64]): The target or to pose
        from_arr (NDArray[float64]): The from pose

    Returns:
        np.float64 | float: The distance between the specified poses in C-space.
    """
    return np.sqrt(np.sum(difference(to_arr, from_arr) ** 2))


@jit(float64[:](float64[:], float64[:]), nopython=True)
def unit_vector(
    to_arr: npt.NDArray[np.float64], from_arr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates a unit vector that goes from one pose to another.

    Args:
        from_arr (NDArray[float64]): point in C-space
        to_arr (NDArray[float64]): point in C-space

    Returns:
        NDArray[float64]: The unit vector.
    """
    diff = difference(to_arr, from_arr)

    # The lenght must be 1.0 if 0.0, to allow division. This does not change the value.
    length = np.sqrt(np.sum(diff**2)) or 1.0
    diff /= length

    return diff
