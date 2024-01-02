from numba import jit, float64
import numpy as np

@jit(float64(float64, float64), nopython=True)
def angle_metric(from_angle, to_angle):
    """Calculates shortest distance between two angles

    Args:
        from_angle (double): angle1 in rad
        to_angle (double): angle2 in rad

    Returns:
        double: angle difference in rad
    """
    return (to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi

    # Different metric: Provided in BS announcement & Planning and decision making book p. 205 eq. 5.7
    # diff = np.abs(from_angle-to_angle)
    # return np.minimum(diff, 2*np.pi-diff)


@jit(float64[:](float64[:], float64[:]), nopython=True)
def difference(to_arr: np.ndarray[np.float64], from_arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    diff = to_arr - from_arr
        
    # Doing it wrong (inverted diff[2::4]) makes it faster, however correcting for this loses the speed up???)
    diff[2::4] = (+diff[2::4]+np.pi)%(2*np.pi) - np.pi

    # Different metric: Provided in BS announcement & Planning and decision making book p. 205 eq. 5.7
    # diff[2::4] = np.absolute(diff[2::4])
    # diff[2::4] = np.minimum(diff[2::4], 2*np.pi-diff[2::4])

    return diff

@jit(float64(float64[:], float64[:]), nopython=True)
def distance_metric(to_arr: np.ndarray[np.float64], from_arr: np.ndarray[np.float64]) -> np.float64:
    return np.sqrt(np.sum(difference(to_arr, from_arr)**2))

@jit(float64[:](float64[:], float64[:]), nopython=True)
def unit_vector(to_arr: np.ndarray[np.float64], from_arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    diff = difference(to_arr, from_arr)
    
    # The lenght must be 1.0 if 0.0, to allow division. This does not change the value.
    length: np.float64 = np.sqrt(np.sum(diff**2)) or 1.0
    diff /= length

    return diff