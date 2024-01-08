import numpy as np


# TODO: put this somewhere better
R = [0.2, 0.05, 0.05, 0.05, 0.05] # Link radii
L = [0.5, 0.025, 0.5, 0.25, 0.125, 0.1] # Robot lengths

def calc_ns(collision, link, r=0, theta=0, phi=0, alpha_1=0, alpha_2=0):
    '''
    Calculate the null space for a given link and collision point
    Select the solution(s) returned are based on the input parameters. 
    Input parameters must either be a constant or column vectors of the same length.

    ---
    Arguments:
        collision - A workspace vector where the link is in collision.
        link - An index from 0-3 to select which link to use where the base is
            link 0. Link 4 is the robot hand but it is ignored for the
            purpose of calculating the null space because we model it as a 
            cylinder.

    Parameters, each a constant or a column vector of length N:
        r - Radius from joint 3 to collision. Ignored if link > 0.
        theta - Angle r_1 makes with XZ plane.
        phi - Azimuth angle from object to joint 4. Ignored if link < 1.
        alpha_1 - Interior angle between link 1 and r_2. Ignored if link < 2.
        alpha_2 - Interior angle between link 2 and r_3. Ignored if link < 3.
    ---
    Returns:
        A numpy array of size (N,6) where configuration n corresponds to
        solution n from the input parameters.
    ---
    '''

    N = np.shape(theta)[0] # number of data points

    # ensure input shapes are the same
    assert np.shape(r) == () or np.shape(r)[0] == N
    assert np.shape(phi) == () or np.shape(phi)[0] == N
    assert np.shape(alpha_1) == () or np.shape(alpha_1)[0] == N
    assert np.shape(alpha_2) == () or np.shape(alpha_2)[0] == N

    # base
    q = np.zeros((N,6))
    if link == 0:
        q[:,:2] = cylindrical_ns(collision, r, theta)
        return q

    h = collision.z - (L[0] + L[1])
    r_1 = h * np.tan(phi)
    q[:,:2] = cylindrical_ns(collision, r_1, theta)

    # link 1
    q[:,2] = theta # range from 0 to 2pi
    q[:,3] = np.pi/2 - phi # range from 0 to pi
    if link == 1:
        return q

    # link 2
    r_2 = np.sqrt(r_1**2 + h**2)
    q[:,2] += np.pi - alpha_1 - np.arcsin(L[2] * np.sin(alpha_1) / r_2)
    q[:,4] = alpha_1 - np.pi # range from -pi to pi
    if link == 2:
        return q

    # link 3 (and 4)
    r_3 = np.sqrt(L[2]**2 + r_2**2)
    q[:,4] -= np.pi - alpha_2 - np.arcsin(L[3] * np.sin(alpha_2) / r_3)
    q[:,5] = alpha_2 - np.pi # range from -pi to pi
    return q


def base_ns(collision, r, theta):
    return cylindrical_ns(collision, r, theta)


def link_one_ns(collision, theta, phi):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N

    # dimensions
    h = collision.z - (L[0] + L[1])
    r = h * np.tan(phi)

    # joint angles
    q = np.zeros((N,4))
    q[:,:2] = base_ns(collision, r, theta)
    q[:,2] = theta # range from 0 to 2pi
    q[:,3] = np.pi/2 - phi # range from 0 to pi
    
    return q


def link_two_ns(collision, theta, phi, alpha):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N
    assert np.shape(alpha)[0] == N

    # dimensions
    h = collision.z - (L[1] + L[1])
    r_1 = h * np.tan(phi)
    r_2 = np.sqrt(r_1**2 + h**2)
    beta = np.pi - alpha - np.arcsin(L[2] * np.sin(alpha) / r_2)

    # joint angles
    q = np.zeros((N,5))
    q[:,:4] = link_one_ns(collision, theta, phi)
    q[:,2] += beta
    q[:,4] = alpha - np.pi # range from -pi to pi
    
    return q


def link_three_ns(collision, theta, phi, alpha_1, alpha_2):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N
    assert np.shape(alpha_1)[0] == N
    assert np.shape(alpha_2)[0] == N

    # dimensions
    h = collision.z - (L[0] + L[1])
    r_1 = h * np.tan(phi)
    r_2 = np.sqrt(r_1**2 + h**2)
    r_3 = np.sqrt(L[2]**2 + r_2**2)
    beta_2 = np.pi - alpha_1 - np.arcsin(L[3] * np.sin(alpha_1) / r_3)

    # joint angles
    q = np.zeros((N,6))
    q[:,:5] = link_two_ns(collision, theta, phi, alpha_1)
    q[:,4] -= beta_2
    q[:,5] = alpha_2 - np.pi # range from -pi to pi
    return q


# TODO: convert link equations to use these general functions
def cylindrical_ns(collision, r, theta):
    N = np.shape(r)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(theta)[0] == N

    q = np.zeros((N,2))
    q[:,0] = collision.x - r * cos(theta)
    q[:,1] = collision.y - r * sin(theta)
    
    return q


def planar_joint_ns(collision, params, length, r_up, downstream=None):
    N = np.shape(params)[0] # number of points
    alpha = params[:,1]

    if downstream is None:
        beta_down = 0
        q = np.atleast_2d(alpha - np.pi) # range from -pi to pi
    else:
        r_down = np.sqrt(length**2 + r_up**2)   # radius from base of this link to collision (used downstream)
        q_rest, beta_down = downstream(collision, params[:,1:], r_down)
        q_first = np.atleast_2d(alpha - np.pi - beta_down) # range from -pi to pi
        q = np.vstack(q_rest, q_first)

    beta_up = np.pi - alpha - np.arcsin(length * np.sin(alpha) / r_up)
    return q, beta_up


def update_sample_space(collisions, sample_space, link):
    N_POINTS = 20
    ERROR = 1e-10
    h = collision.z - (L[0] + L[1])
    r2_max = np.sum(L[2:2+link] + ERROR)

    r = np.linspace(0,R[0],N_POINTS)
    theta = np.linspace(0, 2*np.pi, N_POINTS)
    phi = np.linspace(-np.arccos(h/r2_max), np.arccos(h/r2_max), N_POINTS)
    alpha_1 = np.linspace(np.pi + q5_min, np.pi - q5_max, N_POINTS)
    alpha_2 = np.linspace(np.pi + q6_min, np.pi - q6_max, N_POINTS)

    midpoints = []
    node_stack = []
    for collision in collisions:
        points = calc_ns(collision, link, r, theta, phi, alpha_1, alpha_2)
        midpoints.append(points[len(points) // 2, :])
        # TODO: connect points to ensure a smooth manifold.
        # Right now we have no guarantees that the regions don't have holes
        # so flood-fill might escape the boundary.
        voxels = sample_space.locate(points)

        # TODO: parallelize this with multiple workers
        # may require load balancing
        for voxel in voxels:
            node_stack = sample_space.set(voxel, node_stack=node_stack)

    # TODO: get a smarter interior point or multiple interior points
    # multiple interior points would require load balancing between workers
    interior_point = np.average(midpoints, axis=0)
    sample_space.flood_fill(interior_point)

# TODO: add functionality to generate full cspace objects
# class Obstacle:
#     '''
#     Obstacle boundary representation (brep) in configuration space.
#     '''
# 
#     def __init__(self, vertices):
#         '''
#         Create a configuration space Obstacle.
# 
#         Arguments:
#             vertices - An array of d-dimensional vertices which define an
#                 enclosed space.
#         '''
# 
#         self.content = 0                # High dimensional analog of volume
#         self.vertices = vertices        # Collection of boundary vertices (0-faces)
#         
# 
#     def __repr__():
#         pass
# 
#     def __lt__():
#         pass
# 
#     def __gt__():
#         pass


if __name__ == "__main__":
    pass
