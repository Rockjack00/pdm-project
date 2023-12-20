import numpy as np


def base_collision(collision, r, theta):
    return cylindrical_collision(collision, r, theta)


# TODO: put this somewhere better
L = [0.5 + 0.025, 0.5, 0.25, 0.125, 0.1] # Robot lengths

def link_one_collision(collision, theta, phi, beta=0):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N
    assert np.shape(beta) == () or np.shape(beta)[0] == N

    # dimensions
    h = collision.z - (L[1] + L[2])
    r = h * np.tan(phi)

    # joint angles
    q = np.zeros((N,4))
    q[:,:2] = base_collision(collision, r, theta)
    q[:,2] = theta + beta # range from 0 to 2pi
    q[:,3] = np.pi/2 - phi # range from 0 to pi
    
    return q


def link_two_collision(collision, theta, phi, alpha, beta_2=0):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N
    assert np.shape(alpha)[0] == N
    assert np.shape(beta_2) == () or np.shape(beta_2)[0] == N

    # dimensions
    h = collision.z - (L[1] + L[2])
    r_1 = h * np.tan(phi)
    r_2 = np.sqrt(r_1**2 + h**2)
    beta = np.pi - alpha - np.arcsin(L[2] * np.sin(alpha) / r_2)

    # joint angles
    q = np.zeros((N,5))
    q[:,:4] = link_one_collision(collision, theta, phi, beta)
    q[:,4] = alpha - np.pi - beta_2 # range from -pi to pi
    
    return q


def link_three_collision(collision, theta, phi, alpha_1, alpha_2):
    N = np.shape(theta)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(phi)[0] == N
    assert np.shape(alpha_1)[0] == N
    assert np.shape(alpha_2)[0] == N

    # dimensions
    h = collision.z - (L[1] + L[2])
    r_1 = h * np.tan(phi)
    r_2 = np.sqrt(r_1**2 + h**2)
    r_3 = np.sqrt(L[2]**2 + r_2**2)
    beta_2 = np.pi - alpha - np.arcsin(L[3] * np.sin(alpha) / r_3)

    # joint angles
    q = np.zeros((N,5))
    q[:,:5] = link_two_collision(collision, theta, phi, alpha_1, beta_2)
    q[:,5] = alpha - np.pi # range from -pi to pi
    return q




# TODO: convert link equations to use these general functions
def cylindrical_collision(collision, r, theta):
    N = np.shape(r)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(theta)[0] == N

    q = np.zeros((N,2))
    q[:,0] = collision.x - r * cos(theta)
    q[:,1] = collision.y - r * sin(theta)
    
    return q


def planar_joint_collision(collision, params, length, r_up, downstream=None):
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


# TODO: add functionality to generate full cspace objects
class Obstacle:
    '''
    Obstacle boundary representation (brep) in configuration space.
    '''

    def __init__(self, vertices):
        '''
        Create a configuration space Obstacle.

        Arguments:
            vertices - An array of d-dimensional vertices which define an
                enclosed space.
        '''

        self.content = 0                # High dimensional analog of volume
        self.vertices = vertices        # Collection of boundary vertices (0-faces)
        

    def __repr__():
        pass

    def __lt__():
        pass

    def __gt__():
        pass


# def cylindrical_collision(r, theta, q_rest=None, dim=6):
#     N = r.shape[0] # number of points
# 
#     # create default q_rest
#     if q_rest is None:
#         q = np.linspace(0,1,N)
#         q_rest = np.hstack(np.ones((N,dim)))
# 
#     # ensure shapes are the same
#     assert theta.shape[0] == N
#     assert ranges.shape[0] == N
# 
#     vertices = np.zeros((N,dim))



if __name__ == "__main__":
    pass
