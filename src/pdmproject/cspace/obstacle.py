import numpy as np

# TODO: put this somewhere better
R = [0.2, 0.05, 0.05, 0.05, 0.05] # Link radii
L = [0.5, 0.025, 0.5, 0.25, 0.125, 0.1] # Robot lengths
#    | h_base  | L2 | L3  |    L4    |
h_base = L[0] + L[1]            # base height


def calc_ns(collisions, link, params, limits):
    '''
    Calculate the null space for a given link and collision point.
    Select the solution(s) from the 5D boundary manifold given by the input 
    parameters.

    Arguments:
        collisions - A numpy array of one or two workspace points (rows) where 
            the link is in collision. Ignores the second collision if solving 
            for link 0.
        link - An index from 0-3 to select which link to use where the base is
            link 0. Link 4 is the robot hand but it is ignored for the
            purpose of calculating the null space because we model it as a 
            cylinder.
        params - A numpy array of shape (N,5) of normalized parameters used to 
            select the point(s) on the solution manifold.
        limits - A numpy array of shape (2,4) which indicates the ranges
            (minimum and maximum values) for joints q3 through q6.

    Returns:
        A numpy array of size (N,6) where configuration n corresponds to
        solution n from the input parameters.
    '''
    N = np.shape(params)[0] # number of data points

    # ensure input shapes are valid
    assert np.shape(params)[1] == 5
    assert np.shape(limits) == (2,4)
    assert np.shape(collisions)[-1] == 3
    if link > 0:
        assert np.shape(collisions) == (2,3)

    q = np.zeros(N,6)

    # get q3, q5, and q6 from parameters
    q_idx = (3,5,6)
    t_q = params[:,q_idx-2]
    t_q4 = params[:,2]
    q[:,q_idx-1] = t_q * limits[1,q_idx-3] + (1 - t_q) * limits[0,q_idx-3])

    # calculate q
    if link > 0:
        # interpolate to get the collision
        t = params[:,0]
        collision = t * collisions[1,:] + (1 - t) * collisions[0,:]
        h = collision[2] - h_base

        # calculate valid values for q_4
        r2_max = np.sum(L[2:2+link])
        limits_q4 = np.pi/2 + np.arccos(h/r2_max) * np.array([-1, 1])
        q[:,3] = t_q4 * limits_q4[1] + (1 - t_q) * limits_q4[0]

        # calculate theta depending on the link
        phi = np.pi/2 - q[:,3]
        r_1 = h * np.tan(phi)
        theta = q[:,2]
        if link > 1:
            r_2 = np.sqrt(r_1**2 + h**2)
            alpha_1 = q[:,4] + np.pi
            if link > 2:
                r_3 = np.sqrt(L[2]**2 + r_2**2)
                alpha_2 = q[:,5] + np.pi
                beta_2 = np.pi - alpha_2 - np.arcsin(L[3] * np.sin(alpha_2) / r_3)
                alpha_1 += beta_2

            beta_1 = np.pi - alpha_1 - np.arcsin(L[2] * np.sin(alpha_1) / r_2)
            theta -= beta_1
    else:
        # base
        r_1 = R[0]
        theta = params[:,0] * 2 * np.pi
        collision = collisions[0,:]

        # use all valid values for q_4
        q[:,3] = t_q4 * limits[1,1] + (1 - t_q) * limits[0,1])

    q[:,0] = collision[0] - r_1 * cos(theta)
    q[:,1] = collision[1] - r_1 * sin(theta)
    return q


# TODO: move this to the ns sampler maybe?
def update_sample_space(collisions, sample_space, link):
    '''
    A callback function to create a new cspace obstacle and insert it into
    the sample space. 

    WARNING: Currently only implemented for link 0.

    Arguments:
        collisions - A numpy array of one or two workspace points (rows) where 
            the link is in collision. Ignores the second collision if solving 
            for link 0.
        sample_space - A SparseOccupanyTree to insert the new collision into.
        link - An index from 0-3 to select which link to use where the base is
            link 0. Link 4 is the robot hand but it is ignored for the
            purpose of calculating the null space because we model it as a 
            cylinder.
    '''


    # TODO: parallelize this with multiple workers
    # may require load balancing

    midpoints = []
    node_stack = []

    # param generator generates sets of parameters [0,1] x 5 where each
    # of the 4-d "hyper-faces" is solved by fixing one of the parameters at 0 and 1
    # and getting a cartesean product of the rest.
    if link == 0:
        marcher = HypercubeIterator(dtheta_fixed,
                                    make_voxel_stepper(3), # remove these if not in the tree
                                    make_voxel_stepper(4), # remove these if not in the tree
                                    make_voxel_stepper(5), # remove these if not in the tree
                                    make_voxel_stepper(6)) # remove these if not in the tree
    # TODO
    if link >= 1:
        # cry
        raise NotImplementedError

    # using the generator, march over the null space boundary
    for params in marcher:
        points = calc_ns(collisions, params, limits, link)
        midpoints.append(points[len(points) // 2, :])
        voxels = sample_space.locate(points)

        for voxel in voxels:
            node_stack = sample_space.set(voxel, node_stack=node_stack)

    # TODO: get a smarter interior point or multiple interior points
    # multiple interior points would require load balancing between workers
    if link == 0:
        interior_point = np.zeros(6)
        interior_point[:2] = collisions[0,:2]
    else:
        interior_point = np.average(midpoints, axis=0)

    # fill the inside of the obstacle
    sample_space.flood_fill(interior_point)


class HypercubeIterator:
    '''
    An Iterator which cycles over an hypercube.
    This generates blocks of parameters to traverse each sub-manifold 
    lexicographically with a maximum step size at each point.
    '''

    def __init__(self, step_functions, resolution, limits=None, stepper=False):
        '''
        Create a HypercubeIterator.

        Arguments:
            step_functions - A list of functions to calculate the step size for
                each dimension to march over.
            resolution - The highest tree depth. The number of cells along any
                given dimension will be 2^resolution.
            limits - A numpy array of shape (2,d) which indicates minimum and
                maximum values for each dimension. Defaults to [0,1] for every
                dimension.
            stepper - If true, calculates the step size dynamically, otherwise
                uses a fixed step size. Currently not implemented.
        '''
        self.dimension = len(step_functions)
        self.current_dim = 0
        self.current_face = 0
        self.last_point = np.zeros((1,self.dimension))
        self.do_step = stepper
        self.step_functions = step_functions

        if limits is None:
            limits = np.indices((2,self.dimension))[0]
        self.limits = limits
        self.resolution = resolution
        # if self.do_step:
        #    assert isinstance(self.step_functions[i], Stepper)

    def __iter__(self):
        '''
        Reset this and return it.
        '''
        self.current_dim = 0
        self.current_face = 0
        return self

    def __next__(self):
        '''
        Get the next set of parameters to evaluate.

        Returns:
            A numpy array of shape (N,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube, one
            surface at a time. If this is a stepper, N=1 otherwise N is
            dependent on the step size for each dimension, but will return
            parameters for an entire face at once.
        '''
        # check if we are done
        if current_dim > self.dimesnion:
            self.current_dim = 0
            self.current_face = 0
            raise StopIteration
        
        # get all the parameters for this face
        params = self.march()

        if self.last_point[self.current_dim] >= self.limits[1,self.current_dim]:
            # change which face to march over
            if self.current_face == 0:
                self.current_face = 1
            else:
                self.current_face = 0
                self.current_dim += 1

        return params 

    def march(self):
        '''
        March forward, getting the next set of parameters.

        Returns:
            A numpy array of shape (N,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube, one
            surface at a time. If this is a stepper, N=1 otherwise N is
            dependent on the step size for each dimension, but will return
            parameters for an entire face at once.
        '''
        if self.do_step:
            return self._step()
        else:
            return self._sweep()

    # TODO: fix this to be recursive.  I ran out of brain.
    def _step(self):
        '''
        Return the next parameter, based on the step functions.

        WARNING: Not implemented yet.

        Returns:
            A numpy array of shape (1,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube.
        '''
        
        raise NotImplementedError

        last_param = limits[0,param]
        permutations = (np.arange((self.dimension - 1)**2) % self.dimension) + 1

        step_dim = 0
        #self.current_dim
        while step_dim < self.dimension:
            outer_param = permutations[i]
            if i == self.current_dim:
                continue

            # step in a direction until done
            done = False
            while not done:
                last_param += self.step_functions[step_dim](self.last_point)
                if last_param <= limits[1,param]:
                    self.last_point[i] = last_param
                else:
                    self.last_point[i] = limits[1,param]
                    done = True
                yield self.last_point

            # step once in the next direction and reset
            last_param += self.step_functions[step_dim](self.last_point)

            last_param = 0
            step_dim += 1

    def _sweep(self):
        '''
        March forward, getting the next set of parameters.

        Returns:
            A numpy array of shape (N,D) of normalized parameters used to
            select the parameters from a D-dimensional hypercube. When called
            successively, this will march around the entire hypercube, one
            surface at a time. N is dependent on the step size for each
            dimension, but will return parameters for an entire face at once.
        '''
        spaces = []
        for i in range(self.d):
            if i == self.current_dim:
                spaces.append([self.current_dir])
            fixed_delta = self.step_functions[i](self.limits, self.resolution)
            param_space = np.linspace(self.limits[0,i], self.limits[1,i], num=np.ceil(1 / fixed_delta))
            spaces.append(param_space)

        params = _cartesian_product(spaces)
        self.last_point = params[-1,:]
        return params

    def _cartesian_product(*arrays):
        '''
        Get the cartesian product of several arrays.
        Adopted from https://stackoverflow.com/a/11146645
        '''
        la = len(arrays)
        dtype = numpy.result_type(*arrays)
        arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(numpy.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)



def dtheta_fixed(limits, resolution):
    '''
    Calculate a fixed step size for theta (used only for link 0).

    Arguments:
        limits - A numpy array of shape (2,d) which indicates minimum and
            maximum values for each dimension. Defaults to [0,1] for every
            dimension.
        resolution - The highest tree depth. The number of cells along any
            given dimension will be 2^resolution.
    '''
    voxel_size = min((limits[1,:2] - limits[0,:2]) / (2**resolution))
    return np.arcsin(voxel_size / R[0]) / (2 * np.pi * R[0])


def make_voxel_fixed(q):
    '''
    Create a step function that moves one voxel at a time.

    Arguments:
        q - joint number

    Returns:
        A function which can be used by HypercubeIterator to get steps the size
        of one voxel for the q dimension.
    '''

    def voxel_fixed(limits, resolution):
        return (limits[1,q-1] - limits[0,q-1]) / (2**resolution)

    return voxel_fixed



def calc_ns2(collision, link, r=0, theta=0, phi=0, alpha_1=0, alpha_2=0):
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


def cylindrical_ns(collision, r, theta):
    N = np.shape(r)[0] # number of points

    # ensure input shapes are the same
    assert np.shape(theta)[0] == N

    q = np.zeros((N,2))
    q[:,0] = collision.x - r * cos(theta)
    q[:,1] = collision.y - r * sin(theta)
    
    return q
 

def update_sample_space2(collisions, sample_space, link):
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
        points = calc_ns2(collision, link, r, theta, phi, alpha_1, alpha_2)
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

# TODO: convert link equations to use these general functions
# def planar_joint_ns(collision, params, length, r_up, downstream=None):
#     N = np.shape(params)[0] # number of points
#     alpha = params[:,1]
# 
#     if downstream is None:
#         beta_down = 0
#         q = np.atleast_2d(alpha - np.pi) # range from -pi to pi
#     else:
#         r_down = np.sqrt(length**2 + r_up**2)   # radius from base of this link to collision (used downstream)
#         q_rest, beta_down = downstream(collision, params[:,1:], r_down)
#         q_first = np.atleast_2d(alpha - np.pi - beta_down) # range from -pi to pi
#         q = np.vstack(q_rest, q_first)
# 
#     beta_up = np.pi - alpha - np.arcsin(length * np.sin(alpha) / r_up)
#     return q, beta_up

# def base_ns(collision, r, theta):
#     return cylindrical_ns(collision, r, theta)
# 
# 
# def link_one_ns(collision, theta, phi):
#     N = np.shape(theta)[0] # number of points
# 
#     # ensure input shapes are the same
#     assert np.shape(phi)[0] == N
# 
#     # dimensions
#     h = collision.z - (L[0] + L[1])
#     r = h * np.tan(phi)
# 
#     # joint angles
#     q = np.zeros((N,4))
#     q[:,:2] = base_ns(collision, r, theta)
#     q[:,2] = theta # range from 0 to 2pi
#     q[:,3] = np.pi/2 - phi # range from 0 to pi
#     
#     return q
# 
# 
# def link_two_ns(collision, theta, phi, alpha):
#     N = np.shape(theta)[0] # number of points
# 
#     # ensure input shapes are the same
#     assert np.shape(phi)[0] == N
#     assert np.shape(alpha)[0] == N
# 
#     # dimensions
#     h = collision.z - (L[1] + L[1])
#     r_1 = h * np.tan(phi)
#     r_2 = np.sqrt(r_1**2 + h**2)
#     beta = np.pi - alpha - np.arcsin(L[2] * np.sin(alpha) / r_2)
# 
#     # joint angles
#     q = np.zeros((N,5))
#     q[:,:4] = link_one_ns(collision, theta, phi)
#     q[:,2] += beta
#     q[:,4] = alpha - np.pi # range from -pi to pi
#     
#     return q
# 
# 
# def link_three_ns(collision, theta, phi, alpha_1, alpha_2):
#     N = np.shape(theta)[0] # number of points
# 
#     # ensure input shapes are the same
#     assert np.shape(phi)[0] == N
#     assert np.shape(alpha_1)[0] == N
#     assert np.shape(alpha_2)[0] == N
# 
#     # dimensions
#     h = collision.z - (L[0] + L[1])
#     r_1 = h * np.tan(phi)
#     r_2 = np.sqrt(r_1**2 + h**2)
#     r_3 = np.sqrt(L[2]**2 + r_2**2)
#     beta_2 = np.pi - alpha_1 - np.arcsin(L[3] * np.sin(alpha_1) / r_3)
# 
#     # joint angles
#     q = np.zeros((N,6))
#     q[:,:5] = link_two_ns(collision, theta, phi, alpha_1)
#     q[:,4] -= beta_2
#     q[:,5] = alpha_2 - np.pi # range from -pi to pi
#     return q
# 
# 
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
