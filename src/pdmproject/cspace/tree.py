import numpy as np
import numpy.typing as npt


class SparseBinaryTetrahexacontree:
    '''
    A sparse binary degree-64 analog to a sparse voxel octree data structure.
    This is the head of the tree, which implements tree traversal algorithms.
    Topological nodes and leaf nodes are implemented separately.
    '''

    d = 6                        # dimension of the tree
    # TODO: make the dimension arbirary?
    # All functions currently support this, however they assume voxels keys
    # fit inside a 64-bit integer.

    def __init__(self, 
                 resolution=7, 
                 limits=np.indices((2,SparseBinaryTetrahexacontree.d))[1],
                 wraps=[False] * SparseBinaryTetrahexacontree.d):
        '''
        Create a sparse binary 64-tree.

        Arguments:
            resolution - The highest tree depth. The number of cells along any
                given dimension will be 2^resolution.
            limits - A numpy array of shape (2,6) which indicates minimum and
                maximum values for each dimension.
            wraps - A list of booleans which indicate if a dimension is homomorphic to S^1.
        '''

        # TODO: remove this constraint?
        # make sure integers are long enough to hold the keys
        assert sys.maxsize.bit_length() + 1 >= self.res * self.d, \
            'Failed to initialize 64-tree. System must use 64-bit integers.'

        # TODO: remove this constraint?
        # z-order() and vector() rely on numpy bit manipulation which expect chars
        # so indexes must be 8 bits or fewer
        assert self.res > 8, 'Not implemented for resolution > 8'
        
        self.res = resolution                            # Highest tree depth
        self.max_content = 2 ** (self.d * resolution)
        self.content = 0                                 # High dimensional analog of volume of occupied regions
        self._root = TopologyNode(2 ** self.d)           # Root node
        self.limits = limits                             # Minimum and maximum values for each dimension
        self.wraps = wraps
        # TODO: add a node_stack here (and maybe depth?) to make traversal easier?

    def __repr__(self):
        '''
        The string representation of a 64-tree.
        '''
        return f'({2 ** self.d}-Tree [res={self.res}, content={100 * self.content / self.max_content}%])'

    def locate(self, points: npt.NDArray):
        '''
        Get the voxel(s) which encoloses the point(s). If a point lies on a
        voxel boundary, return the voxel with minimum value in the orthogonal
        direction.

        Arguments:
            points - A numpy array of shape (... , self.d) where the final dimension 
                contains a 6 dimensional point to query.

        Return:
            locations - A numpy array of the same shape (excluding the last dimension)
                containing keys to the voxels enclosing each point.
        '''

        query_shape = np.shape(points)[:-1] 
        
        # normalize points within the valid ranges and conver to small integer quotients
        # NOTE: the denominator is 2**self.res - 1
        norm_points = (np.reshape(points, (-1, self.d)) - self.limits[0,:]) / \
                        (self.limits[1,:] - self.limits[0,:])
        norm_points = np.round(norm_points * (2**self.res-1), 0, dtype=np.uint8)

        locations = self.z_order(norm_points)
        return np.reshape(np.array(locations), query_shape)

    # TODO: maybe implement topology as a hash table, using the z-order as the hash code?
    # TODO: include a depth header for the keys so they can be used for nodes as well
    def z_order(self, p):
        '''
        Calculate the z-order voxel key of a vector.

        Arguments:
            p - A numpy array of shape (N,d) position vectors of unsigned 
                integers (normalized denominators) where d is the dimension
                of this tree and N is the number of vectors.

        Returns:
            A list of densely packed unsigned integers of tree traversal keys. 
            Codes are ordered right-to-left (little-endian).
            |voxel |(lsb)           topology            (msb)|            
            |------|------ ------ ------ ------ ------ ------|
        '''
        assert self.res > 8, 'Not implemented for resolution > 8'
        assert len(np.shape(p)) == 2
        assert np.shape(p)[1] == self.d

        key_length = self.d * self.res              # length of voxel keys
        n_pad = 8 - self.res                        # number of bits numpy pads onto each vector component

        bits = np.unpackbits(vv[:,None] << n_pad, axis=1, count=self.res).reshape(-1, n_pad)
        z = np.sum(bits * (2**np.arange(key_length)), axis=1)

        return z

    def vector(self, voxels):
        '''
        Calculate the vectors to the minimum vertex in voxels.

        Arguments:
            voxels - An array-like of integer z-order voxel keys.

        Returns:
            A numpy array of shape (N,d) position vectors of unsigned 
            integers (normalized numerators) where d is the dimension
            of this tree and N is the number of vectors.
        '''
        assert self.res > 8, 'Not implemented for resolution > 8'

        key_length = self.d * self.res              # length of voxel keys
        n_pad = 8 - self.res                        # number of bits numpy pads onto each vector component

        voxels = np.array(voxels).flatten()
        bits = (((voxels[:,None] & (1 << np.arange(key_length)))) > 0).reshape(-1, self.res, self.d)
        vectors = np.squeeze(np.packbits(bits, axis=1)) >> n_pad

        return vectors

    def get_configurations(self, node_key, depth):
        '''
        Get a range of valid configurations contained in node.

        Arguments:
            node_key - A voxel key using the first depth indices.
            depth - The depth of the node. This is equal to self.res by default
                which sets a single voxel.

        Returns:
            A numpy array of shape (N,d) configuration, scaled from self.limits.
        '''
        node_key = self.get_node(node_key, depth)

        lower_bound_numerators = self.vector(node_key)
        upper_bound_numerators = 2**(self.res - depth) + lower_bound_numerators
        numerators = np.vstack((lower_bound_numerators, upper_bound_numerators))
        norm_bounds = numerators / (2**(self.res + 1))

        # linear interpolation
        limits = norm_bounds * (self.limits[1,:] - self.limits[0,:]) + self.limist[0,:]
        return limits

    def max_content(self, depth):
        '''
        Calculate the maximum content of a node at depth.

        Arguments:
            depth - The depth of a node in the tree. 
        '''
        assert depth > 0 and depth <= self.res
        return 2 ** ((self.res - depth) * self.d)

    def get_node(voxel, depth):
        '''
        Get the node_key for the parent node containing voxel.

        Arguments:
            voxel - A voxel key using the first depth indices.
            depth - The depth of the node. This is equal to self.res by default
                which sets a single voxel.

        Returns:
            A voxel key using only the first depth indices.
        '''
        return voxel & (2**(depth + 1) - 1)

    def _insert(self, parent, index, depth):
        '''
        A helper to insert a new child node at index.

        Arguments:
            parent - The parent node.
            index - The index of the new child node.
            depth - Depth of the parent node.

        Returns:
            The inserted node.
        '''
        leaf_depth = self.res - 1
        # The new node already exists or is a voxel
        if parent.values[index] != 0 or depth > leaf_depth:
            return None

        if depth < leaf_depth:
            next_node = TopologicalNode(self.d)
        else: 
            next_node = BinaryLeafNode(self.d)
        parent.children[index] = next_node
        parent.values[index] = next_node.sum_values()
        return next_node

    def _merge(self, key):
        '''
        Merge all of the nodes children into one.
        '''
        raise NotImplementedError

    def _traverse(self, node_stack, voxel_code, insert=False):
        '''
        Walk from the node on the top of node_stack to the leaf node which 
        contains the voxel at voxel_code.

        Arguments:
            node_stack - A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).
            voxel_code - An integer key of the voxel to traverse to. Ignores 
                the voxel index in the leaf node - the top d bits.
            insert - Insert nodes if they do not exist (default is False).

        Returns:
            A new node_stack with the node containing the voxel of interest on
            top or its smallest full ancestor. Guaranteed to have length from 1
            to self.res (inclusive).
        '''
        assert len(node_stack) <= self.res, \
            f'node_stack has more elements ({len(node_stack)}) than maximum tree depth ({self.res}).'

        # build stack of ids
        # NOTE: the head of this stack is at 0 instead of -1
        id_mask = 2**(self.d + 1) - 1
        id_stack = [(voxel_code >> self.d*i) & id_mask for i in range(self.res)]

        # Find the closest common ancestor
        i = 0
        while i < len(node_stack) and node_stack[i][0] == id_stack[i]:
            i += 1

        # navigate down branch
        node_stack = node_stack[:i]
        while i < self.res:
            if i < 1:
                cur_node = self._root
            else:
                cur_node = node_stack[i - 1][1]
            next_idx = id_stack[i]
            next_node = cur_node.children[next_idx]

            # insert a new node if it doesn't exist
            if insert and next_node is None:
                next_node = self._insert(cur_node, next_idx, i)
            node_stack.append((next_idx, next_node))

            # return early if all the descendants have been set
            if next_node is None:
                break
            i += 1

        return node_stack

    def set(self, node_key, depth=self.res, node_stack=None):
        '''
        Set all of the voxels contained in node to true and update counts. 
        Performs merging operations on the tree if necessary.

        Arguments:
            node_key - A voxel key using the first depth indices.
            depth - The depth of the node. This is equal to self.res by default
                which sets a single voxel.
            node_stack - A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Return:
            A new node_stack with the node of interest or its smallest full 
            ancestor on top.
        '''
        assert depth > 0 and depth <= self.res
        leaf_depth = self.res - 1
        node_key = self.get_node(node_key, depth)

        # traverse from the last visited node to the voxel instead of walking from top
        if node_stack is None:
            node_stack = []
        node_stack = self._traverse(node_stack, node_key, insert=True)
        is_set, node_stack = self.is_full(node_key, depth, node_stack)
        if is_set:
            return node_stack

        # update content values 
        content = self.max_content(depth)
        i = 0
        cur_node = self._root
        while i < min(depth, leaf_depth):
            next_idx = node_stack[i][0]
            cur_node.values[next_idx] += content

            # Prune the tree once a node is filled
            child_max_content = self.max_content(i + 1)
            if cur_node.values[next_idx] >= child_max_content:
                cur_node.child[next_idx] = None
                cur_node.values[next_idx] = child_max_content
                return node_stack[:i+1]

            # increment
            cur_node = node_stack[i][1]
            i += 1

        # current node is a leaf so update voxel individually
        if i == leaf_depth:
            cur_node.increment(next_idx)

        return node_stack

    def unset(self, voxels):
        '''
        Set all the voxels to false and update counts.
        Performs splitting operations on the tree if necessary.
        '''
        raise NotImplementedError

    def is_full(self, node_key, depth=self.res, node_stack=None):
        '''
        Check if a node is set.

        Arguments:
            node_key - A voxel key using the first depth indices.
            depth - The depth of the node. This is equal to self.res by default
                which sets a single voxel.
            node_stack - A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Return:
            A boolean indicating if the node is full. If node_stack was
            provided, also return a new node_stack with the node of interest or
            its smallest full ancestor on top.
        '''
        assert depth > 0 and depth <= self.res
        node_key = self.get_node(node_key, depth)
        is_set = False

        # traverse from the last visited node to the voxel instead of walking from top
        if node_stack is None:
            return_stack = True
            node_stack = []
        node_stack = self._traverse(node_stack, node_key)
        
        # check if a voxel is set
        if len(node_stack) == self.res:
            is_set = node_stack[-2][1].values & (1 << node_stack[-1][0]) 

        # check if a node is filled
        else:
            if len(node_stack) == 1:
                parent = self._root
            else:
                parent = node_stack[-2][1]
            is_set = parent.values[node_stack[-1][0]] >= self.max_content(depth)

        if return_stack:
            return is_set, node_stack
        return is_set

    def get_neighbors(self, node_key, directions, depth=self.res):
        '''
        Get the neighbor(s) at the same depth of a node in the specified directions.
        
        Arguments:
            node_key - A voxel key using the first depth indices.
            directions - A collection of flags where each dimension has a 
                positive and negative direction.
                For 6 dimensions:       |123456 123456|
                                        |++++++ ------|
            depth - The depth of the node. This is equal to self.res by default
                which finds neighbors at the voxel level.

        Returns:
            Voxel keys for all the existing neighbors in the specified 
            directions at the same depth as this node. Keys point to the minimum 
            valued voxel contained under this node. Keys will follow the order
            of direction flags (left-to-right).
        '''
        assert depth > 0 and depth <= self.res
        node_key = self.get_node(node_key, depth) # ensure indices of lowwe depth are ignored

        modulus = 2**(self.res + 1)
        inc = np.vstack((np.eye(self.d), -1 * np.eye(self.d))).astype(int)

        inc <<= self.res - depth
        vector = self.vector(node_key) 
        neighbors = vector + inc[(directions & (1 << np.arange(self.d * 2))) > 0]
        neighbors[:,self.wraps] %= modulus
        valid = np.logical_and(neighbors > 0, neighbors < modulus)
        return self.z_order(neighbors[np.all(valid, axis=1),:])

    def get_smallest_neighbors(self, node_key, directions, depth=self.res):
        '''
        Get all smallest neighbor(s) of a node in the specified directions.
        
        Arguments:
            node_key - A voxel key using the first depth indices.
            directions - A collection of flags where each dimension has a 
                positive and negative direction.
                For 6 dimensions:       |123456 123456|
                                        |++++++ ------|
            depth - The depth of the node. This is equal to self.res by default
                which assumes node_key is a voxel key.

        Returns:
            A list of 2-tuples of (node_key, node_depth) for all the neighbors
            in the specified directions where node_key points to the minimum
            valued voxel contained under the neighboring node at depth
            node_depth.
        '''
        node_key = self.get_node(node_key, depth)
        # neighbors at the same tree depth
        queue = self.get_neighbors(node_key, directions, depth)
        # return early if already at voxels
        # NOTE: this is just to save some computation time
        if depth == self.res:
            return zip(queue, [self.res] * len(queue))

        neighbors = []
        node_stack = []
        direction_flags = (directions & (1 << np.arange(self.d * 2))) > 0
        for i in range(len(queue)):
            neighbor = queue[i]
            node_stack = self._traverse(node_stack, neighbor)

            i = 0
            while i < depth and node_stack[i][1] is not None:
                i += 1
            if i < depth:
                # neighbor is larger
                neighbors.append((neighbor, i+1))
            else:
                # recursively get any smaller neighbors
                neighbors += self._get_smallest_neighbors(node_stack[:i], 
                                                          direction_flags[i])
        return neighbors

    def _assemble_voxel_key(self, node_stack):
        '''
        A helper to assemble a voxel key from a node stack.

        Arguments:
            node_stack - A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Returns:
            An integer voxel key which points to the minimum valued voxel
            contained under the top node on the stack.
        '''

        key = 0
        for i in reversed(range(len(node_stack))):
            key <<= self.d
            key += node_stack[i][0]
        return key

    def _get_smallest_neighbors(self, node_stack, direction):
        '''
        A helper to get all the smallest descendents of a node in the specified 
        direction.
        
        Arguments:
            node_stack - A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).
            direction - A direction flag where each dimension has a positive
                and negative direction.
                For 6 dimensions:       |123456 123456|
                                        |++++++ ------|
        Returns:
            A list of 2-tuples of (node_key, node_depth) for all of the smallest
            descendants in the specified direction where node_key points to the
            minimum valued voxel contained under the descendant at depth
            node_depth.
        '''
        # Base case: voxel level or empty child
        if node_stack[-1][1] is None:
            return [(self._assemble_voxel_key(node_stack), len(node_stack))]

        # Otherwise get children from lower levels
        if direction > (2 ** (self.d + 1)):
            branch = 1
            direction >>= self.d
        else:
            branch = 0
        children = np.array([i for i in range(2**(self.d + 1)) if (i & direction) == branch])
        
        neighbors = []
        for child_idx in children:
            child = node_stack[-1][1].children[child_idx]
            neighbors += self._get_smallest_neighbors(node_stack + (child_idx, child), direction)
        return neighbors

    # TODO: parallelize this
    def flood_fill(self, seed):
        '''
        Fill an enclosed volume containing the seed point.
        Performs merging operations on the tree if necessary.

        Arguments:
            seed - A numpy column vector of size (1,self.d) to a point inside 
                the region to fill.
        '''
        directions = np.sum(1 << np.arange(self.d*2)) # all directions
        seed_voxel = self.locate(seed)
        
        # Initialize the queue of nodes with the deepest unfilled ancestor
        node_stack = self._traverse([], seed_voxel)
        queue = [(seed_voxel, len(node_stack))]

        # get all of the neighbors
        while len(queue) > 0:
            node_key, depth = queue.pop(0)
            node_key = self.get_node(node_key, depth)
            if self.get(node_key, depth, node_stack):
                continue
            queue += self.get_smallest_neighbors(node_key, directions, depth)
            node_stack = self.set(node_key, depth, node_stack)

class TopologicalNode:
    '''
    A node in a sparse voxel tree data structure which contains topological 
    information.
    '''

    def __init__(self, dim):
        '''
        Create a Topological node in the tree.

        Arguments:
            dim - The number of children of this node.
        '''

        self.dim = dim                              # dimension of the tree
        self.children = [None] * (2**dim)           # A list of references to the children
        self.values = np.zeros(2**dim, dtype=int)   # A list of values assiciated with each child

    def __repr__(self):
        '''
        The string representation of a topological node.
        '''
        return f'({2 ** self.dim}-Node [content={self.sum_values()}, values={self.values}])'

    def increment(self, index):
        '''
        Increment the value of the child index.

        Arguments:
            index - The child index to set.

        Returns:
            The new sum of all values of children.
        '''

        self.values[index] += 1
        return self.sum_values()

    def sum_values(self):
        '''
        Get the sum of all values of the children
        '''
        return np.sum(self.values)


class BinaryLeafNode:
    '''
    A node in a sparse voxel tree data structure which contains binary 
    data for a collection of leaves. Each leaf is represented with one bit.
    '''

    def __init__(self, dim):
        '''
        Create a Binary Leaf node in the tree.

        Arguments:
           
            dim - The number of children of this node.
        '''

        self.dim = dim                        # dimension of the tree
        self.values = 0                       # A list of values assiciated with each voxel
        
    def __repr__(self):
        '''
        The string representation of a leaf node.
        '''
        return f'(BinaryLeafNode [content={self.sum_values()}, values={self.values:b}])'

    def increment(self, index):
        '''
        Set the value at index to 1.

        Arguments:
           index - The voxel index to set.

        Returns:
            The new sum of all values of children.
        '''

        self.values &= 1 << index
        return self.sum_values()

    def sum_values(self):
        '''
        Get the sum of all values of the children
        '''
        return self.values.bit_count()

if __name__ == "__main__":
    pass
