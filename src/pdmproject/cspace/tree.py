from dataclasses import dataclass
from itertools import takewhile
from typing import Optional, overload

import numpy as np
import numpy.typing as npt

from . import BinaryLeafNode, SparseVoxelTreeNode, TopologyNode

NodeStack = list[tuple[int, Optional[SparseVoxelTreeNode]]]


class SparseOccupancyTree:
    """A sparse occupancy tree analogous to a sparse voxel octree data structure.

    Voxels only store binary information and nodes store a count of the number of voxels set to True in each of their children.
    """

    def __init__(
        self,
        dimension,
        resolution,
        limits: Optional[npt.NDArray[np.float64]] = None,
        wraps=None,
    ):
        """Create a sparse occupancy tree.

        Args:
            dimension: The dimensionality of a voxel.
            resolution: The highest tree depth. The number of cells along any
                given dimension will be 2^resolution.
            limits: A numpy array of shape (2,d) which indicates minimum and
                maximum values for each dimension. Defaults to [0,1] for every
                dimension.
            wraps: A list of booleans which indicate which dimensions are
                homomorphic to S^1. Must be of length dimension. Defaults to
                False for all dimensions.
        """
        assert dimension > 0
        assert resolution > 0

        # TODO: remove this constraint?
        # z-order() and vector() rely on numpy bit manipulation which expect chars
        # so vectors must be 8 bits or fewer
        assert resolution <= 8, "Not implemented for resolution > 8"

        # TODO: remove this constraint?
        # Currently assume voxels keys fit inside a 28-byte (224-bit) BIG_NUM.
        assert (
            resolution * dimension <= 224
        ), "Voxel keysize must fit inisde a BIG_NUM (224-bits)"

        # Ensure limits and wraps are the correct size
        if limits is None:
            limits = np.indices((2, dimension))[0]
        if wraps is None:
            wraps = [False] * dimension
        assert limits is not None
        assert np.shape(limits) == (2, dimension)
        assert len(wraps) == dimension

        self.d = dimension  # Dimension of the tree
        self.res = resolution  # Highest tree depth
        self._root = TopologyNode(self.d)  # Root node
        self.limits: npt.NDArray[
            np.float64
        ] = limits  # Minimum and maximum values for each dimension
        self.wraps = wraps
        # TODO: add a node_stack here (and maybe depth?) to make traversal easier?
        self.filled_set = set()

    def __repr__(self):
        """The string representation of a 64-tree."""
        return f"({2 ** self.d}-Tree [res={self.res}, content={self._root.sum_values() / self.max_content(0):.2%}])"

    # TODO: maybe implement topology as a hash table, using the z-order as the hash code?
    # TODO: include a depth header for the keys so they can be used for nodes as well
    def z_order(self, p):
        """Calculate the z-order voxel key of a vector.

        Args:
            p: A numpy array of shape (N,d) position vectors of unsigned
                integers (normalized denominators) where d is the dimension
                of this tree and N is the number of vectors.

        Returns:
            A list of densely packed unsigned integers of tree traversal keys.
            Codes are ordered right-to-left where the most significant digit
            (finest voxel in the tree) is on the right and for each 'digit',
            the lowest dimension is on the right.
            |voxel |(lsb)            topology           (msb)|
            |654321|654321            [...]            654321| <-- dimension
            |------|------ ------ ------ ------ ------ ------|
        """
        # required for np.unpackbits
        assert self.res <= 8, "Not implemented for resolution > 8"
        assert len(np.shape(p)) == 2
        assert np.shape(p)[1] == self.d

        key_length = self.d * self.res  # length of voxel keys
        n_pad = 8 - self.res  # number of bits numpy pads onto each vector component

        padded_p = (p[:, None] << n_pad).astype(np.uint8)
        bits = np.unpackbits(padded_p, axis=1, count=self.res).reshape(-1, key_length)
        z = np.sum(bits * (2 ** np.arange(key_length)), axis=1)

        return z

    def locate(self, points: npt.NDArray) -> npt.NDArray[np.integer]:
        """Get the voxel(s) which encoloses the point(s).

        If a point lies on a voxel boundary, return the voxel with minimum value in the orthogonal direction.

        Args:
            points: A numpy array of shape (... , self.d) where the final dimension
                contains a 6 dimensional point to query.

        Return:
            locations - A numpy array of the same shape (excluding the last dimension)
                containing keys to the voxels enclosing each point.
        """
        query_shape = np.shape(points)[:-1]

        # normalize points within the valid ranges and convert to small integer quotients
        # NOTE: the denominator is 2**self.res
        norm_points = (np.reshape(points, (-1, self.d)) - self.limits[0, :]) / (
            self.limits[1, :] - self.limits[0, :]
        )
        norm_points = np.round(norm_points * (2**self.res), 0).astype(np.uint8)

        locations = self.z_order(norm_points)
        return np.reshape(np.array(locations), query_shape)

    def vector(self, voxels):
        """Calculate the vectors to the minimum vertex in voxels.

        Args:
            voxels: An array-like of integer z-order voxel keys.

        Returns:
            A numpy array of shape (N,d) position vectors of unsigned
            integers (normalized numerators) where d is the dimension
            of this tree and N is the number of vectors.
        """
        assert self.res <= 8, "Not implemented for resolution > 8"

        key_length = self.d * self.res  # length of voxel keys
        n_pad = 8 - self.res  # number of bits numpy pads onto each vector component

        voxels = np.array(voxels).flatten()
        bits = ((voxels[:, None] & (1 << np.arange(key_length))) > 0).reshape(
            -1, self.res, self.d
        )
        vectors = np.squeeze(np.packbits(bits, axis=1)) >> n_pad

        return vectors

    def max_content(self, depth):
        """Calculate the maximum content of a node at this depth.

        Args:
            depth: The depth of a node in the tree.
        """
        assert depth >= 0 and depth <= self.res
        return 2 ** ((self.res - depth) * self.d)

    @overload
    def make_node(self, voxel: int, depth: Optional[int] = None) -> tuple[int, int]:
        ...

    @overload
    def make_node(
        self, voxel: npt.NDArray[np.integer], depth: Optional[int] = None
    ) -> tuple[int, int]:
        ...

    def make_node(self, voxel, depth: Optional[int] = None):
        """Get the node_key for the parent node containing voxel.

        Args:
            voxel: A voxel key using the first depth indices.
            depth: The depth of the node. This is equal to self.res by default
                which gives the key for a single voxel.

        Returns:
            A voxel key using only the first depth indices.
        """
        depth = depth or self.res
        assert depth >= 0 and depth <= self.res

        return voxel & (2 ** (self.d * depth) - 1), depth

    def get_configurations(self, node_key: int, depth: Optional[int] = None):
        """Get a range of valid configurations contained in node.

        Args:
            node_key: A voxel key using the first depth indices.
            depth: The depth of the node. This is equal to self.res by default
                which sets a single voxel.

        Returns:
            A numpy array of shape (N,d) configuration, scaled from self.limits.
        """
        node_key, depth = self.make_node(node_key, depth)

        lower_bound_numerators = self.vector(node_key)
        upper_bound_numerators = 2 ** (self.res - depth) + lower_bound_numerators
        numerators = np.vstack((lower_bound_numerators, upper_bound_numerators))
        norm_bounds = numerators / (2**self.res)

        # linear interpolation
        limits = (
            norm_bounds * (self.limits[1, :] - self.limits[0, :]) + self.limits[0, :]
        )
        return limits

    def _insert(
        self, parent: TopologyNode, index: int, depth: int
    ) -> Optional[SparseVoxelTreeNode]:
        """A helper to insert a new child node at index.

        Args:
            parent: The parent node.
            index: The index of the new child node.
            depth: Depth of the child node.

        Returns:
            The inserted node.
        """
        leaf_depth = self.res - 1
        # The new node already exists or is a voxel
        if parent.values[index] != 0 or depth > leaf_depth:
            return None

        if depth < leaf_depth:
            next_node = TopologyNode(self.d)
        else:
            next_node = BinaryLeafNode(self.d)

        assert next_node is not None

        parent.children[index] = next_node
        parent.values[index] = next_node.sum_values()
        return next_node

    def _merge(self, key):
        """Merge all of the nodes children into one."""
        raise NotImplementedError

    def _traverse(
        self,
        node_stack: NodeStack,
        voxel: npt.ArrayLike,
        insert=False,
    ) -> NodeStack:
        """Walk from the node on the top of node_stack to the leaf node which contains the voxel.

        Assumes node_stack is up to date with the tree.

        Args:
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).
            voxel: An integer key of the voxel to traverse to. Ignores
                the voxel index in the leaf node - the top d bits.
            insert: Insert nodes if they do not exist (default is False).

        Returns:
            A new node_stack with the node containing the voxel of interest on
            top or its smallest ancestor. Guaranteed to have length from 1
            to self.res (inclusive).
        """
        assert (
            len(node_stack) <= self.res
        ), f"node_stack has more elements ({len(node_stack)}) than maximum tree depth ({self.res})."

        # build stack of ids
        # NOTE: the head of this stack is at 0 instead of -1
        id_mask = 2 ** (self.d) - 1
        id_stack = [(voxel >> self.d * i) & id_mask for i in range(self.res)]
        # node_id = NodeID(voxel, self.d)

        # Find the closest common ancestor
        i = len(
            list(
                (
                    takewhile(
                        lambda item: (
                            item[1][0] == id_stack[item[0]] and item[1][1] is not None
                        ),
                        enumerate(node_stack),
                    )
                )
            )
        )

        # navigate down branch
        node_stack = node_stack[:i]
        while i < self.res:
            if i < 1:
                cur_node = self._root
            else:
                cur_node = node_stack[i - 1][1]
            next_idx = id_stack[i]

            if i == self.res - 1:
                # reached the voxels
                next_node = None
            else:
                assert isinstance(cur_node, TopologyNode)
                next_node: Optional[SparseVoxelTreeNode] = cur_node.children[next_idx]
                # insert a new node if it doesn't exist
                if insert and next_node is None:
                    next_node = self._insert(cur_node, next_idx, i + 1)

            node_stack.append((next_idx, next_node))

            # return early if all the descendants have been set
            if next_node is None:
                break
            i += 1
        return node_stack

    @overload
    def is_full(
        self,
        node_key: int,
        depth: Optional[int] = None,
        node_stack: None = None,
    ) -> bool:
        ...

    @overload
    def is_full(
        self,
        node_key: int,
        depth: Optional[int] = None,
        node_stack: NodeStack = ...,
    ) -> tuple[bool, NodeStack]:
        ...

    def is_full(
        self,
        node_key: int,
        depth: Optional[int] = None,
        node_stack: Optional[NodeStack] = None,
    ) -> bool | tuple[bool, NodeStack]:
        """Check if a node is set.

        Args:
            node_key: A voxel key using the first depth indices.
            depth: The depth of the node. This is equal to self.res by default
                which sets a single voxel.
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Return:
            A boolean indicating if the node is full. If node_stack was
            provided, also return a new node_stack with the node of interest or
            its smallest full ancestor on top.
        """
        node_key, depth = self.make_node(node_key, depth)
        is_set = False

        # traverse from the last visited node to the voxel instead of walking from top
        return_stack = True
        if node_stack is None:
            return_stack = False
            node_stack = []
        node_stack = self._traverse(node_stack, node_key)

        if (node_key, depth) in self.filled_set:
            if return_stack:
                return True, node_stack
            return True

        # check if a voxel is set
        if depth == self.res and len(node_stack) == self.res:
            assert isinstance(node_stack[-2][1], BinaryLeafNode)
            is_set = node_stack[-2][1].values & (1 << node_stack[-1][0]) > 0

        # check if a node is filled
        else:
            parent_idx = min(depth, len(node_stack)) - 2
            if parent_idx < 0:
                parent = self._root
            else:
                parent: TopologyNode = node_stack[parent_idx][1]  # type: ignore
            is_set = parent.values[node_stack[parent_idx + 1][0]] >= self.max_content(
                depth
            )

        if is_set:
            self.filled_set.add((node_key, depth))

        if return_stack:
            return is_set, node_stack
        return is_set

    def paint(
        self,
        voxel,
        brush_size=1,
        directions=None,
        node_stack: Optional[NodeStack] = None,
    ):
        """Set the center voxel and brush_size of its nearest neighbors in the specified directions.

        Args:
            voxel: A voxel key in the center of the group to paint.
            brush_size: The number of neighboring voxels to set.
            directions: A collection of flags where each dimension has a
                positive and negative direction. Defaults to all directions.
                For 6 dimensions:       |654321 654321|
                                        |------ ++++++|
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).
        """
        if directions is None:
            directions = 2 ** (self.d * 2) - 1  # all directions
        if node_stack is None:
            node_stack = []
        node_stack = self._traverse(node_stack, voxel, insert=True)

        voxels = {voxel}
        i = 0
        while i < brush_size:
            new_voxels = {}
            for voxel in voxels:
                new_voxels.update(self.get_neighbors(voxel, directions))
            voxels = new_voxels
            i += 1

        node_stack = []
        for voxel in voxels:
            node_stack = self.set(voxel, node_stack=node_stack)

    def set(
        self,
        node_key: int,
        depth: Optional[int] = None,
        node_stack: Optional[list[tuple[int, Optional[SparseVoxelTreeNode]]]] = None,
    ):
        """Set all of the voxels contained in node to true and update counts.

        Performs merging operations on the tree if necessary.

        Args:
            node_key: A voxel key using the first depth indices.
            depth: The depth of the node. This is equal to self.res by default
                which sets a single voxel.
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Return:
            A new node_stack with the node of interest or its smallest full
            ancestor on top.
        """
        node_key, depth = self.make_node(node_key, depth)

        # traverse from the last visited node to the voxel instead of walking from top
        if node_stack is None:
            node_stack = []
        node_stack = self._traverse(node_stack, node_key, insert=True)
        is_set, node_stack = self.is_full(node_key, depth, node_stack)
        if is_set:
            return node_stack

        # update content values back up the tree
        # RESUME: subtract smaller node content from content when mergine
        content = self.max_content(depth)
        filled_depth = depth

        i = depth - 1
        while i >= 0:
            if i > 0:
                cur_node = node_stack[i - 1][1]
            else:
                cur_node = self._root
            next_idx = node_stack[i][0]
            child_max_content = self.max_content(i + 1)

            if isinstance(cur_node, BinaryLeafNode):
                # current node is a leaf so update voxel individually
                cur_node.increment(next_idx)
            elif cur_node.values[next_idx] + content < child_max_content:
                assert isinstance(cur_node, TopologyNode)
                # add content to ancestor nodes
                cur_node.values[next_idx] += content
            else:
                assert isinstance(cur_node, TopologyNode)
                # Prune the tree once a node is filled
                content = child_max_content - cur_node.values[next_idx]
                cur_node.children[next_idx] = None
                cur_node.values[next_idx] = child_max_content
                node_stack[i] = (next_idx, None)
                filled_depth = i + 1

            i -= 1
        return node_stack[:filled_depth]

    def unset(self, voxels):
        """Set all the voxels to false and update counts.

        Performs splitting operations on the tree if necessary.
        """
        raise NotImplementedError

    def get_neighbors(self, node_key, directions, depth=None):
        """Get the neighbor(s) at the same depth of a node in the specified directions.

        Args:
            node_key: A voxel key using the first depth indices.
            directions: A collection of flags where each dimension has a
                positive and negative direction.
                For 6 dimensions:       |654321 654321|
                                        |------ ++++++|
            depth: The depth of the node. This is equal to self.res by default
                which finds neighbors at the voxel level.

        Returns:
            A list of 2-tuples of (node_key, direction) for all the possible
            neighbors at the same depth as this node in the specified
            directions where node_key points to the minimum valued voxel
            contained under the neighboring node at depth node_depth. Order of
            the neighbors will be the reverse of direction flags
            (right-to-left).
        """
        node_key, depth = self.make_node(
            node_key, depth
        )  # ensure indices of lowwest depth are ignored

        modulus = 2**self.res
        inc = np.vstack((np.eye(self.d), -1 * np.eye(self.d))).astype(int)
        flags = 1 << np.arange(self.d * 2)

        inc <<= self.res - depth
        vector = self.vector(node_key)
        selected = (directions & flags) > 0
        neighbors = vector + inc[selected]
        neighbors[:, self.wraps] %= modulus
        valid = np.all(np.logical_and(neighbors >= 0, neighbors < modulus), axis=1)
        node_keys = self.z_order(neighbors[valid, :])
        selected_directions = flags[selected]
        return list(zip(node_keys, selected_directions[valid]))

    # TODO: what to do if the requested node doesn't exist?
    # right now: assumes it does and gives the neighbors it would have
    def get_smallest_neighbors(self, node_key, directions, depth=None):
        """Get all smallest neighbor(s) of a node in the specified directions.

        Args:
            node_key: A voxel key using the first depth indices.
            directions: A collection of flags where each dimension has a
                positive and negative direction.
                For 6 dimensions:       |654321 654321|
                                        |------ ++++++|
            depth: The depth of the node. This is equal to self.res by default
                which assumes node_key is a voxel key.

        Returns:
            A list of 2-tuples of (node_key, node_depth) for all the existing
            neighbors in the specified directions where node_key points to the
            minimum valued voxel contained under the neighboring node at depth
            node_depth.
        """
        node_key, depth = self.make_node(node_key, depth)
        # neighbors at the same tree depth
        queue = self.get_neighbors(node_key, directions, depth)

        neighbors = []
        node_stack = []
        for neighbor, direction in queue:
            node_stack = self._traverse(node_stack, neighbor)

            i = 0
            while i < depth and node_stack[i][1] is not None:
                i += 1
            if i < depth:
                # neighbor is same size or larger
                neighbors.append(self.make_node(neighbor, i + 1))
            else:
                # recursively get any smaller neighbors
                neighbors += self._get_smallest_neighbors(node_stack[:i], direction)
        return neighbors

    def _assemble_voxel_key(self, node_stack):
        """A helper to assemble a voxel key from a node stack.

        Args:
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).

        Returns:
            An integer voxel key which points to the minimum valued voxel
            contained under the top node on the stack.
        """
        key = 0
        for i in reversed(range(len(node_stack))):
            key <<= self.d
            key += node_stack[i][0]
        return key

    def _get_smallest_neighbors(self, node_stack: NodeStack, direction):
        """A helper to get all the smallest descendents of a node from the specified direction.

        Args:
            node_stack: A list of key/value pairs where the value is a node in
                the tree and the key is the index of the node from its parent.
                The node stack cannot contain more elements than the max tree
                depth (self.res).
            direction: A direction flag where each dimension has a positive
                and negative direction.
                For 6 dimensions:       |654321 654321|
                                        |------ ++++++|
        Returns:
            A list of 2-tuples of (node_key, node_depth) for all of the smallest
            descendants from the specified direction where node_key points to the
            minimum valued voxel contained under the descendant at depth
            node_depth.
        """
        depth = len(node_stack)
        # Base case: voxel level or empty child
        if node_stack[-1][1] is None:
            return [(self._assemble_voxel_key(node_stack), depth)]

        # Otherwise get children from lower levels
        children = []
        if direction < (2**self.d):
            # direction is + so get the - children
            children = np.array([i for i in range(2**self.d) if (i & direction) == 0])
        else:
            # direction is - so get the + children
            direction >>= self.d
            children = np.array([i for i in range(2**self.d) if (i & direction) > 0])

        neighbors = []
        for child_idx in children:
            if depth < self.res - 1:
                child = node_stack[-1][1].children[child_idx]
            else:
                # this is a leaf node so its children must be voxels
                child = None
            neighbors += self._get_smallest_neighbors(
                node_stack + [(child_idx, child)], direction
            )
        return neighbors

    # TODO: parallelize this
    # TODO: add fill directions? (requires splitting boundary nodes though...)
    def flood_fill(self, seed):
        """Fill an enclosed volume containing the seed point.

        Performs merging operations on the tree if necessary.

        Args:
            seed: A numpy column vector of size (1,self.d) to a point inside
                the region to fill.
        """
        directions = 2 ** (self.d * 2) - 1  # all directions
        seed_voxel = self.locate(seed)

        # Initialize the queue of nodes with the deepest unfilled ancestor
        node_stack = self._traverse([], seed_voxel)
        queue = [self.make_node(seed_voxel, len(node_stack))]

        # get all of the neighbors
        while len(queue) > 0:
            node_key, depth = queue.pop(0)
            is_set, node_stack = self.is_full(node_key, depth, node_stack)
            if is_set:
                continue
            queue += self.get_smallest_neighbors(node_key, directions, depth)
            node_stack = self.set(node_key, depth, node_stack)


@dataclass(slots=True, eq=False, frozen=True)
class NodeID:
    id: np.int64
    d: int

    def __getitem__(self, i) -> bool:
        return (self.id >> self.d * i) & (2 ** (self.d) - 1)

    def __int__(self) -> int:
        return int(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(int(self.id))


if __name__ == "__main__":
    pass
