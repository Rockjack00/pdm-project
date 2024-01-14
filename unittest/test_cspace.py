import copy
import unittest

import numpy as np

from pdmproject.cspace.tree import BinaryLeafNode, SparseOccupancyTree, TopologyNode

# run tests in the order they are written
from order_tests import load_ordered_tests

load_tests = load_ordered_tests


class test_tree(unittest.TestCase):
    def setUp(self):
        self.dimension = 3
        self.resolution = 5
        self.tree = SparseOccupancyTree(self.dimension, self.resolution)

    def test_init(self):
        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(0, self.resolution)

        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(self.dimension, 0)

        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(1000, self.resolution)

        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(self.dimension, self.resolution, limits=[0, 1])

        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(self.dimension, self.resolution, limits=[0, 1])

        with self.assertRaises(AssertionError):
            _tree = SparseOccupancyTree(self.dimension, self.resolution, wraps=[0, 1])

    def test_z_order(self):
        points = np.array(
            [
                [0b10000, 0b11111, 0b11010],  # 16, 31, 26 / 32
                [0b01100, 0b10011, 0b00111],
            ]
        )  # 12, 19, 7  / 32
        z_orders = np.array([0b010110010110111, 0b110110101001010])
        self.assertTrue(np.all(self.tree.z_order(points) == z_orders))

    def test_locate(self):
        self.tree.limits = np.array([[-1, -1, -1], [31, 31, 31]])
        points = np.array(
            [
                [15.43, 29.75, 25],  # 0b10000,0b11111,0b11010
                [11, 18, 6],
            ]
        )  # 0b01100,0b10011,0b00111
        voxels = np.array([0b010110010110111, 0b110110101001010])
        out = self.tree.locate(points)
        self.assertEqual(np.shape(out), voxels.shape)
        self.assertTrue(np.all(out == voxels))

    def test_vector(self):
        points = np.array(
            [
                [0b10000, 0b11111, 0b11010],  # 16, 31, 26 / 32
                [0b01100, 0b10011, 0b00111],
            ]
        )  # 12, 19, 7  / 32
        voxels = [0b010110010110111, 0b110110101001010]
        self.assertTrue(np.all(self.tree.vector(voxels) == points))

    def test_max_content(self):
        with self.assertRaises(AssertionError):
            self.tree.max_content(6)
        with self.assertRaises(AssertionError):
            self.tree.max_content(-1)

        self.assertEqual(self.tree.max_content(5), 1)
        self.assertEqual(self.tree.max_content(4), 8**1)
        self.assertEqual(self.tree.max_content(3), 8**2)
        self.assertEqual(self.tree.max_content(2), 8**3)
        self.assertEqual(self.tree.max_content(1), 8**4)
        self.assertEqual(self.tree.max_content(0), 8**5)

    def test_make_node(self):
        voxels = [0b010110010110111, 0b110110101001010]
        expect = [(0b010110010110111, 5), (0b000000000001010, 2)]

        with self.assertRaises(AssertionError):
            self.tree.make_node(voxels[0], -1)
        with self.assertRaises(AssertionError):
            self.tree.make_node(voxels[0], 6)
        self.assertEqual(self.tree.make_node(voxels[0]), expect[0])
        self.assertEqual(self.tree.make_node(voxels[1], 2), expect[1])

    def test_get_configurations(self):
        self.tree.limits = np.array([[-1, -1, -1], [31, 31, 31]])
        points = np.array(
            [
                [15, 30, 25],  # 0b10000,0b11111,0b11010
                [11, 18, 6],
            ]
        )  # 0b01100,0b10011,0b00111
        z_orders = [0b010110010110111, 0b110110101001010]
        # voxel level
        lower = points
        upper = lower + 1
        z_orders = self.tree.locate(points)
        limits0 = self.tree.get_configurations(z_orders[0])
        self.assertTrue(np.all(limits0 == np.vstack((lower[0, :], upper[0, :]))))
        limits1 = self.tree.get_configurations(z_orders[1])
        self.assertTrue(np.all(limits1 == np.vstack((lower[1, :], upper[1, :]))))

        # depth = 2
        lower_shallow = np.array(
            [
                [15, 23, 23],  # 0b10000,0b11000,0b11000
                [7, 15, -1],
            ]
        )  # 0b01000,0b10000,0b00000
        upper_shallow = lower_shallow + 8
        limits0 = self.tree.get_configurations(z_orders[0], 2)
        self.assertTrue(
            np.all(limits0 == np.vstack((lower_shallow[0, :], upper_shallow[0, :])))
        )
        limits1 = self.tree.get_configurations(z_orders[1], 2)
        self.assertTrue(
            np.all(limits1 == np.vstack((lower_shallow[1, :], upper_shallow[1, :])))
        )

    def test__insert(self):
        index = 0b101010

        # NOTE: self.tree.res = 5 --> leaf_depth = 4
        parent = TopologyNode(6)
        out = self.tree._insert(parent, index, 3)
        self.assertFalse(out is None)
        self.assertTrue(isinstance(out, TopologyNode))
        self.assertEqual(parent.values[index], 0)
        self.assertEqual(parent._children[index], out)
        self.assertEqual(out.d, self.dimension)  # type: ignore
        # make sure it fails to reinsert
        parent.values[index] = 13
        out = self.tree._insert(parent, index, 3)
        self.assertTrue(out is None)

        # NOTE: self.tree.res = 5 --> leaf_depth = 4
        parent = TopologyNode(6)
        out = self.tree._insert(parent, index, 4)
        self.assertFalse(out is None)
        self.assertTrue(isinstance(out, BinaryLeafNode))
        self.assertEqual(parent.values[index], 0)
        self.assertEqual(parent._children[index], out)
        self.assertEqual(out.d, self.dimension)  # type: ignore

        # NOTE: self.tree.res = 5 --> leaf_depth = 4
        parent = TopologyNode(6)
        # make sure it failed to insert a voxel
        out = self.tree._insert(parent, index, 5)
        self.assertTrue(out is None)
        # make sure it failed to insert an already removed node
        parent.values[index] = 64
        out = self.tree._insert(parent, index, 4)
        self.assertTrue(out is None)

    def test__traverse(self):
        voxels = [
            0b010110010110111,  # 010 110 010 110 111
            0b110110101001010,  # 110 110 101 001 010
            0b000111101001010,
        ]  # 000 111 101 001 010
        # make sure it fails if the node stack is too long
        with self.assertRaises(AssertionError):
            self.tree._traverse([(0b000, None)] * 10, voxels[0])

        # traverse empty tree
        root = copy.deepcopy(self.tree._root)
        out = self.tree._traverse([], voxels[0])
        self.assertEqual(out, [(0b111, None)])
        self.assertTrue(np.all(root.values == self.tree._root.values))
        self.assertEqual(root._children, self.tree._root._children)

        out = self.tree._traverse([], voxels[1])
        self.assertEqual(out, [(0b010, None)])
        self.assertTrue(np.all(root.values == self.tree._root.values))
        self.assertEqual(root._children, self.tree._root._children)

        out = self.tree._traverse([], voxels[2])
        self.assertEqual(out, [(0b010, None)])
        self.assertTrue(np.all(root.values == self.tree._root.values))
        self.assertEqual(root._children, self.tree._root._children)

        # insert a new voxel
        node_stack_out = self.tree._traverse([], voxels[0], insert=True)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b111, 0b110, 0b010, 0b110, 0b010])
        self.assertEqual(self.tree._root.values[0b111], 0)
        self.assertEqual(self.tree._root._children[0b111], children[0])
        self.assertEqual(children[0].values[0b110], 0)
        self.assertEqual(children[0].children[0b110], children[1])
        self.assertEqual(children[1].values[0b010], 0)
        self.assertEqual(children[1].children[0b010], children[2])
        self.assertEqual(children[2].values[0b110], 0)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertFalse(children[3].values & (1 << 0b010))
        self.assertTrue(children[4] is None)

        # insert another new voxel
        node_stack_out = self.tree._traverse(node_stack_out, voxels[1], insert=True)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101, 0b110, 0b110])
        self.assertEqual(self.tree._root.values[0b010], 0)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(children[0].values[0b001], 0)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[1].values[0b101], 0)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[2].values[0b110], 0)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertFalse(children[3].values & (1 << 0b110))
        self.assertTrue(children[4] is None)

        # insert another new voxel
        node_stack_out = self.tree._traverse(node_stack_out, voxels[2], insert=True)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101, 0b111, 0b000])
        self.assertEqual(self.tree._root.values[0b010], 0)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(children[0].values[0b001], 0)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[1].values[0b101], 0)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[2].values[0b111], 0)
        self.assertEqual(children[2].children[0b111], children[3])
        self.assertFalse(children[3].values & (1 << 0b000))
        self.assertTrue(children[4] is None)

        # traverse to the second one again
        node_stack_out = self.tree._traverse(node_stack_out, voxels[1], insert=True)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101, 0b110, 0b110])
        self.assertEqual(self.tree._root.values[0b010], 0)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(children[0].values[0b001], 0)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[1].values[0b101], 0)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[2].values[0b110], 0)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertFalse(children[3].values & (1 << 0b110))
        self.assertTrue(children[4] is None)

        # traverse to a filled node of depth 3
        children[1].values[0b101] = 64
        children[1].children[0b101] = None
        node_stack_out = self.tree._traverse([], voxels[1], insert=True)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 3)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101])
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[1].values[0b101], 64)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertTrue(children[2] is None)

    def test_is_full(self):
        voxels = [
            0b010110010110111,  # 010 110 010 110 111
            0b110110101001010,  # 110 110 101 001 010
            0b000111101100010,
        ]  # 000 111 101 001 010

        # Test if an empty tree is full
        self.assertFalse(self.tree.is_full(voxels[0]))
        self.assertFalse(self.tree.is_full(voxels[0], depth=1))
        is_set, node_stack = self.tree.is_full(voxels[0], depth=3, node_stack=[])
        self.assertFalse(is_set)
        self.assertEqual(node_stack, [(0b111, None)])
        self.assertFalse(self.tree.is_full(voxels[0], depth=5))
        self.assertFalse(self.tree.is_full(voxels[1]))
        self.assertFalse(self.tree.is_full(voxels[2]))

        # Test if a voxel is full
        node_stack_out = self.tree._traverse([], voxels[0], insert=True)
        children = list(zip(*node_stack_out))[1]
        self.assertFalse(self.tree.is_full(voxels[0]))
        children[-2].values += 1 << 0b010
        self.assertTrue(self.tree.is_full(voxels[0]))

        # Test if a node of depth 3 is full
        node_stack_out = self.tree._traverse([], voxels[1], insert=True)
        children = list(zip(*node_stack_out))[1]
        self.assertFalse(self.tree.is_full(voxels[1], depth=1))
        self.assertFalse(self.tree.is_full(voxels[1], depth=3))
        self.assertFalse(self.tree.is_full(voxels[1], depth=5))
        children[1].values[0b101] = 64
        children[1].children[0b101] = None
        self.assertFalse(self.tree.is_full(voxels[1], depth=1))
        self.assertTrue(self.tree.is_full(voxels[1], depth=3))
        is_set, node_stack_out = self.tree.is_full(voxels[1], depth=5, node_stack=[])
        self.assertTrue(is_set)
        self.assertEqual(
            node_stack_out, [(0b010, children[0]), (0b001, children[1]), (0b101, None)]
        )

        # Test if an unfilled node with a filled voxel is full
        self.tree._root = TopologyNode(3)
        node_stack_out = self.tree._traverse([], voxels[2], insert=True)
        indices, children = list(zip(*node_stack_out))
        children[3].values |= 1 << indices[-1]
        self.assertTrue(self.tree.is_full(voxels[2], depth=5))
        self.assertFalse(self.tree.is_full(voxels[2], depth=3))
        self.assertFalse(self.tree.is_full(voxels[2], depth=1))

    def test_set(self):
        voxels = [
            0b010110010110111,  # 010 110 010 110 111
            0b110110101001010,  # 110 110 101 001 010
            0b000111101001010,
        ]  # 000 111 101 001 010

        # Set a voxel in an empty tree
        node_stack_out = self.tree.set(voxels[0])
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b111, 0b110, 0b010, 0b110, 0b010])
        self.assertEqual(self.tree._root.values[0b111], 1)
        self.assertEqual(self.tree._root._children[0b111], children[0])
        self.assertEqual(self.tree._root.sum_values(), 1)
        self.assertEqual(children[0].values[0b110], 1)
        self.assertEqual(children[0].children[0b110], children[1])
        self.assertEqual(children[1].values[0b010], 1)
        self.assertEqual(children[1].children[0b010], children[2])
        self.assertEqual(children[2].values[0b110], 1)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertTrue(children[3].values & (1 << 0b010))
        self.assertTrue(children[4] is None)

        # Try to set it again
        node_stack_out = self.tree.set(voxels[0])
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b111, 0b110, 0b010, 0b110, 0b010])
        self.assertEqual(self.tree._root.values[0b111], 1)
        self.assertEqual(self.tree._root._children[0b111], children[0])
        self.assertEqual(self.tree._root.sum_values(), 1)
        self.assertEqual(children[0].values[0b110], 1)
        self.assertEqual(children[0].children[0b110], children[1])
        self.assertEqual(children[1].values[0b010], 1)
        self.assertEqual(children[1].children[0b010], children[2])
        self.assertEqual(children[2].values[0b110], 1)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertTrue(children[3].values & (1 << 0b010))
        self.assertTrue(children[4] is None)

        # Set a voxel in an occupied tree
        node_stack_out = self.tree.set(voxels[1], node_stack=node_stack_out)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101, 0b110, 0b110])
        self.assertEqual(self.tree._root.values[0b010], 1)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(self.tree._root.sum_values(), 2)
        self.assertEqual(children[0].values[0b001], 1)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[0].sum_values(), 1)
        self.assertEqual(children[1].values[0b101], 1)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[1].sum_values(), 1)
        self.assertEqual(children[2].values[0b110], 1)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertEqual(children[2].sum_values(), 1)
        self.assertTrue(children[3].values & (1 << 0b110))
        self.assertEqual(children[3].sum_values(), 1)
        self.assertTrue(children[4] is None)

        # Set another voxel in a occupied tree
        node_stack_out = self.tree.set(voxels[2], node_stack=node_stack_out)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(len(node_stack_out), 5)
        self.assertEqual(list(indices), [0b010, 0b001, 0b101, 0b111, 0b000])
        self.assertEqual(self.tree._root.values[0b010], 2)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(self.tree._root.sum_values(), 3)
        self.assertEqual(children[0].values[0b001], 2)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[0].sum_values(), 2)
        self.assertEqual(children[1].values[0b101], 2)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[1].sum_values(), 2)
        self.assertEqual(children[2].values[0b111], 1)
        self.assertEqual(children[2].children[0b111], children[3])
        self.assertEqual(children[2].sum_values(), 2)
        self.assertTrue(children[3].values & (1 << 0b000))
        self.assertEqual(children[3].sum_values(), 1)
        self.assertTrue(children[4] is None)

        # Set a node at a depth of 3
        node_stack_out = self.tree.set(voxels[2], depth=3, node_stack=node_stack_out)
        children = list(zip(*node_stack_out))[1]
        self.assertEqual(len(node_stack_out), 3)
        self.assertEqual(self.tree._root.values[0b010], 64)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(self.tree._root.sum_values(), 65)
        self.assertEqual(children[0].values[0b001], 64)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[0].sum_values(), 64)
        self.assertEqual(children[1].values[0b101], 64)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[1].sum_values(), 64)
        self.assertTrue(children[2] is None)

        # Try to set it again
        node_stack_out = self.tree.set(voxels[2], depth=3, node_stack=node_stack_out)
        children = list(zip(*node_stack_out))[1]
        self.assertEqual(len(node_stack_out), 3)
        self.assertEqual(self.tree._root.values[0b010], 64)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(self.tree._root.sum_values(), 65)
        self.assertEqual(children[0].values[0b001], 64)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[0].sum_values(), 64)
        self.assertEqual(children[1].values[0b101], 64)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[1].sum_values(), 64)
        self.assertTrue(children[2] is None)

        # Try to set a voxel it contains
        node_stack_out = self.tree.set(voxels[2], depth=4, node_stack=node_stack_out)
        children = list(zip(*node_stack_out))[1]
        self.assertEqual(len(node_stack_out), 3)
        self.assertEqual(self.tree._root.values[0b010], 64)
        self.assertEqual(self.tree._root._children[0b010], children[0])
        self.assertEqual(self.tree._root.sum_values(), 65)
        self.assertEqual(children[0].values[0b001], 64)
        self.assertEqual(children[0].children[0b001], children[1])
        self.assertEqual(children[0].sum_values(), 64)
        self.assertEqual(children[1].values[0b101], 64)
        self.assertEqual(children[1].children[0b101], children[2])
        self.assertEqual(children[1].sum_values(), 64)
        self.assertTrue(children[2] is None)

        # Merge a filled voxel
        self.tree._root = TopologyNode(3)
        node_stack_out = []
        for i in range(8):
            voxel = (voxels[0] & 0b000111111111111) + (i << 12)
            node_stack_out = self.tree.set(voxel, node_stack=node_stack_out)
        self.assertEqual(len(node_stack_out), 4)
        indices, children = list(zip(*node_stack_out))
        self.assertEqual(list(indices), [0b111, 0b110, 0b010, 0b110])
        self.assertEqual(self.tree._root.values[0b111], 8)
        self.assertEqual(self.tree._root._children[0b111], children[0])
        self.assertEqual(self.tree._root.sum_values(), 8)
        self.assertEqual(children[0].values[0b110], 8)
        self.assertEqual(children[0].children[0b110], children[1])
        self.assertEqual(children[0].sum_values(), 8)
        self.assertEqual(children[1].values[0b010], 8)
        self.assertEqual(children[1].children[0b010], children[2])
        self.assertEqual(children[1].sum_values(), 8)
        self.assertEqual(children[2].values[0b110], 8)
        self.assertEqual(children[2].children[0b110], children[3])
        self.assertEqual(children[2].sum_values(), 8)
        self.assertTrue(children[3] is None)

    def test_get_neighbors(self):
        # TODO: get neighbors that wrap around
        voxels = [
            0b000000000000000,  # 000 000 000 000 000
            0b111000000000000,
        ]  # 111 000 000 000 000
        directions = [
            0b111111,  # all neighbors
            0b000001,  # +x
            0b000100,  # +z
            0b001000,  # -x
            0b100010,
        ]  # -z, +y

        # get all neighbors at voxel level
        neighbors = self.tree.get_neighbors(voxels[0], directions[0])
        self.assertEqual(
            neighbors,
            [
                (0b001000000000000, 0b000001),  # +x
                (0b010000000000000, 0b000010),  # +y
                (0b100000000000000, 0b000100),
            ],
        )  # +z
        neighbors = self.tree.get_neighbors(voxels[1], directions[0])
        self.assertEqual(
            neighbors,
            [
                (0b110001000000000, 0b000001),  # +x
                (0b101010000000000, 0b000010),  # +y
                (0b011100000000000, 0b000100),  # +z
                (0b110000000000000, 0b001000),  # -x
                (0b101000000000000, 0b010000),  # -y
                (0b011000000000000, 0b100000),
            ],
        )  # -z

        # get neighbors in specific directions at voxel level
        neighbors = self.tree.get_neighbors(voxels[0], directions[1])
        self.assertEqual(neighbors, [(0b001000000000000, directions[1])])
        neighbors = self.tree.get_neighbors(voxels[0], directions[2])
        self.assertEqual(neighbors, [(0b100000000000000, directions[2])])
        neighbors = self.tree.get_neighbors(voxels[0], directions[3])
        self.assertEqual(neighbors, [])
        neighbors = self.tree.get_neighbors(voxels[0], directions[4])
        self.assertEqual(neighbors, [(0b010000000000000, 0b000010)])  # +y only
        neighbors = self.tree.get_neighbors(voxels[1], directions[1])
        self.assertEqual(neighbors, [(0b110001000000000, directions[1])])
        neighbors = self.tree.get_neighbors(voxels[1], directions[2])
        self.assertEqual(neighbors, [(0b011100000000000, directions[2])])
        neighbors = self.tree.get_neighbors(voxels[1], directions[3])
        self.assertEqual(neighbors, [(0b110000000000000, directions[3])])
        neighbors = self.tree.get_neighbors(voxels[1], directions[4])
        self.assertEqual(
            neighbors,
            [
                (0b101010000000000, 0b000010),  # +y
                (0b011000000000000, 0b100000),
            ],
        )  # -z

        # get neighbors in all directions of a node at various depths
        neighbors = self.tree.get_neighbors(voxels[0], directions[0], depth=1)
        self.assertEqual(
            neighbors,
            [
                (0b000000000000001, 0b000001),  # +x
                (0b000000000000010, 0b000010),  # +y
                (0b000000000000100, 0b000100),
            ],
        )  # +z
        neighbors = self.tree.get_neighbors(voxels[1], directions[0], depth=1)
        self.assertEqual(
            neighbors,
            [
                (0b000000000000001, 0b000001),  # +x
                (0b000000000000010, 0b000010),  # +y
                (0b000000000000100, 0b000100),
            ],
        )  # +z
        neighbors = self.tree.get_neighbors(voxels[0], directions[0], depth=3)
        self.assertEqual(
            neighbors,
            [
                (0b000000001000000, 0b000001),  # +x
                (0b000000010000000, 0b000010),  # +y
                (0b000000100000000, 0b000100),
            ],
        )  # +z

    def test__assemble_voxel_key(self):
        voxels = [
            0b010110010110111,  # 010 110 010 110 111
            0b110110101001010,  # 110 110 101 001 010
            0b000000101001010,
        ]  # 000 000 101 001 010

        # assemble voxel key for full stack
        node_stack = [
            (0b111, TopologyNode(self.dimension)),
            (0b110, TopologyNode(self.dimension)),
            (0b010, TopologyNode(self.dimension)),
            (0b110, BinaryLeafNode(self.dimension)),
            (0b010, None),
        ]
        self.assertEqual(self.tree._assemble_voxel_key(node_stack), voxels[0])
        node_stack = [
            (0b010, TopologyNode(self.dimension)),
            (0b001, TopologyNode(self.dimension)),
            (0b101, TopologyNode(self.dimension)),
            (0b110, BinaryLeafNode(self.dimension)),
            (0b110, None),
        ]
        self.assertEqual(self.tree._assemble_voxel_key(node_stack), voxels[1])

        # assemble voxel key for partial stack
        node_stack = [
            (0b010, TopologyNode(self.dimension)),
            (0b001, TopologyNode(self.dimension)),
            (0b101, None),
        ]
        self.assertEqual(self.tree._assemble_voxel_key(node_stack), voxels[2])
        self.assertEqual(self.tree._assemble_voxel_key([]), 0b000000000000000)

    def test_get_smallest_neighbors(self):
        # TODO: what to do if the requested node doesn't exist?
        # right now: assumes it does and gives the neighbors it would have,
        # but that wouldn't be valid as soon as it gets inserted. maybe that
        # doesnt matter?

        voxels = [
            0b000000000000000,  # 000 000 000 000 000
            0b111000000000000,  # 111 000 000 000 000
            0b000000000000111,  # 000 000 000 000 111
            0b000001000000000,  # 000 001 000 000 000
            0b000000000000011,  # 000 000 000 000 011
            0b000000000000100,
        ]  # 000 000 000 000 100
        directions = [
            0b111111,  # all neighbors
            0b001000,  # -x
            0b110100,
        ]  # -z, -y, +z

        self.tree.set(voxels[1])
        self.tree.set(voxels[2])

        # get all smallest neighbors for voxels
        neighbors = self.tree.get_smallest_neighbors(voxels[0], directions[0])
        self.assertEqual(
            set(neighbors),
            {(0b001000000000000, 5), (0b010000000000000, 5), (0b100000000000000, 5)},
        )
        neighbors = self.tree.get_smallest_neighbors(voxels[1], directions[0])
        self.assertEqual(
            set(neighbors),
            {
                (0b000001000000000, 4),  # +x
                (0b000010000000000, 4),  # +y
                (0b000100000000000, 4),  # +z
                (0b110000000000000, 5),  # -x
                (0b101000000000000, 5),  # -y
                (0b011000000000000, 5),
            },
        )  # -z
        neighbors = self.tree.get_smallest_neighbors(voxels[2], directions[0])
        self.assertEqual(
            set(neighbors),
            {
                (0b001000000000111, 5),
                (0b010000000000111, 5),
                (0b100000000000111, 5),
                (0b000000000000110, 1),
                (0b000000000000101, 1),
                (0b000000000000011, 1),
            },
        )

        # get all smallest neighbors for nodes at some depth
        neighbors = self.tree.get_smallest_neighbors(voxels[2], directions[0], depth=1)
        self.assertEqual(
            set(neighbors),
            {(0b000000000000110, 1), (0b000000000000101, 1), (0b000000000000011, 1)},
        )
        neighbors = self.tree.get_smallest_neighbors(voxels[0], directions[0], depth=4)
        self.assertEqual(
            set(neighbors),
            {(0b000001000000000, 4), (0b000010000000000, 4), (0b000100000000000, 4)},
        )

        # get smallest neighbors in some direction for nodes at some depth
        neighbors = self.tree.get_smallest_neighbors(voxels[3], directions[1], depth=4)
        self.assertEqual(
            set(neighbors),
            {
                (0b001000000000000, 5),
                (0b011000000000000, 5),
                (0b101000000000000, 5),
                (0b111000000000000, 5),
            },
        )
        neighbors = self.tree.get_smallest_neighbors(voxels[4], directions[2], depth=1)
        self.assertEqual(
            set(neighbors),
            {
                (0b000000000000111, 5),
                (0b010000000000111, 5),
                (0b001000000000111, 5),
                (0b011000000000111, 5),
                (0b000010000000111, 4),
                (0b000001000000111, 4),
                (0b000011000000111, 4),
                (0b000000010000111, 3),
                (0b000000001000111, 3),
                (0b000000011000111, 3),
                (0b000000000001111, 2),
                (0b000000000010111, 2),
                (0b000000000011111, 2),  # ^^^ +z neighbors
                (0b000000000000001, 1),
            },
        )  # -y neighbor, no -z neighbors

        # ensure that the smallest neighbors match their appropriate directions
        self.tree._root = TopologyNode(3)

        for i in range(1, 8):
            node_key = i << 3
            self.tree.set(node_key, depth=2)
        assert self.tree._root.sum_values() == 7 * (8**3)  # testing the tests
        assert not self.tree.is_full(voxels[0], depth=3)
        neighbors = self.tree.get_smallest_neighbors(voxels[5], 0b111111, depth=1)
        self.assertEqual(
            set(neighbors),
            {
                (0b000000000000101, 1),  # +x
                (0b000000000000110, 1),  # +y
                (0b000000000100000, 2),
                (0b000000000101000, 2),
                (0b000000000110000, 2),
                (0b000000000111000, 2),
            },
        )  # ^^^ -z neighbors

    def test_flood_fill(self):
        seeds = (
            np.array(
                [
                    [0b00000, 0b00000, 0b00000],  #  0,  0,  0
                    [0b10000, 0b11111, 0b11010],  # 16, 31, 26 / 32
                    [0b01100, 0b10011, 0b00111],
                ]
            )
            / 32
        )  # 12, 19,  7 / 32

        voxels = [
            0b000000000000000,  # 000 000 000 000 000
            0b111000000000000,  # 111 000 000 000 000
            0b000000000000111,
        ]  # 000 000 000 000 111

        # fill the whole tree
        self.tree.flood_fill(seeds[0])
        self.assertEqual(self.tree._root.sum_values(), 8**5)
        self.tree._root = TopologyNode(3)

        self.tree.flood_fill(seeds[1])
        self.assertEqual(self.tree._root.sum_values(), 8**5)
        self.tree._root = TopologyNode(3)

        self.tree.set(voxels[0])
        self.tree.flood_fill(seeds[2])
        self.assertEqual(self.tree._root.sum_values(), 8**5)
        self.tree._root = TopologyNode(3)

        # fail to fill a filled node
        self.tree.set(voxels[0])
        self.tree.flood_fill(seeds[0])
        self.assertEqual(self.tree._root.sum_values(), 1)

        self.tree.set(voxels[1], depth=4)
        self.tree.flood_fill(seeds[0])
        self.assertEqual(self.tree._root.sum_values(), 8)
        self.tree._root = TopologyNode(3)

        # fill a closed volume
        for i in range(1, 8):
            node_key = i << 6
            self.tree.set(node_key, depth=3)
        self.tree.set(voxels[1])
        assert self.tree._root.sum_values() == 7 * (8**2) + 1  # testing the tests
        assert not self.tree.is_full(voxels[0], depth=3)
        self.tree.flood_fill(seeds[0])
        self.assertEqual(self.tree._root.sum_values(), 8**3)
        self.tree._root = TopologyNode(3)

        for i in range(1, 8):
            node_key = i << 3
            self.tree.set(node_key, depth=2)
        assert self.tree._root.sum_values() == 7 * (8**3)  # testing the tests
        assert not self.tree.is_full(voxels[0], depth=3)
        # breakpoint()
        self.tree.flood_fill(seeds[1])
        self.assertEqual(self.tree._root.sum_values(), 8**5 - 8**3)
        self.tree._root = TopologyNode(3)

        # fill in specified directions
        # self.tree.flood_fill(seeds[0], directions=0b001000)


if __name__ == "__main__":
    unittest.main()
