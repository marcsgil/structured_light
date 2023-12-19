import unittest
import numpy as np
import structures


class TestStructures(unittest.TestCase):

    def setUp(self):
        self.x = np.array([1, 2, 3])
        self.y = np.array([4, 5, 6])
        self.w0 = 1.0
        self.m = 1
        self.n = 1
        self.p = 1
        self.l = 1
        self.k = 1
        self.order = 1
        self.basis = 'hg'
        self.fx = 1.0
        self.fy = 1.0
        self.lamb = 1.0
        self.f = 1.0
        self.theta = 1.0
        self.a = 1.0
        self.b = 1.0
        self.l = 1.0
        self.d = 1.0
        self.radius = 1.0
        self.side_length = 1.0

    def test_hg(self):
        result = structures.hg(self.x, self.y, self.m, self.n, self.w0)
        self.assertEqual(result.shape, self.x.shape)

    def test_lg(self):
        result = structures.lg(self.x, self.y, self.p, self.l, self.w0)
        self.assertEqual(result.shape, self.x.shape)

    def test_diagonal_hg(self):
        result = structures.diagonal_hg(
            self.x, self.y, self.m, self.n, self.w0)
        self.assertEqual(result.shape, self.x.shape)

    def test_b(self):
        result = structures.b(self.m, self.n, self.k)
        self.assertIsInstance(result, float)

    def test_fixed_order_basis(self):
        result = structures.fixed_order_basis(
            self.x, self.y, self.w0, self.order, self.basis)
        self.assertEqual(result.shape, (self.order+1, self.x.shape[0]))

    def test_lens(self):
        result = structures.lens(self.x, self.y, self.fx, self.fy, self.lamb)
        self.assertEqual(result.shape, self.x.shape)

    def test_tilted_lens(self):
        result = structures.tilted_lens(
            self.x, self.y, self.f, self.theta, self.lamb)
        self.assertEqual(result.shape, self.x.shape)

    def test_rectangular_apperture(self):
        result = structures.rectangular_apperture(
            self.x, self.y, self.a, self.b)
        self.assertEqual(result.shape, self.x.shape)

    def test_square(self):
        result = structures.square(self.x, self.y, self.l)
        self.assertEqual(result.shape, self.x.shape)

    def test_single_slit(self):
        result = structures.single_slit(self.x, self.y, self.a)
        self.assertEqual(result.shape, self.x.shape)

    def test_double_slit(self):
        result = structures.double_slit(self.x, self.y, self.a, self.d)
        self.assertEqual(result.shape, self.x.shape)

    def test_pupil(self):
        result = structures.pupil(self.x, self.y, self.radius)
        self.assertEqual(result.shape, self.x.shape)

    def test_triangle(self):
        result = structures.triangle(self.x, self.y, self.side_length)
        self.assertEqual(result.shape, self.x.shape)


if __name__ == '__main__':
    unittest.main()
