"""CPU regression checks: python -m unittest test_delta_memory."""
import unittest
import os
from unittest.mock import patch

import numpy as np

import delta_memory as dm


class DeltaMemoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(4)
        self.inputs = [rng.randn(1, 2, 5, d) for d in (6, 3, 3, 4)]
        self.dy = rng.randn(1, 2, 5, 4)

    def test_gradients(self):
        _, cache = dm.forward(np, *self.inputs)
        gradients = dm.backward(np, self.dy, cache)
        for array, gradient in zip(self.inputs, gradients):
            for index in np.ndindex(array.shape):
                old = array[index]
                array[index] = old + 1e-5
                plus = (dm.forward(np, *self.inputs)[0] * self.dy).sum()
                array[index] = old - 1e-5
                minus = (dm.forward(np, *self.inputs)[0] * self.dy).sum()
                array[index] = old
                self.assertAlmostEqual((plus - minus) / 2e-5, gradient[index], places=7)

    def test_causality(self):
        original = dm.forward(np, *self.inputs)[0]
        changed = [a.copy() for a in self.inputs]
        for array in changed:
            array[:, :, 3:] += 10
        np.testing.assert_array_equal(original[:, :, :3], dm.forward(np, *changed)[0][:, :, :3])

    def test_repeated_write_corrects_instead_of_accumulating(self):
        phase = np.ones((1, 1, 40, 1))
        code = np.zeros((1, 1, 40, 1))
        value = np.ones_like(code)
        out, _ = dm.forward(np, phase, code, code, value)
        self.assertLess(abs(float(out[0, 0, -1, 0]) - 1), 1e-5)
        self.assertLessEqual(float(out.max()), 1.000001)

    @unittest.skipIf(dm._native is None, 'Optional C extension is not built')
    def test_native_matches_reference(self):
        for dtype, tolerance in ((np.float32, 2e-6), (np.float64, 1e-12)):
            inputs = [a.astype(dtype) for a in self.inputs]
            with patch.dict(os.environ, {'VEG_NONATIVE': ''}):
                actual, cache = dm.forward(np, *inputs)
                gradients = dm.backward(np, self.dy.astype(dtype), cache)
            with patch.dict(os.environ, {'VEG_NONATIVE': '1'}):
                expected, cache = dm.forward(np, *inputs)
                reference = dm.backward(np, self.dy.astype(dtype), cache)
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            for a, b in zip(gradients, reference):
                np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


if __name__ == '__main__':
    unittest.main()
