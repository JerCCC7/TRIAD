import unittest

import torch

from triad.rotation import quaternion_matrix, rotate_panorama
from train import mse_weight


class ProtocolTests(unittest.TestCase):
    def test_identity_preserves_erp(self):
        image = torch.rand(1, 3, 32, 64)
        rotated = rotate_panorama(image, torch.eye(3).unsqueeze(0))
        torch.testing.assert_close(rotated, image, atol=2e-5, rtol=0)

    def test_longitude_shift_and_wrap(self):
        import math
        image = torch.rand(1, 3, 32, 64)
        angle = 2 * math.pi * 8 / 64
        matrix = quaternion_matrix([0, 0, math.sin(angle / 2), math.cos(angle / 2)])
        rotated = rotate_panorama(image, matrix)
        torch.testing.assert_close(rotated, image.roll(8, -1), atol=2e-5, rtol=0)

    def test_invalid_quaternions(self):
        for q in [[0, 0, 0, 0], [float("nan"), 0, 0, 1], [0, 1]]:
            with self.assertRaises(ValueError):
                quaternion_matrix(q)

    def test_paper_loss_schedule(self):
        cfg = dict(ramp_start=101, ramp_end=200, lambda_mse_start=1, lambda_mse_end=20)
        self.assertEqual(mse_weight(1, cfg), 1)
        self.assertEqual(mse_weight(100, cfg), 1)
        self.assertEqual(mse_weight(101, cfg), 1)
        self.assertEqual(mse_weight(200, cfg), 20)
        self.assertEqual(mse_weight(300, cfg), 20)


if __name__ == "__main__":
    unittest.main()
