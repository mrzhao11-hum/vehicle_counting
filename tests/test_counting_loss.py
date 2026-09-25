"""图像级相对计数损失的单元测试。"""

from __future__ import annotations

import unittest

import torch

from losses import RelativeCountSmoothL1Loss


class RelativeCountSmoothL1LossTest(unittest.TestCase):
    def test_exact_count_has_zero_loss(self) -> None:
        """密度积分等于真实计数时，损失必须严格为0。"""

        prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        criterion = RelativeCountSmoothL1Loss(offset=1.0, beta=0.1)
        loss = criterion(prediction, torch.tensor([10.0]))
        self.assertEqual(loss.item(), 0.0)

    def test_equal_under_and_over_count_are_symmetric(self) -> None:
        """相同幅度的少计和多计应受到相同惩罚。"""

        criterion = RelativeCountSmoothL1Loss(offset=1.0, beta=0.1)
        under = criterion(torch.full((1, 1, 2, 2), 2.25), torch.tensor([10.0]))
        over = criterion(torch.full((1, 1, 2, 2), 2.75), torch.tensor([10.0]))
        self.assertAlmostEqual(under.item(), over.item(), places=7)

    def test_valid_mask_excludes_ignored_region(self) -> None:
        """无效区域中的预测质量不能进入有效区域计数。"""

        prediction = torch.tensor([[[[2.0, 2.0], [100.0, 100.0]]]])
        valid_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        criterion = RelativeCountSmoothL1Loss()
        loss = criterion(
            prediction,
            torch.tensor([4.0]),
            valid_mask=valid_mask,
        )
        self.assertEqual(loss.item(), 0.0)

    def test_undercount_gradient_increases_density(self) -> None:
        """少计时梯度下降应推动所有有效密度像素增大。"""

        prediction = torch.ones((1, 1, 2, 2), requires_grad=True)
        criterion = RelativeCountSmoothL1Loss(offset=1.0, beta=0.1)
        loss = criterion(prediction, torch.tensor([8.0]))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.all(prediction.grad < 0.0).item())

    def test_zero_count_is_finite(self) -> None:
        """空场景由offset保护，不应产生除零或非有限值。"""

        prediction = torch.ones((1, 1, 2, 2))
        criterion = RelativeCountSmoothL1Loss(offset=1.0, beta=0.1)
        loss = criterion(prediction, torch.tensor([0.0]))
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
