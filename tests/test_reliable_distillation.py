"""B3可靠性空间加权的CPU单元测试。

运行方式：

    python -m unittest discover -s tests -p "test_reliable_distillation.py" -v

测试不需要CARPK数据和GPU，重点保证可靠性归一化、五列bbox兼容性以及
uniform模式退化为B2这三个实验公平性条件。
"""

from __future__ import annotations

import unittest

import torch

from losses import DensityMSELoss
from losses.object_reliability import InstanceMaskConfig
from losses.reliable_distillation import (
    ReliabilityDistillationConfig,
    build_batch_reliability_maps,
    normalize_reliability_map,
)


class ReliableDistillationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
        self.boxes = [
            torch.tensor(
                [
                    [4.0, 4.0, 12.0, 12.0, 1.0],
                    [18.0, 18.0, 28.0, 28.0, 1.0],
                ],
                dtype=torch.float32,
            )
        ]
        self.mask_config = InstanceMaskConfig(
            sigma_scale=0.35,
            sigma_min=0.75,
            sigma_max=4.0,
            truncate=2.5,
        )

    def make_config(
        self, mode: str, strength: float = 1.0
    ) -> ReliabilityDistillationConfig:
        return ReliabilityDistillationConfig(
            mode=mode,
            tau_map=0.1,
            tau_count=0.1,
            tau_consistency=0.1,
            minimum=0.05,
            background_value=0.05,
            normalize_mean=True,
            strength=strength,
            instance_mask=self.mask_config,
        )

    def test_normalization_has_unit_mean_on_valid_pixels(self) -> None:
        """可靠性归一化后只在有效像素上均值为1，无效区域必须为0。"""

        raw = torch.tensor([[0.1, 0.2], [0.4, 0.8]])
        valid = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        normalized = normalize_reliability_map(raw, valid)
        self.assertAlmostEqual(float(normalized[valid.bool()].mean()), 1.0, places=6)
        self.assertEqual(float(normalized[1, 1]), 0.0)

    def test_uniform_mode_exactly_recovers_b2_output_loss(self) -> None:
        """uniform模式应得到全1权重，因此输出蒸馏损失必须与B2一致。"""

        torch.manual_seed(7)
        student = torch.randn((1, 1, 8, 8))
        teacher = torch.randn((1, 1, 8, 8))
        target = torch.zeros_like(student)
        maps, statistics = build_batch_reliability_maps(
            teacher_density=teacher,
            target_density=target,
            boxes=self.boxes,
            input_size=(32, 32),
            valid_mask=self.valid_mask,
            config=self.make_config("uniform"),
        )
        criterion = DensityMSELoss(reduction="batch_mean_sum")
        b2_loss = criterion(student, teacher, valid_mask=self.valid_mask)
        b3_uniform_loss = criterion(
            student,
            teacher,
            weight=maps,
            valid_mask=self.valid_mask,
        )
        self.assertTrue(torch.allclose(maps, torch.ones_like(maps)))
        self.assertTrue(torch.allclose(b2_loss, b3_uniform_loss))
        self.assertAlmostEqual(statistics["normalized_map_mean"], 1.0, places=6)

    def test_perfect_teacher_gets_full_object_reliability(self) -> None:
        """教师与GT及翻转视图完全一致时，两个实例可靠性都应为1。"""

        target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target[0, 0, 2, 2] = 1.0
        target[0, 0, 5, 5] = 1.0
        maps, statistics = build_batch_reliability_maps(
            teacher_density=target,
            target_density=target,
            boxes=self.boxes,
            input_size=(32, 32),
            valid_mask=self.valid_mask,
            config=self.make_config("combined"),
            transformed_teacher_density=target,
        )
        self.assertEqual(tuple(maps.shape), (1, 1, 8, 8))
        self.assertAlmostEqual(statistics["object_reliability_mean"], 1.0, places=6)
        self.assertAlmostEqual(float(maps.mean()), 1.0, places=6)
        self.assertTrue(torch.isfinite(maps).all())

    def test_unreliable_teacher_is_downweighted_before_normalization(self) -> None:
        """教师漏掉目标且翻转不稳定时，实例原始可靠性应明显低于1。"""

        target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target[0, 0, 2, 2] = 1.0
        target[0, 0, 5, 5] = 1.0
        teacher = torch.zeros_like(target)
        transformed = torch.ones_like(target) * 0.05
        _, statistics = build_batch_reliability_maps(
            teacher_density=teacher,
            target_density=target,
            boxes=self.boxes,
            input_size=(32, 32),
            valid_mask=self.valid_mask,
            config=self.make_config("combined"),
            transformed_teacher_density=transformed,
        )
        self.assertLess(statistics["object_reliability_mean"], 1.0)
        self.assertGreaterEqual(statistics["object_reliability_mean"], 0.05)
        # 空间归一化只保持总梯度预算，不会把实例可靠性统计伪装回1。
        self.assertAlmostEqual(statistics["normalized_map_mean"], 1.0, places=6)

    def test_consistency_mode_requires_transformed_teacher(self) -> None:
        """需要稳定性信号的模式缺少翻转教师输出时必须尽早报错。"""

        density = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "必须提供变换视图"):
            build_batch_reliability_maps(
                teacher_density=density,
                target_density=density,
                boxes=self.boxes,
                input_size=(32, 32),
                valid_mask=self.valid_mask,
                config=self.make_config("consistency"),
            )

    def test_residual_strength_interpolates_b2_and_b3(self) -> None:
        """lambda应在B2全1权重和B3-v1可靠性权重之间线性插值。"""

        target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target[0, 0, 2, 2] = 1.0
        target[0, 0, 5, 5] = 1.0
        teacher = torch.zeros_like(target)
        transformed = torch.ones_like(target) * 0.05

        def build(strength: float) -> torch.Tensor:
            maps, _ = build_batch_reliability_maps(
                teacher_density=teacher,
                target_density=target,
                boxes=self.boxes,
                input_size=(32, 32),
                valid_mask=self.valid_mask,
                config=self.make_config("combined", strength=strength),
                transformed_teacher_density=transformed,
            )
            return maps

        b2_weights = build(0.0)
        mixed_weights = build(0.5)
        b3_v1_weights = build(1.0)

        self.assertTrue(torch.allclose(b2_weights, torch.ones_like(b2_weights)))
        self.assertTrue(
            torch.allclose(
                mixed_weights,
                0.5 * torch.ones_like(b3_v1_weights) + 0.5 * b3_v1_weights,
            )
        )
        self.assertAlmostEqual(float(mixed_weights.mean()), 1.0, places=6)
        self.assertLess(float(mixed_weights.max()), float(b3_v1_weights.max()))


if __name__ == "__main__":
    unittest.main()
