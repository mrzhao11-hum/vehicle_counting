"""车辆实例可靠性基础函数的轻量单元测试。

运行方式：

    python -m unittest tests.test_object_reliability

测试不需要CARPK数据或GPU，用于在服务器正式分析前检查实例区域、误差和
可靠性函数的基本数值性质。
"""

from __future__ import annotations

import unittest

import torch

from losses.object_reliability import (
    InstanceMaskConfig,
    build_soft_instance_regions,
    compute_instance_errors,
    rasterize_instance_values,
    reliability_from_errors,
)


class ObjectReliabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.boxes = torch.tensor(
            [
                # CARPK 的真实标注格式为 x1,y1,x2,y2,class_id。测试直接
                # 使用五列输入，确保类别列不会被误当成坐标参与计算。
                [8.0, 8.0, 24.0, 24.0, 1.0],
                [18.0, 8.0, 34.0, 24.0, 1.0],
            ]
        )
        self.regions = build_soft_instance_regions(
            self.boxes,
            input_size=(64, 64),
            output_size=(16, 16),
            config=InstanceMaskConfig(
                sigma_scale=0.35,
                sigma_min=0.75,
                sigma_max=4.0,
                truncate=2.5,
            ),
        )

    def test_overlapping_regions_form_soft_ownership(self) -> None:
        """重叠像素的实例归属之和应为1，背景处应为0。"""

        ownership_sum = self.regions.ownership.sum(dim=0)
        foreground = self.regions.foreground.bool()
        self.assertTrue(torch.allclose(ownership_sum[foreground], torch.ones_like(ownership_sum[foreground])))
        self.assertTrue(torch.allclose(ownership_sum[~foreground], torch.zeros_like(ownership_sum[~foreground])))

    def test_perfect_teacher_has_zero_error_and_full_reliability(self) -> None:
        """教师与GT完全相同时，所有误差应为0且可靠性应为1。"""

        target = torch.zeros((16, 16))
        target[4, 4] = 1.0
        target[4, 6] = 1.0
        errors = compute_instance_errors(
            target,
            target,
            self.regions,
            transformed_teacher_density=target,
        )
        self.assertTrue(torch.allclose(errors["local_map_mae"], torch.zeros(2)))
        self.assertTrue(torch.allclose(errors["local_count_error"], torch.zeros(2)))
        self.assertTrue(torch.allclose(errors["view_consistency_mae"], torch.zeros(2)))

        reliability = reliability_from_errors(
            errors["local_map_mae"],
            errors["local_count_error"],
            tau_map=0.1,
            tau_count=0.1,
            view_consistency_error=errors["view_consistency_mae"],
            tau_consistency=0.1,
        )
        self.assertTrue(torch.allclose(reliability["combined"], torch.ones(2)))

    def test_missing_prediction_reduces_reliability(self) -> None:
        """教师漏掉目标后，实例误差应增大且可靠性低于1。"""

        target = torch.zeros((16, 16))
        target[4, 4] = 1.0
        target[4, 6] = 1.0
        teacher = torch.zeros_like(target)
        errors = compute_instance_errors(teacher, target, self.regions)
        reliability = reliability_from_errors(
            errors["local_map_mae"],
            errors["local_count_error"],
            tau_map=0.1,
            tau_count=0.1,
        )
        self.assertTrue(torch.all(errors["local_count_error"] > 0))
        self.assertTrue(torch.all(reliability["supervised"] < 1))

    def test_rasterized_reliability_uses_background_value(self) -> None:
        """实例值应写入前景，背景应保持用户指定值。"""

        reliability_map = rasterize_instance_values(
            self.regions,
            torch.tensor([0.2, 0.8]),
            background_value=0.05,
        )
        background = self.regions.foreground == 0
        self.assertTrue(
            torch.allclose(
                reliability_map[background],
                torch.full_like(reliability_map[background], 0.05),
            )
        )
        self.assertGreater(float(reliability_map.max()), 0.2)
        self.assertLessEqual(float(reliability_map.max()), 0.8 + 1e-6)


if __name__ == "__main__":
    unittest.main()
