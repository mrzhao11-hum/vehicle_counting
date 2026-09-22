"""不读取数据集，快速检查B2 SKT蒸馏结构和梯度路径。

上传服务器后先运行：

    python check_skt.py --device cuda:0

该检查只使用小尺寸随机Tensor，不代表模型精度；它用于在正式训练前发现
特征位置、通道适配、FSP尺寸或教师冻结等实现错误。
"""

from __future__ import annotations

import argparse
from itertools import chain

import torch

from engine.common import set_random_seed, trainable_parameter_count
from losses import (
    DensityMSELoss,
    SKTFeatureAdapters,
    cosine_feature_loss,
    dense_fsp_loss,
)
from models import CSRNetStudent, CSRNetTeacher, TEACHER_FEATURE_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查B2 SKT蒸馏结构和梯度")
    parser.add_argument("--device", default="cpu", help="cpu或cuda:0")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--channel-ratio", type=int, default=4)
    return parser.parse_args()


def _unpack(output: dict[str, object]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    density = output["density"]
    features = output["features"]
    if not isinstance(density, torch.Tensor) or not isinstance(features, list):
        raise TypeError("return_features输出格式不正确")
    if not all(isinstance(feature, torch.Tensor) for feature in features):
        raise TypeError("features中存在非Tensor元素")
    return density, features


def _assert_finite_gradients(parameters, name: str) -> None:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise AssertionError(f"{name}没有收到梯度")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{name}出现非有限梯度")


def main() -> None:
    args = parse_args()
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height和width必须能被8整除")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到GPU")

    set_random_seed(42, deterministic=True)
    # 与正式B2相同，先创建学生以保持它与B1的随机初始化顺序一致。
    student = CSRNetStudent(channel_ratio=args.channel_ratio).to(device).train()
    adapters = SKTFeatureAdapters(
        student.feature_channels,
        TEACHER_FEATURE_CHANNELS,
    ).to(device).train()
    teacher = CSRNetTeacher(pretrained_frontend=False).to(device).eval()
    teacher.requires_grad_(False)

    image = torch.randn(1, 3, args.height, args.width, device=device)
    expected_density_shape = (1, 1, args.height // 8, args.width // 8)
    target = torch.rand(expected_density_shape, device=device) * 0.01

    with torch.no_grad():
        teacher_density, teacher_features = _unpack(
            teacher(image, return_features=True)
        )
    student_density, student_features = _unpack(
        student(image, return_features=True)
    )
    aligned_features = adapters(student_features)

    if tuple(student_density.shape) != expected_density_shape:
        raise AssertionError("学生密度图尺寸不正确")
    if tuple(teacher_density.shape) != expected_density_shape:
        raise AssertionError("教师密度图尺寸不正确")
    if len(aligned_features) != 6 or len(teacher_features) != 6:
        raise AssertionError("SKT必须使用六层学生/教师特征")
    if any(
        student_feature.shape != teacher_feature.shape
        for student_feature, teacher_feature in zip(
            aligned_features, teacher_features, strict=True
        )
    ):
        raise AssertionError("1x1适配后的学生特征与教师特征尺寸不一致")

    density_criterion = DensityMSELoss(reduction="batch_mean_sum")
    gt_loss = density_criterion(student_density, target)
    output_loss = density_criterion(student_density, teacher_density)
    fsp_loss = dense_fsp_loss(
        [*aligned_features, student_density],
        [*teacher_features, teacher_density],
    )
    cosine_loss = cosine_feature_loss(aligned_features, teacher_features)
    total_loss = gt_loss + output_loss + 0.5 * fsp_loss + 0.5 * cosine_loss
    if not torch.isfinite(total_loss):
        raise AssertionError("SKT总损失不是有限数值")
    total_loss.backward()

    _assert_finite_gradients(student.parameters(), "学生主干")
    _assert_finite_gradients(adapters.parameters(), "训练期适配层")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise AssertionError("冻结教师不应产生梯度")

    optimizer_parameter_count = sum(
        parameter.numel()
        for parameter in chain(student.parameters(), adapters.parameters())
        if parameter.requires_grad
    )
    print(f"输入尺寸：{tuple(image.shape)}")
    print(f"输出密度图：{tuple(student_density.shape)}")
    print(f"教师特征：{[tuple(feature.shape) for feature in teacher_features]}")
    print(f"适配学生特征：{[tuple(feature.shape) for feature in aligned_features]}")
    print(
        f"loss GT={gt_loss.item():.6f} output={output_loss.item():.6f} "
        f"FSP={fsp_loss.item():.6f} cosine={cosine_loss.item():.6f} "
        f"total={total_loss.item():.6f}"
    )
    print(f"部署学生参数：{trainable_parameter_count(student):,}")
    print(f"训练期适配层参数：{trainable_parameter_count(adapters):,}")
    print(f"优化器总参数：{optimizer_parameter_count:,}")
    print("B2 SKT结构、损失、反向传播和教师冻结检查通过。")


if __name__ == "__main__":
    main()
