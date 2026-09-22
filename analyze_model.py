"""统计教师或学生的参数量、卷积计算量、权重体积与推理延迟。

示例：

    python analyze_model.py --model teacher --device cuda:0 \
        --checkpoint outputs/carpk/b0_teacher_fixed_sigma8/best_mae.pth

    python analyze_model.py --model student --device cuda:0 \
        --checkpoint outputs/carpk/b1_student_fixed_sigma8/best_mae.pth \
        --output outputs/carpk/b1_student_fixed_sigma8/complexity.json

FLOPs按卷积乘法和加法各一次（FLOPs=2*MACs）统计，不计ReLU、池化和数据
搬运。因此论文表格中必须注明本脚本的统计口径，并对所有模型使用同一脚本。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from engine.common import load_checkpoint
from models import CSRNetStudent, CSRNetTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计CSRNet模型复杂度和延迟")
    parser.add_argument("--model", choices=("teacher", "student"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--channel-ratio", type=int, default=4)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true", help="关闭CUDA FP16推理")
    parser.add_argument("--output", type=Path, default=None, help="可选JSON输出路径")
    return parser.parse_args()


def build_model(name: str, channel_ratio: int) -> nn.Module:
    if name == "teacher":
        return CSRNetTeacher(pretrained_frontend=False)
    return CSRNetStudent(channel_ratio=channel_ratio)


def load_model_state(model: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = load_checkpoint(path, device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("checkpoint中没有model_state或state_dict")
    model.load_state_dict(state)


def parameter_statistics(model: nn.Module) -> dict[str, int | float]:
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    bytes_total = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "parameter_size_mb": bytes_total / (1024**2),
    }


def convolution_macs(
    model: nn.Module, image: torch.Tensor
) -> tuple[int, tuple[int, ...]]:
    """用前向hook按实际输出尺寸累计Conv2d的MACs。"""

    macs = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if not isinstance(module, nn.Conv2d) or not isinstance(output, torch.Tensor):
            return
        output_elements_per_sample = output[0].numel()
        kernel_height, kernel_width = module.kernel_size
        operations_per_output = (
            module.in_channels // module.groups * kernel_height * kernel_width
        )
        macs += int(output_elements_per_sample * operations_per_output)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(image)
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(output, torch.Tensor):
        raise TypeError("复杂度统计要求模型普通前向返回Tensor")
    return macs, tuple(output.shape)


def benchmark_latency(
    model: nn.Module,
    image: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    amp_enabled: bool,
) -> dict[str, float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup不能为负，repeats必须为正")
    device = image.device

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                model(image)
        synchronize()

        durations_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                model(image)
            synchronize()
            durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations_ms.sort()
    mean_ms = sum(durations_ms) / len(durations_ms)
    median_ms = durations_ms[len(durations_ms) // 2]
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "fps_from_mean_latency": 1000.0 / mean_ms,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到GPU")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("输入宽高必须为正")

    model = build_model(args.model, args.channel_ratio).to(device).eval()
    if args.checkpoint is not None:
        load_model_state(model, args.checkpoint, device)

    image = torch.randn(1, 3, args.height, args.width, device=device)
    macs, output_shape = convolution_macs(model, image)
    amp_enabled = device.type == "cuda" and not args.no_amp
    report: dict[str, Any] = {
        "model": args.model,
        "channel_ratio": args.channel_ratio if args.model == "student" else 1,
        "input_shape": list(image.shape),
        "output_shape": list(output_shape),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        **parameter_statistics(model),
        "conv_macs": macs,
        "conv_gmacs": macs / 1e9,
        "conv_flops": 2 * macs,
        "conv_gflops": 2 * macs / 1e9,
        "latency_amp": amp_enabled,
        **benchmark_latency(
            model,
            image,
            warmup=args.warmup,
            repeats=args.repeats,
            amp_enabled=amp_enabled,
        ),
    }
    if args.checkpoint is not None:
        report["checkpoint_size_mb"] = args.checkpoint.stat().st_size / (1024**2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"复杂度报告已保存到：{args.output}")


if __name__ == "__main__":
    main()
