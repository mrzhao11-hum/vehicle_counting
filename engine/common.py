"""训练和评估共享的基础工具。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class AverageMeter:
    """按样本数累计平均值，避免最后一个小batch产生额外权重。"""

    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, number: int = 1) -> None:
        self.total += value * number
        self.count += number


@dataclass
class CountingAccumulator:
    """逐图片累计MAE和RMSE所需统计量。"""

    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    samples: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        errors = prediction.detach().float().cpu() - target.detach().float().cpu()
        self.absolute_error_sum += errors.abs().sum().item()
        self.squared_error_sum += errors.square().sum().item()
        self.samples += errors.numel()

    def compute(self) -> dict[str, float]:
        if self.samples == 0:
            return {"mae": 0.0, "rmse": 0.0}
        return {
            "mae": self.absolute_error_sum / self.samples,
            "rmse": (self.squared_error_sum / self.samples) ** 0.5,
        }


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """固定Python、NumPy和PyTorch随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 完全确定性有利于排查问题，但可能降低卷积速度。
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """让每个DataLoader进程拥有可复现且不同的随机种子。"""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_density_output(output: Any) -> torch.Tensor:
    """兼容直接Tensor输出和未来带中间特征的字典输出。"""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) and isinstance(output.get("density"), torch.Tensor):
        return output["density"]
    raise TypeError("模型输出必须是Tensor，或包含density Tensor的字典")


def save_checkpoint_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """先写临时文件再替换，避免训练中断留下半个checkpoint。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """兼容不同PyTorch版本的checkpoint读取。"""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint不存在：{path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch较旧版本没有weights_only参数
        return torch.load(path, map_location=device)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """恢复断点后把优化器动量等Tensor移动到当前设备。"""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def trainable_parameter_count(model: nn.Module) -> int:
    """统计部署模型中需要梯度的参数总数。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
