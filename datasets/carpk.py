"""CARPK图像、密度图和空间权重的PyTorch Dataset。

训练代码通过manifest读取固定的数据划分，通过``target_root``切换固定核、
自适应核或带小目标权重的标签。这样做可以保证不同消融实验只改变目标标签，
不会悄悄改变训练图像列表。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def carpk_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """CARPK专用批处理函数。

    图像和密度图尺寸固定，可以直接堆叠；每张图车辆数不同，points和boxes
    长度不一致，因此保留为列表。创建DataLoader时应显式传入本函数：

    ``DataLoader(dataset, batch_size=4, collate_fn=carpk_collate_fn)``
    """

    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "density": torch.stack([sample["density"] for sample in batch]),
        "weight": torch.stack([sample["weight"] for sample in batch]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in batch]),
        "count": torch.stack([sample["count"] for sample in batch]),
        "points": [sample["points"] for sample in batch],
        "boxes": [sample["boxes"] for sample in batch],
        "meta": [sample["meta"] for sample in batch],
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """读取新版带元数据manifest，也兼容直接保存样本列表的旧格式。"""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"manifest缺少samples列表：{path}")
    return samples


def image_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """将RGB PIL图像转换为CHW浮点Tensor，并可选ImageNet标准化。"""

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def downsample_density_preserving_count(
    density: torch.Tensor, output_height: int, output_width: int
) -> torch.Tensor:
    """下采样密度图并重新校正积分。

    普通图像resize保持亮度而不保持像素和；密度图像素和代表车辆数量，因此
    下采样后必须根据新旧积分重新缩放。这个实现也适用于尺寸不能整除stride的
    图像，比简单乘``stride²``更稳健。
    """

    old_sum = density.sum()
    resized = F.interpolate(
        density.unsqueeze(0),
        size=(output_height, output_width),
        mode="area",
    ).squeeze(0)
    new_sum = resized.sum()
    if float(old_sum) > 0.0 and float(new_sum) > 0.0:
        resized = resized * (old_sum / new_sum)
    return resized


class CARPKDataset(Dataset):
    """读取CARPK图像与预生成HDF5训练目标。

    参数：
        data_root: 官方``Images/Annotations/ImageSets``所在目录。
        manifest: ``make_carpk_splits.py``生成的JSON。
        target_root: ``generate_carpk_density.py``生成的HDF5目录。
        output_stride: 模型密度图输出步长。CSRNet使用8。
        horizontal_flip_probability: 训练集水平翻转概率；验证和测试应为0。
        normalize: 是否使用ImageNet均值方差标准化图像。
        transform: 可选的高级同步变换函数。输入并返回样本字典。
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        target_root: str | Path,
        output_stride: int = 8,
        horizontal_flip_probability: float = 0.0,
        normalize: bool = True,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.target_root = Path(target_root).expanduser().resolve()
        self.samples = load_manifest(manifest)
        self.output_stride = output_stride
        self.horizontal_flip_probability = horizontal_flip_probability
        self.normalize = normalize
        self.transform = transform

        if output_stride <= 0:
            raise ValueError("output_stride必须为正整数")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability必须位于[0, 1]")
        if not self.samples:
            raise ValueError(f"manifest中没有样本：{manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        image_path = self.data_root / record["image"]
        target_path = self.target_root / record["target"]
        if not image_path.is_file():
            raise FileNotFoundError(f"图像不存在：{image_path}")
        if not target_path.is_file():
            raise FileNotFoundError(
                f"密度标签不存在：{target_path}\n请先运行generate_carpk_density.py"
            )

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with h5py.File(target_path, "r") as handle:
            density = handle["density"][:].astype(np.float32, copy=False)
            weight = handle["weight"][:].astype(np.float32, copy=False)
            valid_mask = handle["valid_mask"][:].astype(np.float32, copy=False)
            points = handle["points"][:].astype(np.float32, copy=False)
            boxes = handle["boxes"][:].astype(np.float32, copy=False)
            count = int(handle["count"][()])

        width, height = image.size
        expected_shape = (height, width)
        for name, array in (
            ("density", density),
            ("weight", weight),
            ("valid_mask", valid_mask),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{record['id']}的{name}尺寸{array.shape}与图像{expected_shape}不一致"
                )
        if abs(float(density.sum(dtype=np.float64)) - count) > 1e-3:
            raise ValueError(f"{record['id']}的密度积分与count不一致")

        return {
            "image": image,
            "density": density,
            "weight": weight,
            "valid_mask": valid_mask,
            "points": points,
            "boxes": boxes,
            "count": float(count),
            "meta": {
                "id": record["id"],
                "sequence": record["sequence"],
                "split": record["split"],
                "image_path": str(image_path),
                "target_path": str(target_path),
            },
        }

    @staticmethod
    def _horizontal_flip(sample: dict[str, Any]) -> dict[str, Any]:
        """同步翻转图像、三个像素标签、点和框。"""

        width, _ = sample["image"].size
        sample["image"] = sample["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for name in ("density", "weight", "valid_mask"):
            sample[name] = np.flip(sample[name], axis=1).copy()

        sample["points"] = sample["points"].copy()
        if sample["points"].size:
            sample["points"][:, 0] = (width - 1) - sample["points"][:, 0]

        sample["boxes"] = sample["boxes"].copy()
        if sample["boxes"].size:
            old_x1 = sample["boxes"][:, 0].copy()
            old_x2 = sample["boxes"][:, 2].copy()
            sample["boxes"][:, 0] = (width - 1) - old_x2
            sample["boxes"][:, 2] = (width - 1) - old_x1
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._load_raw_sample(self.samples[index])
        if self.horizontal_flip_probability > 0 and random.random() < self.horizontal_flip_probability:
            sample = self._horizontal_flip(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        image_tensor = image_to_tensor(sample["image"], normalize=self.normalize)
        density = torch.from_numpy(sample["density"]).unsqueeze(0)
        weight = torch.from_numpy(sample["weight"]).unsqueeze(0)
        valid_mask = torch.from_numpy(sample["valid_mask"]).unsqueeze(0)

        input_height, input_width = density.shape[-2:]
        output_height = max(1, input_height // self.output_stride)
        output_width = max(1, input_width // self.output_stride)
        density = downsample_density_preserving_count(density, output_height, output_width)
        weight = F.interpolate(
            weight.unsqueeze(0), size=(output_height, output_width), mode="area"
        ).squeeze(0)
        valid_mask = F.interpolate(
            valid_mask.unsqueeze(0), size=(output_height, output_width), mode="nearest"
        ).squeeze(0)

        return {
            "image": image_tensor,
            "density": density,
            "weight": weight,
            "valid_mask": valid_mask,
            "count": torch.tensor(sample["count"], dtype=torch.float32),
            "points": torch.from_numpy(sample["points"]),
            "boxes": torch.from_numpy(sample["boxes"]),
            "meta": sample["meta"],
        }
