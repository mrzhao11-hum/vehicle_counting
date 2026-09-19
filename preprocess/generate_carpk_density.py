"""把CARPK边界框转换为密度图HDF5标签。

支持两种高斯核：

1. ``fixed``：所有车辆使用相同sigma，作为标准基线。
2. ``adaptive``：sigma由bbox面积决定，用于尺度感知实验。

每辆车的高斯核都会在实际落入图像的区域重新归一化为1，因此即使车辆位于
图像边缘，整张密度图的积分仍应等于车辆框数量。脚本逐文件原子写入，默认
跳过配置一致的已有HDF5，网络中断后重新运行即可续跑。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import h5py
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carpk_utils import (
    BoundingBox,
    find_image,
    official_splits,
    read_annotation,
    resolve_data_root,
)


@dataclass(frozen=True)
class DensityConfig:
    """会写入HDF5属性的标签生成配置。"""

    mode: str
    fixed_sigma: float
    adaptive_alpha: float
    sigma_min: float
    sigma_max: float
    truncate: float
    with_scale_weight: bool
    reference_area: float
    max_scale_weight: float

    def signature(self) -> str:
        """稳定序列化，用来阻止不同配置误写到同一输出目录。"""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成CARPK固定核或自适应核密度图")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/CARPK/CARPK_devkit/data"),
        help="CARPK data、CARPK_devkit或其上层目录",
    )
    parser.add_argument("--output", type=Path, default=None, help="HDF5输出目录")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--fixed-sigma", type=float, default=8.0, help="固定核sigma，单位为原图像素")
    parser.add_argument(
        "--adaptive-alpha",
        type=float,
        default=0.15,
        help="自适应sigma系数：sigma=alpha*sqrt(bbox面积)",
    )
    parser.add_argument("--sigma-min", type=float, default=2.0)
    parser.add_argument("--sigma-max", type=float, default=15.0)
    parser.add_argument("--truncate", type=float, default=3.0, help="核半径为truncate*sigma")
    parser.add_argument(
        "--with-scale-weight",
        action="store_true",
        help="额外生成小目标空间权重；基线实验不要开启",
    )
    parser.add_argument("--max-scale-weight", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=0, help="并行进程数，0表示单进程")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有HDF5")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N张，用于快速测试；0为全部")
    parser.add_argument("--num-visualizations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def object_sigma(box: BoundingBox, config: DensityConfig) -> float:
    """根据配置计算单个车辆核宽度。"""

    if config.mode == "fixed":
        return config.fixed_sigma
    scale = config.adaptive_alpha * math.sqrt(float(box.area))
    return float(np.clip(scale, config.sigma_min, config.sigma_max))


def gaussian_region(
    center_x: float,
    center_y: float,
    sigma: float,
    width: int,
    height: int,
    truncate: float,
) -> tuple[slice, slice, np.ndarray]:
    """返回图像范围内的二维高斯局部区域。

    只计算中心附近的有限窗口，而不是为每辆车创建一整张1280x720数组，
    这样既节省内存，也显著减少预处理时间。
    """

    radius = max(1, int(math.ceil(truncate * sigma)))
    x_min = max(0, int(math.floor(center_x - radius)))
    x_max = min(width - 1, int(math.ceil(center_x + radius)))
    y_min = max(0, int(math.floor(center_y - radius)))
    y_max = min(height - 1, int(math.ceil(center_y + radius)))

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    dx = xs[None, :] - np.float32(center_x)
    dy = ys[:, None] - np.float32(center_y)
    gaussian = np.exp(-(dx * dx + dy * dy) / np.float32(2.0 * sigma * sigma))
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1), gaussian


def build_targets(
    boxes: list[BoundingBox], width: int, height: int, config: DensityConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成密度图、尺度权重图、中心点和每个目标的sigma。"""

    density = np.zeros((height, width), dtype=np.float32)
    weight = np.ones((height, width), dtype=np.float32)
    points = np.asarray([box.center for box in boxes], dtype=np.float32).reshape(-1, 2)
    sigmas = np.empty((len(boxes),), dtype=np.float32)

    for index, box in enumerate(boxes):
        sigma = object_sigma(box, config)
        sigmas[index] = sigma
        center_x, center_y = box.center
        y_slice, x_slice, gaussian = gaussian_region(
            center_x, center_y, sigma, width, height, config.truncate
        )

        # 边缘截断后重新归一化，保证每辆车对总计数的贡献严格接近1。
        gaussian_sum = float(gaussian.sum(dtype=np.float64))
        if gaussian_sum <= 0:
            raise RuntimeError(f"车辆框产生了空高斯核：{box.as_list()}")
        density[y_slice, x_slice] += gaussian / np.float32(gaussian_sum)

        if config.with_scale_weight:
            # 小于训练集参考面积的框权重大于1；大框和背景保持1。
            object_weight = math.sqrt(config.reference_area / max(float(box.area), 1.0))
            object_weight = float(np.clip(object_weight, 1.0, config.max_scale_weight))
            if object_weight > 1.0:
                peak_normalized = gaussian / max(float(gaussian.max()), 1e-12)
                local_weight = 1.0 + (object_weight - 1.0) * peak_normalized
                weight[y_slice, x_slice] = np.maximum(weight[y_slice, x_slice], local_weight)

    return density, weight, points, sigmas


def validate_existing(path: Path, config: DensityConfig) -> dict:
    """检查续跑时已有文件是否由同一配置生成。"""

    with h5py.File(path, "r") as handle:
        existing_signature = str(handle.attrs.get("configuration", ""))
        if existing_signature != config.signature():
            raise RuntimeError(
                f"已有文件配置与当前参数不同：{path}\n"
                "请改用新的输出目录，或明确添加 --overwrite。"
            )
        count = int(handle["count"][()])
        # 新版文件直接读取属性即可快速续跑；兼容早期测试文件时再读取数组。
        if "density_sum" in handle.attrs:
            density_sum = float(handle.attrs["density_sum"])
        else:
            density_sum = float(handle["density"][:].sum(dtype=np.float64))
    return {
        "id": path.stem,
        "count": count,
        "density_sum": density_sum,
        "absolute_error": abs(density_sum - count),
        "status": "skipped",
    }


def generate_one(task: tuple[str, str, str, dict, bool]) -> dict:
    """生成一张图的HDF5；函数位于模块顶层以支持Windows多进程。"""

    sample_id, data_root_text, output_text, config_dict, overwrite = task
    data_root = Path(data_root_text)
    output = Path(output_text)
    config = DensityConfig(**config_dict)
    target_path = output / f"{sample_id}.h5"
    if target_path.exists() and not overwrite:
        return validate_existing(target_path, config)

    image_path = find_image(data_root / "Images", sample_id)
    annotation_path = data_root / "Annotations" / f"{sample_id}.txt"
    boxes = read_annotation(annotation_path)
    with Image.open(image_path) as image:
        width, height = image.size

    density, weight, points, sigmas = build_targets(boxes, width, height, config)
    density_sum = float(density.sum(dtype=np.float64))
    boxes_array = np.asarray([box.as_list() for box in boxes], dtype=np.int32).reshape(-1, 5)
    valid_mask = np.ones((height, width), dtype=np.uint8)

    # 先写临时文件再原子替换。若进程或网络中断，最终文件不会处于半写状态。
    temporary_path = output / f".{sample_id}.{os.getpid()}.tmp.h5"
    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.create_dataset(
                "density", data=density, compression="gzip", compression_opts=4, shuffle=True
            )
            handle.create_dataset(
                "weight", data=weight, compression="gzip", compression_opts=4, shuffle=True
            )
            handle.create_dataset(
                "valid_mask", data=valid_mask, compression="gzip", compression_opts=1
            )
            handle.create_dataset("points", data=points)
            handle.create_dataset("boxes", data=boxes_array)
            handle.create_dataset("sigmas", data=sigmas)
            handle.create_dataset("count", data=np.int32(len(boxes)))
            handle.attrs["sample_id"] = sample_id
            handle.attrs["image_width"] = width
            handle.attrs["image_height"] = height
            handle.attrs["density_sum"] = density_sum
            handle.attrs["configuration"] = config.signature()
            handle.attrs["source_annotation"] = annotation_path.name
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "id": sample_id,
        "count": len(boxes),
        "density_sum": density_sum,
        "absolute_error": abs(density_sum - len(boxes)),
        "status": "generated",
    }


def save_visualization(
    data_root: Path, target_path: Path, output_path: Path, sample_id: str
) -> None:
    """保存原图、密度图叠加和权重图，便于人工检查标签形状。"""

    image_path = find_image(data_root / "Images", sample_id)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    with h5py.File(target_path, "r") as handle:
        density = handle["density"][:]
        weight = handle["weight"][:]
        count = int(handle["count"][()])

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Image | GT count={count}")
    axes[1].imshow(image)
    axes[1].imshow(density, cmap="jet", alpha=0.55)
    axes[1].set_title(f"Density overlay | sum={density.sum():.4f}")
    axes[2].imshow(weight, cmap="viridis", vmin=1.0)
    axes[2].set_title(f"Scale weight | max={weight.max():.3f}")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(sample_id)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.fixed_sigma <= 0 or args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        raise ValueError("sigma参数必须为正数，并满足 sigma_max >= sigma_min")
    if args.workers < 0:
        raise ValueError("--workers不能为负数")

    data_root = resolve_data_root(args.root)
    official_train, official_test = official_splits(data_root)
    sample_ids = official_train + official_test
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    # 小目标参考面积只能由官方训练集统计，绝不能读取测试集分布。
    training_areas = [
        box.area
        for sample_id in official_train
        for box in read_annotation(data_root / "Annotations" / f"{sample_id}.txt")
    ]
    reference_area = float(median(training_areas))
    config = DensityConfig(
        mode=args.mode,
        fixed_sigma=args.fixed_sigma,
        adaptive_alpha=args.adaptive_alpha,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        truncate=args.truncate,
        with_scale_weight=args.with_scale_weight,
        reference_area=reference_area,
        max_scale_weight=args.max_scale_weight,
    )

    if args.output is None:
        suffix = "adaptive_sigma" if args.mode == "adaptive" else "fixed_sigma"
        if args.with_scale_weight:
            suffix += "_weighted"
        output = Path("data/processed/CARPK") / suffix
    else:
        output = args.output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    tasks = [
        (sample_id, str(data_root), str(output), asdict(config), args.overwrite)
        for sample_id in sample_ids
    ]
    if args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(generate_one, tasks),
                    total=len(tasks),
                    desc="生成密度图",
                    unit="image",
                )
            )
    else:
        results = [
            generate_one(task)
            for task in tqdm(tasks, desc="生成密度图", unit="image")
        ]

    max_error = max((row["absolute_error"] for row in results), default=0.0)
    failed = [row for row in results if row["absolute_error"] > args.tolerance]
    with (output / "generation_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    report = {
        "data_root": str(data_root),
        "output": str(output),
        "configuration": asdict(config),
        "processed_images": len(results),
        "generated_images": sum(row["status"] == "generated" for row in results),
        "skipped_images": sum(row["status"] == "skipped" for row in results),
        "maximum_count_error": max_error,
        "tolerance": args.tolerance,
        "failed_images": failed,
    }
    (output / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    visualization_count = min(args.num_visualizations, len(sample_ids))
    if visualization_count > 0:
        # 等间隔选择而不是只看文件名前若干张，以覆盖不同拍摄序列。
        indices = np.linspace(0, len(sample_ids) - 1, visualization_count, dtype=int)
        for index in tqdm(indices, desc="保存密度图可视化", unit="image"):
            sample_id = sample_ids[int(index)]
            save_visualization(
                data_root,
                output / f"{sample_id}.h5",
                output / "visualizations" / f"{sample_id}.jpg",
                sample_id,
            )

    print(f"标签输出目录：{output}")
    print(f"处理图像：{len(results)}，最大积分误差：{max_error:.8f}")
    # 使用普通中文而不是Unicode上标，兼容Windows默认GBK终端。
    print(f"训练集bbox参考面积：{reference_area:.2f}平方像素")
    if failed:
        raise SystemExit(
            f"有{len(failed)}张图的密度积分误差超过{args.tolerance}，请查看generation_report.json"
        )


if __name__ == "__main__":
    main()
