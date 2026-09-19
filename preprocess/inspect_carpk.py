"""检查 CARPK 原始数据并输出统计报告和标注框可视化。

建议任何训练之前先运行本脚本。它不会修改 ``data/raw``，所有结果都会
写入 ``outputs/data_check/carpk``（或 ``--output`` 指定的目录）。

示例：

    python preprocess/inspect_carpk.py \
        --root data/raw/CARPK/CARPK_devkit/data \
        --output outputs/data_check/carpk \
        --num-visualizations 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from carpk_utils import (
    find_image,
    official_splits,
    read_annotation,
    resolve_data_root,
    sequence_name,
    validate_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查CARPK原始数据完整性并生成可视化")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/CARPK/CARPK_devkit/data"),
        help="CARPK data、CARPK_devkit或其上层目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_check/carpk"),
        help="报告与可视化输出目录",
    )
    parser.add_argument("--num-visualizations", type=int, default=20, help="保存多少张框可视化")
    parser.add_argument("--seed", type=int, default=42, help="可视化抽样随机种子")
    return parser.parse_args()


def percentile(values: list[int], ratio: float) -> int:
    """使用最近秩的简洁分位数，避免为了统计额外依赖pandas。"""

    if not values:
        raise ValueError("不能对空序列计算分位数")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def choose_visualization_ids(
    train_ids: list[str], test_ids: list[str], count: int, seed: int
) -> list[str]:
    """同时从训练和测试中抽样，避免只看到某一个停车场。"""

    if count <= 0:
        return []
    rng = random.Random(seed)
    test_count = min(len(test_ids), count // 2)
    train_count = min(len(train_ids), count - test_count)
    selected = rng.sample(train_ids, train_count) + rng.sample(test_ids, test_count)
    rng.shuffle(selected)
    return selected


def save_box_visualization(
    image_path: Path, boxes, output_path: Path, sample_id: str
) -> None:
    """在原图副本上绘制框和中心点，原始图片不会被覆盖。"""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in boxes:
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(255, 60, 40), width=2)
        center_x, center_y = box.center
        radius = 2
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(30, 255, 90),
        )

    # 黑底文字放在左上角，便于快速核对标注数量是否与肉眼观察一致。
    title = f"{sample_id} | count={len(boxes)}"
    text_box = draw.textbbox((8, 8), title, font=font)
    draw.rectangle((4, 4, text_box[2] + 8, text_box[3] + 8), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def save_histograms(counts: list[int], widths: list[int], heights: list[int], output: Path) -> None:
    """保存车辆数量和框尺寸分布，用于选择高斯核初始参数。"""

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(counts, bins=30, color="#257180", edgecolor="white")
    axes[0].set_title("Vehicles per image")
    axes[0].set_xlabel("count")
    axes[0].set_ylabel("images")

    axes[1].hist(widths, bins=40, color="#CB6040", edgecolor="white")
    axes[1].set_title("Bounding-box width")
    axes[1].set_xlabel("pixels")

    axes[2].hist(heights, bins=40, color="#5B8E7D", edgecolor="white")
    axes[2].set_title("Bounding-box height")
    axes[2].set_xlabel("pixels")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    images_dir = data_root / "Images"
    annotations_dir = data_root / "Annotations"
    train_ids, test_ids = official_splits(data_root)
    split_by_id = {sample_id: "train" for sample_id in train_ids}
    split_by_id.update({sample_id: "test" for sample_id in test_ids})

    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    annotation_paths = sorted(annotations_dir.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    annotation_stems = {path.stem for path in annotation_paths}
    split_stems = set(train_ids + test_ids)

    integrity_errors: list[str] = []
    for sample_id in sorted(image_stems - annotation_stems):
        integrity_errors.append(f"图像缺少标注：{sample_id}")
    for sample_id in sorted(annotation_stems - image_stems):
        integrity_errors.append(f"标注缺少图像：{sample_id}")
    for sample_id in sorted(split_stems - image_stems):
        integrity_errors.append(f"划分中样本缺少图像：{sample_id}")
    for sample_id in sorted(image_stems - split_stems):
        integrity_errors.append(f"图像未进入官方划分：{sample_id}")

    rows: list[dict] = []
    all_counts: list[int] = []
    all_widths: list[int] = []
    all_heights: list[int] = []
    class_counter: Counter[int] = Counter()
    resolution_counter: Counter[str] = Counter()

    for sample_id in tqdm(sorted(split_stems), desc="检查CARPK", unit="image"):
        image_path = find_image(images_dir, sample_id)
        annotation_path = annotations_dir / f"{sample_id}.txt"
        boxes = read_annotation(annotation_path)

        # verify()只校验图像文件结构，不会把整张图长期保留在内存中。
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        resolution_counter[f"{width}x{height}"] += 1

        box_errors = validate_boxes(boxes, width, height)
        integrity_errors.extend(f"{sample_id}: {message}" for message in box_errors)
        widths = [box.width for box in boxes]
        heights = [box.height for box in boxes]
        all_widths.extend(widths)
        all_heights.extend(heights)
        all_counts.append(len(boxes))
        class_counter.update(box.class_id for box in boxes)

        rows.append(
            {
                "id": sample_id,
                "official_split": split_by_id[sample_id],
                "sequence": sequence_name(sample_id),
                "width": width,
                "height": height,
                "count": len(boxes),
                "mean_box_width": round(mean(widths), 4) if widths else 0.0,
                "mean_box_height": round(mean(heights), 4) if heights else 0.0,
            }
        )

    with (output / "per_image_stats.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    sequence_counts = Counter(sequence_name(sample_id) for sample_id in split_stems)
    report = {
        "data_root": str(data_root),
        "images": len(image_paths),
        "annotations": len(annotation_paths),
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "total_boxes": sum(all_counts),
        "classes": dict(sorted(class_counter.items())),
        "resolutions": dict(sorted(resolution_counter.items())),
        "sequences": dict(sorted(sequence_counts.items())),
        "count_statistics": {
            "min": min(all_counts),
            "median": median(all_counts),
            "mean": round(mean(all_counts), 4),
            "max": max(all_counts),
        },
        "box_width_statistics": {
            "min": min(all_widths),
            "p10": percentile(all_widths, 0.10),
            "median": median(all_widths),
            "p90": percentile(all_widths, 0.90),
            "max": max(all_widths),
        },
        "box_height_statistics": {
            "min": min(all_heights),
            "p10": percentile(all_heights, 0.10),
            "median": median(all_heights),
            "p90": percentile(all_heights, 0.90),
            "max": max(all_heights),
        },
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors[:100],
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_histograms(all_counts, all_widths, all_heights, output / "distributions.png")

    selected_ids = choose_visualization_ids(
        train_ids, test_ids, args.num_visualizations, args.seed
    )
    for sample_id in tqdm(selected_ids, desc="绘制标注框", unit="image"):
        boxes = read_annotation(annotations_dir / f"{sample_id}.txt")
        save_box_visualization(
            find_image(images_dir, sample_id),
            boxes,
            output / "bbox_visualizations" / f"{sample_id}.jpg",
            sample_id,
        )

    print(f"检查完成：{len(image_paths)}张图像，{sum(all_counts)}个车辆框")
    print(f"官方划分：train={len(train_ids)}，test={len(test_ids)}")
    print(f"完整性错误：{len(integrity_errors)}")
    print(f"报告目录：{output}")
    if integrity_errors:
        raise SystemExit("数据完整性检查未通过，请先查看 dataset_report.json")


if __name__ == "__main__":
    main()

