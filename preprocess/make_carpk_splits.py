"""根据CARPK官方划分生成固定、可复现实验清单。

CARPK官方只提供 train/test。训练期间不能使用test选择checkpoint，因此本脚本
从官方train中按完整拍摄序列留出validation。默认留出20161029_NTU序列，
这样相邻视频帧不会同时出现在训练和验证中。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from carpk_utils import (
    find_image,
    official_splits,
    read_annotation,
    resolve_data_root,
    sequence_name,
)


DEFAULT_VALIDATION_SEQUENCES = ("20161029_NTU",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成CARPK固定train/val/test清单")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/CARPK/CARPK_devkit/data"),
        help="CARPK data、CARPK_devkit或其上层目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/CARPK"),
        help="manifest输出目录",
    )
    parser.add_argument(
        "--val-sequence",
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEQUENCES),
        help="从官方train中完整留作validation的拍摄序列，可指定一个或多个",
    )
    return parser.parse_args()


def build_record(data_root: Path, sample_id: str, split: str, official_split: str) -> dict:
    """构建一条与机器绝对路径无关的样本记录。"""

    image_path = find_image(data_root / "Images", sample_id)
    annotation_path = data_root / "Annotations" / f"{sample_id}.txt"
    boxes = read_annotation(annotation_path)
    with Image.open(image_path) as image:
        width, height = image.size

    return {
        "id": sample_id,
        "dataset": "CARPK",
        "split": split,
        "official_split": official_split,
        "sequence": sequence_name(sample_id),
        "image": f"Images/{image_path.name}",
        "annotation": f"Annotations/{annotation_path.name}",
        # target只保存相对文件名；固定核或自适应核目录由训练配置决定。
        "target": f"{sample_id}.h5",
        "width": width,
        "height": height,
        "count": len(boxes),
    }


def write_manifest(output: Path, name: str, records: list[dict], data_root: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": "CARPK",
        "split": name,
        "data_root_hint": str(data_root),
        "num_samples": len(records),
        "num_vehicles": sum(record["count"] for record in records),
        "samples": records,
    }
    (output / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    official_train, official_test = official_splits(data_root)
    available_sequences = sorted({sequence_name(sample_id) for sample_id in official_train})
    requested_validation = set(args.val_sequence)
    unknown_sequences = requested_validation.difference(available_sequences)
    if unknown_sequences:
        raise ValueError(
            "以下验证序列不属于CARPK官方训练集："
            f"{sorted(unknown_sequences)}；可选值为：{available_sequences}"
        )

    train_ids = [
        sample_id
        for sample_id in official_train
        if sequence_name(sample_id) not in requested_validation
    ]
    val_ids = [
        sample_id
        for sample_id in official_train
        if sequence_name(sample_id) in requested_validation
    ]
    if not train_ids or not val_ids:
        raise ValueError("训练集或验证集为空，请调整 --val-sequence")

    definitions = {
        "train": (train_ids, "train"),
        "val": (val_ids, "train"),
        "test": (official_test, "test"),
        # train_full用于超参数固定后的最终重训练，不能用于开发阶段选checkpoint。
        "train_full": (official_train, "train"),
    }
    all_manifests: dict[str, list[dict]] = {}
    for split_name, (sample_ids, official_split) in definitions.items():
        records = [
            build_record(data_root, sample_id, split_name, official_split)
            for sample_id in sample_ids
        ]
        all_manifests[split_name] = records
        write_manifest(output, split_name, records, data_root)

    summary = {
        "data_root": str(data_root),
        "validation_sequences": sorted(requested_validation),
        "available_official_train_sequences": available_sequences,
        "splits": {
            name: {
                "images": len(records),
                "vehicles": sum(record["count"] for record in records),
                "sequences": dict(Counter(record["sequence"] for record in records)),
            }
            for name, records in all_manifests.items()
        },
        "notes": [
            "train和val来自官方train，test保持官方459张不变。",
            "validation按完整拍摄序列留出，降低相邻帧泄漏风险。",
            "开发时使用train/val；确定超参数后可用train_full按固定epoch重训练。",
        ],
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"manifest已生成：{output}")
    for name in ("train", "val", "test", "train_full"):
        records = all_manifests[name]
        print(
            f"{name:10s} images={len(records):4d}, "
            f"vehicles={sum(record['count'] for record in records):6d}"
        )


if __name__ == "__main__":
    main()

