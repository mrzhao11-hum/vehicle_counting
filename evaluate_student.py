"""在固定CARPK划分上评估B1/B2使用的轻量CSRNet学生主干。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import CARPKDataset, carpk_collate_fn
from engine.common import load_checkpoint, seed_worker, set_random_seed, trainable_parameter_count
from engine.evaluator import evaluate_model
from losses import DensityMSELoss
from models import CSRNetStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估CARPK轻量CSRNet学生模型")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "train_full"), default="test"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-visualizations", type=int, default=None)
    parser.add_argument("--save-worst-visualizations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是字典")
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training = config["training"]
    evaluation = config.get("evaluation", {})

    device = torch.device(args.device or experiment.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用CUDA，但当前PyTorch检测不到GPU")
    seed = int(experiment.get("seed", 42))
    set_random_seed(seed, deterministic=False)

    manifest_key = f"{args.split}_manifest"
    if manifest_key not in data_config:
        raise KeyError(f"配置缺少data.{manifest_key}")
    dataset = CARPKDataset(
        data_root=data_config["data_root"],
        manifest=data_config[manifest_key],
        target_root=data_config["target_root"],
        output_stride=int(data_config.get("output_stride", 8)),
        horizontal_flip_probability=0.0,
    )
    workers = int(data_config.get("workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("validation_batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=carpk_collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    channel_ratio = int(model_config.get("channel_ratio", 4))
    model = CSRNetStudent(channel_ratio=channel_ratio).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_ratio = int(checkpoint.get("channel_ratio", channel_ratio))
    if checkpoint_ratio != channel_ratio:
        raise ValueError(
            f"checkpoint通道压缩率为{checkpoint_ratio}，配置要求{channel_ratio}"
        )
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("checkpoint中没有model_state或state_dict")
    model.load_state_dict(state)

    criterion = DensityMSELoss(
        reduction=training.get("loss_reduction", "batch_mean_sum")
    ).to(device)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    save_limit = (
        args.save_visualizations
        if args.save_visualizations is not None
        else int(evaluation.get("save_visualizations", 20))
    )
    save_worst_limit = (
        args.save_worst_visualizations
        if args.save_worst_visualizations is not None
        else int(evaluation.get("save_worst_visualizations", 20))
    )
    experiment_name = str(experiment.get("name", "student"))
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        amp_enabled=amp_enabled,
        output_dir=args.output_dir,
        save_visualizations=save_limit,
        save_worst_visualizations=save_worst_limit,
        description=f"CARPK {experiment_name} {args.split}",
    )

    report = {
        "experiment": experiment_name,
        "split": args.split,
        "model": "CSRNetStudent",
        "channel_ratio": channel_ratio,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameters": trainable_parameter_count(model),
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"学生逐图结果和可视化已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
