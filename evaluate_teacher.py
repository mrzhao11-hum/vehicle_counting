"""在固定CARPK划分上评估训练好的CSRNet教师。"""

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
from models import CSRNetTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估CARPK CSRNet教师模型")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "train_full"), default="test"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-visualizations", type=int, default=None)
    parser.add_argument(
        "--save-worst-visualizations",
        type=int,
        default=None,
        help="额外保存绝对计数误差最大的N张结果图",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是字典")
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    data_config = config["data"]
    training = config["training"]
    evaluation = config.get("evaluation", {})

    device = torch.device(args.device or experiment.get("device", "cuda:0"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用CUDA，但当前PyTorch检测不到GPU")
    set_random_seed(int(experiment.get("seed", 42)), deterministic=False)

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
    generator = torch.Generator().manual_seed(int(experiment.get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=int(data_config.get("validation_batch_size", 1)),
        shuffle=False,
        num_workers=int(data_config.get("workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data_config.get("workers", 4)) > 0,
        collate_fn=carpk_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = CSRNetTeacher(pretrained_frontend=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
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
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        amp_enabled=amp_enabled,
        output_dir=args.output_dir,
        save_visualizations=save_limit,
        save_worst_visualizations=save_worst_limit,
        description=f"CARPK {args.split}",
    )

    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameters": trainable_parameter_count(model),
        "metrics": metrics,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"逐图结果和可视化已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
