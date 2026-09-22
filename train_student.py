"""训练CARPK的1/4-CSRNet学生基线（B1，不使用知识蒸馏）。

推荐先执行8张图过拟合检查：

    python train_student.py --config configs/carpk_student_fixed.yaml \
        --overfit-samples 8 --epochs 200 \
        --output-dir outputs/overfit/b1_student_fixed_8

检查通过后再进行完整开发实验：

    python train_student.py --config configs/carpk_student_fixed.yaml

B1只使用真实密度图监督。教师网络、教师输出和中间特征都不会被加载，因而
该实验可以测量轻量学生结构自身的能力，并作为后续B2蒸馏收益的对照组。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from datasets import CARPKDataset, carpk_collate_fn
from engine.common import (
    load_checkpoint,
    move_optimizer_state_to_device,
    save_checkpoint_atomic,
    seed_worker,
    set_random_seed,
    trainable_parameter_count,
)
from engine.evaluator import evaluate_model
from engine.trainer import train_one_epoch
from losses import DensityMSELoss
from models import CSRNetStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练CARPK 1/4-CSRNet学生基线")
    parser.add_argument("--config", type=Path, required=True, help="YAML配置文件")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖实验输出目录")
    parser.add_argument("--resume", type=Path, default=None, help="从checkpoint继续训练")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的总epoch数")
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="训练集前N张同时作为验证集，用于检查学生能否拟合标签",
    )
    parser.add_argument("--device", type=str, default=None, help="例如cuda:0或cpu")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是字典")
    return payload


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用CUDA，但当前PyTorch检测不到GPU")
    return device


def make_loader(
    dataset: CARPKDataset | Subset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """创建可复现的DataLoader，数据接口与B0教师实验完全一致。"""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=carpk_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def append_history(path: Path, row: dict[str, Any]) -> None:
    """逐epoch追加训练曲线；断点恢复时不会覆盖已有历史。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training = config["training"]

    output_dir = Path(args.output_dir or experiment["output_dir"])
    resume_path = args.resume
    if resume_path is None and (output_dir / "last.pth").exists():
        raise FileExistsError(
            f"{output_dir}中已有last.pth。请使用新的--output-dir，"
            "或通过--resume继续训练，避免覆盖一次正式实验。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment.get("seed", 42))
    deterministic = bool(experiment.get("deterministic", False))
    set_random_seed(seed, deterministic=deterministic)
    device = resolve_device(args.device or experiment.get("device", "cuda:0"))

    # 小样本过拟合检查关闭随机翻转，保证同一输入每轮对应完全相同的监督。
    train_flip = float(data_config.get("horizontal_flip_probability", 0.5))
    if args.overfit_samples > 0:
        train_flip = 0.0

    train_dataset = CARPKDataset(
        data_root=data_config["data_root"],
        manifest=data_config["train_manifest"],
        target_root=data_config["target_root"],
        output_stride=int(data_config.get("output_stride", 8)),
        horizontal_flip_probability=train_flip,
    )
    validation_dataset: CARPKDataset | Subset = CARPKDataset(
        data_root=data_config["data_root"],
        manifest=data_config["val_manifest"],
        target_root=data_config["target_root"],
        output_stride=int(data_config.get("output_stride", 8)),
        horizontal_flip_probability=0.0,
    )

    if args.overfit_samples > 0:
        sample_count = min(args.overfit_samples, len(train_dataset))
        indices = list(range(sample_count))
        original_train_dataset = train_dataset
        train_dataset = Subset(original_train_dataset, indices)
        validation_dataset = Subset(original_train_dataset, indices)
        print(f"启用学生过拟合检查：训练和验证均使用同一组{sample_count}张图")

    workers = int(data_config.get("workers", 4))
    batch_size = int(data_config.get("batch_size", 1))
    validation_batch_size = int(data_config.get("validation_batch_size", batch_size))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        device=device,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        workers=workers,
        device=device,
        seed=seed + 1,
    )

    channel_ratio = int(model_config.get("channel_ratio", 4))
    model = CSRNetStudent(channel_ratio=channel_ratio).to(device)
    criterion = DensityMSELoss(
        reduction=training.get("loss_reduction", "batch_mean_sum")
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 10)),
        min_lr=float(training.get("minimum_learning_rate", 1e-7)),
    )

    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_mae = float("inf")

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        checkpoint_ratio = int(checkpoint.get("channel_ratio", channel_ratio))
        if checkpoint_ratio != channel_ratio:
            raise ValueError(
                f"checkpoint通道压缩率为{checkpoint_ratio}，配置要求{channel_ratio}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint.get("best_mae", best_mae))
        print(f"已恢复学生checkpoint：{resume_path}，从epoch {start_epoch}继续")

    total_epochs = int(args.epochs or training.get("epochs", 200))
    if start_epoch > total_epochs:
        raise ValueError(
            f"checkpoint已训练到epoch {start_epoch - 1}，不小于目标epoch {total_epochs}"
        )

    resolved_config = dict(config)
    resolved_config["runtime"] = {
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
        "overfit_samples": args.overfit_samples,
        "resume": str(resume_path) if resume_path else None,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    parameter_count = trainable_parameter_count(model)
    print(f"设备：{device}")
    print(f"模型：1/{channel_ratio}-CSRNet学生（B1，无蒸馏）")
    print(f"训练样本：{len(train_dataset)}，验证样本：{len(validation_dataset)}")
    print(f"可训练参数：{parameter_count:,}")
    print(f"batch={batch_size}，AMP={amp_enabled}，输出目录：{output_dir}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
            print_frequency=int(training.get("print_frequency", 20)),
            gradient_clip_norm=training.get("gradient_clip_norm"),
        )
        validation_metrics = evaluate_model(
            model=model,
            loader=validation_loader,
            device=device,
            criterion=criterion,
            amp_enabled=amp_enabled,
            description=f"B1 validation epoch={epoch:03d}",
        )
        scheduler.step(validation_metrics["mae"])

        is_best = validation_metrics["mae"] < best_mae
        if is_best:
            best_mae = validation_metrics["mae"]

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": "CSRNetStudent",
            "channel_ratio": channel_ratio,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_mae": best_mae,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "config": resolved_config,
        }
        save_checkpoint_atomic(checkpoint_payload, output_dir / "last.pth")
        if is_best:
            save_checkpoint_atomic(checkpoint_payload, output_dir / "best_mae.pth")

        current_lr = optimizer.param_groups[0]["lr"]
        append_history(
            output_dir / "history.csv",
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": validation_metrics["loss"],
                "val_mae": validation_metrics["mae"],
                "val_rmse": validation_metrics["rmse"],
                "best_mae": best_mae,
            },
        )
        print(
            f"epoch={epoch:03d}完成，val_MAE={validation_metrics['mae']:.3f} "
            f"val_RMSE={validation_metrics['rmse']:.3f} "
            f"best_MAE={best_mae:.3f} lr={current_lr:.2e}"
        )

    summary = {
        "experiment": "B1 quarter-CSRNet without distillation",
        "epochs": total_epochs,
        "channel_ratio": channel_ratio,
        "parameters": parameter_count,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_mae": best_mae,
        "best_checkpoint": str((output_dir / "best_mae.pth").resolve()),
        "last_checkpoint": str((output_dir / "last.pth").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"B1训练完成，最佳验证MAE={best_mae:.3f}")


if __name__ == "__main__":
    main()
