"""训练CARPK CSRNet教师模型。

常规训练：
    python train_teacher.py --config configs/carpk_teacher_fixed.yaml

8张图过拟合检查：
    python train_teacher.py --config configs/carpk_teacher_fixed.yaml \
        --overfit-samples 8 --epochs 200 \
        --output-dir outputs/overfit/carpk_teacher_fixed_8

使用官方989张训练图进行固定轮数正式训练：
    python train_teacher.py --config configs/carpk_teacher_fixed_full.yaml

训练过程只在验证集上选择最佳模型，不会自动运行测试集。最终测试请使用
``evaluate_teacher.py``，避免根据测试结果反复调整模型造成数据泄漏。
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
from models import CSRNetTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练CARPK CSRNet教师模型")
    parser.add_argument("--config", type=Path, required=True, help="YAML配置文件")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖配置中的输出目录")
    parser.add_argument("--resume", type=Path, default=None, help="从last/best checkpoint继续训练")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的总epoch数")
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="只取训练集前N张并同时作为验证集；用于排查训练链路",
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
    """把每个epoch的核心结果写入CSV，方便画训练曲线。"""

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
    validation_enabled = bool(training.get("validation_enabled", True))
    if args.overfit_samples > 0 and not validation_enabled:
        raise ValueError("过拟合检查需要启用验证，因为训练和验证使用同一组样本")

    output_dir = Path(args.output_dir or experiment["output_dir"])
    resume_path = args.resume
    existing_checkpoints = [
        path for path in (output_dir / "last.pth", output_dir / "final.pth") if path.exists()
    ]
    if resume_path is None and existing_checkpoints:
        raise FileExistsError(
            f"{output_dir}中已有checkpoint。为避免覆盖实验，请使用新的"
            "--output-dir，或通过--resume继续。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment.get("seed", 42))
    deterministic = bool(experiment.get("deterministic", False))
    set_random_seed(seed, deterministic=deterministic)
    device_name = args.device or experiment.get("device", "cuda:0")
    device = resolve_device(device_name)

    # 过拟合检查必须关闭随机翻转，否则同一张图每次看到的监督并不完全相同。
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
    validation_dataset: CARPKDataset | Subset | None = None
    if validation_enabled:
        validation_dataset = CARPKDataset(
            data_root=data_config["data_root"],
            manifest=data_config["val_manifest"],
            target_root=data_config["target_root"],
            output_stride=int(data_config.get("output_stride", 8)),
            horizontal_flip_probability=0.0,
        )

    if args.overfit_samples > 0:
        sample_count = min(args.overfit_samples, len(train_dataset))
        indices = list(range(sample_count))
        train_dataset = Subset(train_dataset, indices)
        validation_dataset = Subset(train_dataset.dataset, indices)
        print(f"启用过拟合检查：训练和验证均使用同一组{sample_count}张图")

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
    validation_loader: DataLoader | None = None
    if validation_dataset is not None:
        validation_loader = make_loader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            workers=workers,
            device=device,
            seed=seed + 1,
        )

    # 从checkpoint恢复时不需要下载ImageNet权重，随后会完整覆盖模型参数。
    pretrained_frontend = bool(model_config.get("pretrained_frontend", True))
    model = CSRNetTeacher(
        pretrained_frontend=pretrained_frontend and resume_path is None
    ).to(device)
    criterion = DensityMSELoss(
        reduction=training.get("loss_reduction", "batch_mean_sum")
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-5)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
    )
    # 默认保持旧配置的ReduceLROnPlateau行为。新的对照实验可以在YAML中
    # 设置lr_scheduler: none，显式关闭调度器并在整个训练过程中固定学习率。
    scheduler_name = str(training.get("lr_scheduler", "plateau")).strip().lower()
    if scheduler_name not in {"plateau", "none"}:
        raise ValueError(
            "training.lr_scheduler仅支持'plateau'或'none'，"
            f"当前值为{scheduler_name!r}"
        )

    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if validation_enabled and scheduler_name == "plateau":
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
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if validation_enabled and checkpoint.get("best_mae") is not None:
            best_mae = float(checkpoint["best_mae"])
        print(f"已恢复checkpoint：{resume_path}，从epoch {start_epoch}继续")

    total_epochs = int(args.epochs or training.get("epochs", 200))
    resolved_config = dict(config)
    resolved_config["runtime"] = {
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
        "overfit_samples": args.overfit_samples,
        "validation_enabled": validation_enabled,
        "resume": str(resume_path) if resume_path else None,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    parameter_count = trainable_parameter_count(model)
    print(f"设备：{device}")
    if validation_dataset is None:
        print(f"训练样本：{len(train_dataset)}，正式固定轮数训练（不使用验证集选模）")
    else:
        print(f"训练样本：{len(train_dataset)}，验证样本：{len(validation_dataset)}")
    print(f"可训练参数：{parameter_count:,}")
    print(
        f"AMP：{amp_enabled}，学习率调度器：{scheduler_name}，"
        f"输出目录：{output_dir}"
    )

    if start_epoch > total_epochs:
        raise ValueError(
            f"checkpoint已训练到epoch {start_epoch - 1}，不小于目标epoch {total_epochs}"
        )

    final_checkpoint_payload: dict[str, Any] | None = None
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
        validation_metrics: dict[str, float] | None = None
        is_best = False
        if validation_loader is not None:
            validation_metrics = evaluate_model(
                model=model,
                loader=validation_loader,
                device=device,
                criterion=criterion,
                amp_enabled=amp_enabled,
                description=f"validation epoch={epoch:03d}",
            )
            if scheduler is not None:
                scheduler.step(validation_metrics["mae"])
            is_best = validation_metrics["mae"] < best_mae
            if is_best:
                best_mae = validation_metrics["mae"]

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": "CSRNetTeacher",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "best_mae": best_mae if validation_enabled else None,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "config": resolved_config,
        }
        final_checkpoint_payload = checkpoint_payload
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
                "val_loss": validation_metrics.get("loss", "") if validation_metrics else "",
                "val_mae": validation_metrics["mae"] if validation_metrics else "",
                "val_rmse": validation_metrics["rmse"] if validation_metrics else "",
                "best_mae": best_mae if validation_metrics else "",
            },
        )
        if validation_metrics is None:
            print(
                f"epoch={epoch:03d}完成，train_MAE={train_metrics['mae']:.3f} "
                f"train_RMSE={train_metrics['rmse']:.3f} lr={current_lr:.2e}"
            )
        else:
            print(
                f"epoch={epoch:03d}完成，val_MAE={validation_metrics['mae']:.3f} "
                f"val_RMSE={validation_metrics['rmse']:.3f} "
                f"best_MAE={best_mae:.3f} lr={current_lr:.2e}"
            )

    if final_checkpoint_payload is None:
        raise RuntimeError("训练循环没有产生checkpoint")

    summary: dict[str, Any] = {
        "epochs": total_epochs,
        "training_samples": len(train_dataset),
        "validation_enabled": validation_enabled,
        "last_checkpoint": str((output_dir / "last.pth").resolve()),
    }
    if validation_enabled:
        summary.update(
            {
                "best_mae": best_mae,
                "best_checkpoint": str((output_dir / "best_mae.pth").resolve()),
            }
        )
    else:
        final_path = output_dir / "final.pth"
        save_checkpoint_atomic(final_checkpoint_payload, final_path)
        summary["final_checkpoint"] = str(final_path.resolve())
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation_enabled:
        print(f"训练完成，最佳验证MAE={best_mae:.3f}")
    else:
        print(f"正式固定轮数训练完成，最终模型：{output_dir / 'final.pth'}")


if __name__ == "__main__":
    main()
