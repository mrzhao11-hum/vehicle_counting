"""训练CARPK B3：实例可靠性引导的SKT轻量学生。

本入口在B2四项SKT损失的基础上，只改造教师输出蒸馏项：

    L = L_GT + L_reliable_output + 0.5 * L_FSP + 0.5 * L_cos

并支持由配置增加``lambda_count * L_count``，用于修正密度图积分的系统性
少计。旧B3配置中的计数权重默认为0，不改变已经完成的可靠性实验。

教师始终冻结。每个车辆框先在1/8密度图上形成椭圆软实例区域，再根据教师
局部GT误差和翻转一致性生成空间可靠性图。GT监督、Dense-FSP、余弦特征
蒸馏、学生结构和优化参数均与B2一致，保证B2/B3对比公平。

小样本链路检查：

    python train_reliable_skt.py \
        --config configs/carpk_student_skt_reliable_fixed.yaml \
        --overfit-samples 8 --epochs 5 \
        --output-dir outputs/overfit/b3_student_reliable_skt_fixed_8

正式训练：

    python train_reliable_skt.py \
        --config configs/carpk_student_skt_reliable_fixed.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from itertools import chain
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
from engine.reliable_skt_trainer import train_reliable_skt_one_epoch
from losses import (
    DensityMSELoss,
    RelativeCountSmoothL1Loss,
    ReliabilityDistillationConfig,
    SKTFeatureAdapters,
)
from losses.object_reliability import InstanceMaskConfig
from models import CSRNetStudent, CSRNetTeacher, TEACHER_FEATURE_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练CARPK B3可靠性SKT蒸馏学生")
    parser.add_argument("--config", type=Path, required=True, help="B3 YAML配置文件")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖实验输出目录")
    parser.add_argument("--resume", type=Path, default=None, help="从B3 checkpoint继续训练")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="覆盖配置中的教师checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的总epoch数")
    parser.add_argument(
        "--reliability-strength",
        type=float,
        default=None,
        help="覆盖可靠性混合强度lambda；0为B2，1为B3-v1",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="训练集前N张同时作为验证集，用于检查蒸馏链路",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_teacher(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CSRNetTeacher, dict[str, Any]]:
    """加载并冻结教师；不下载VGG权重，因为checkpoint会完整覆盖参数。"""

    teacher = CSRNetTeacher(pretrained_frontend=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("教师checkpoint中没有model_state或state_dict")
    teacher.load_state_dict(state)
    teacher.requires_grad_(False)
    teacher.eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("教师冻结失败，仍有参数需要梯度")
    return teacher, checkpoint


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    teacher_config = config["teacher"]
    distillation = config["distillation"]
    reliability = config["reliability"]
    count_loss_config = config.get("count_loss", {})
    training = config["training"]

    output_dir = Path(args.output_dir or experiment["output_dir"])
    resume_path = args.resume
    if resume_path is None and (output_dir / "last.pth").exists():
        raise FileExistsError(
            f"{output_dir}中已有last.pth。请使用新的--output-dir，"
            "或通过--resume继续训练，避免覆盖正式实验。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment.get("seed", 42))
    deterministic = bool(experiment.get("deterministic", False))
    set_random_seed(seed, deterministic=deterministic)
    device = resolve_device(args.device or experiment.get("device", "cuda:0"))

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
        print(f"启用B3蒸馏链路检查：训练和验证使用同一组{sample_count}张图")

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

    # 必须先创建学生，确保seed=42时核心网络与B1拥有相同的随机初始化。
    student = CSRNetStudent(channel_ratio=channel_ratio).to(device)
    adapters = SKTFeatureAdapters(
        student.feature_channels,
        TEACHER_FEATURE_CHANNELS,
    ).to(device)

    teacher_checkpoint_path = Path(
        args.teacher_checkpoint or teacher_config["checkpoint"]
    )
    teacher, teacher_checkpoint = _load_teacher(teacher_checkpoint_path, device)
    teacher_epoch = teacher_checkpoint.get("epoch")

    density_criterion = DensityMSELoss(
        reduction=training.get("loss_reduction", "batch_mean_sum")
    ).to(device)
    count_criterion = RelativeCountSmoothL1Loss(
        offset=float(count_loss_config.get("offset", 1.0)),
        beta=float(count_loss_config.get("beta", 0.1)),
    ).to(device)
    optimizer = torch.optim.Adam(
        chain(student.parameters(), adapters.parameters()),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 1000)),
        min_lr=float(training.get("minimum_learning_rate", 1e-7)),
    )

    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
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
        student.load_state_dict(checkpoint["model_state"])
        if "adapter_state" not in checkpoint:
            raise KeyError("B3 checkpoint缺少adapter_state，无法恢复蒸馏训练")
        adapters.load_state_dict(checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint.get("best_mae", best_mae))
        print(f"已恢复B3 checkpoint：{resume_path}，从epoch {start_epoch}继续")

    total_epochs = int(args.epochs or training.get("epochs", 200))
    if start_epoch > total_epochs:
        raise ValueError(
            f"checkpoint已训练到epoch {start_epoch - 1}，不小于目标epoch {total_epochs}"
        )

    gt_density_weight = float(distillation.get("gt_density_weight", 1.0))
    output_distillation_weight = float(
        distillation.get("output_distillation_weight", 1.0)
    )
    fsp_weight = float(distillation.get("fsp_weight", 0.5))
    cosine_weight = float(distillation.get("cosine_weight", 0.5))
    count_loss_weight = float(distillation.get("count_loss_weight", 0.0))
    fsp_scales = tuple(int(value) for value in distillation.get("fsp_scales", (3, 2, 1)))
    if any(
        value < 0.0
        for value in (
            gt_density_weight,
            output_distillation_weight,
            fsp_weight,
            cosine_weight,
            count_loss_weight,
        )
    ):
        raise ValueError("所有训练损失权重均不能为负数")

    instance_mask_data = reliability.get("instance_mask", {})
    reliability_config = ReliabilityDistillationConfig(
        mode=str(reliability.get("mode", "combined")),
        tau_map=float(reliability["tau_map"]),
        tau_count=float(reliability["tau_count"]),
        tau_consistency=float(reliability["tau_consistency"]),
        minimum=float(reliability.get("minimum", 0.05)),
        background_value=float(reliability.get("background_value", 0.05)),
        normalize_mean=bool(reliability.get("normalize_mean", True)),
        strength=float(
            args.reliability_strength
            if args.reliability_strength is not None
            else reliability.get("strength", 1.0)
        ),
        epsilon=float(reliability.get("epsilon", 1e-8)),
        instance_mask=InstanceMaskConfig(
            sigma_scale=float(instance_mask_data.get("sigma_scale", 0.35)),
            sigma_min=float(instance_mask_data.get("sigma_min", 0.75)),
            sigma_max=float(instance_mask_data.get("sigma_max", 4.0)),
            truncate=float(instance_mask_data.get("truncate", 2.5)),
            epsilon=float(instance_mask_data.get("epsilon", 1e-8)),
        ),
    )
    reliability_config.validate()

    resolved_config = dict(config)
    resolved_config["runtime"] = {
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
        "teacher_checkpoint": str(teacher_checkpoint_path.resolve()),
        "teacher_epoch": teacher_epoch,
        "overfit_samples": args.overfit_samples,
        "reliability_strength": reliability_config.strength,
        "resume": str(resume_path) if resume_path else None,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    student_parameters = trainable_parameter_count(student)
    adapter_parameters = trainable_parameter_count(adapters)
    print(f"设备：{device}")
    print(f"模型：1/{channel_ratio}-CSRNet学生（B3，可靠性SKT蒸馏）")
    print(f"教师：{teacher_checkpoint_path}（epoch={teacher_epoch}，已冻结）")
    print(f"训练样本：{len(train_dataset)}，验证样本：{len(validation_dataset)}")
    print(
        f"部署学生参数：{student_parameters:,}，"
        f"训练期适配层参数：{adapter_parameters:,}"
    )
    print(
        "损失权重："
        f"GT={gt_density_weight:g}，output={output_distillation_weight:g}，"
        f"FSP={fsp_weight:g}，cosine={cosine_weight:g}，"
        f"count={count_loss_weight:g}"
    )
    print(
        "计数损失：relative Smooth L1，"
        f"offset={count_criterion.offset:g}，beta={count_criterion.beta:g}"
    )
    print(
        "可靠性："
        f"mode={reliability_config.mode}，"
        f"tau=({reliability_config.tau_map:.8g}, "
        f"{reliability_config.tau_count:.8g}, "
        f"{reliability_config.tau_consistency:.8g})，"
        f"minimum={reliability_config.minimum:g}，"
        f"background={reliability_config.background_value:g}，"
        f"strength={reliability_config.strength:g}，"
        f"mean_normalization={reliability_config.normalize_mean}"
    )
    print(f"batch={batch_size}，AMP={amp_enabled}，输出目录：{output_dir}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = train_reliable_skt_one_epoch(
            teacher=teacher,
            student=student,
            adapters=adapters,
            loader=train_loader,
            density_criterion=density_criterion,
            count_criterion=count_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
            print_frequency=int(training.get("print_frequency", 20)),
            gt_density_weight=gt_density_weight,
            output_distillation_weight=output_distillation_weight,
            fsp_weight=fsp_weight,
            cosine_weight=cosine_weight,
            count_loss_weight=count_loss_weight,
            fsp_scales=fsp_scales,
            reliability_config=reliability_config,
            gradient_clip_norm=training.get("gradient_clip_norm"),
        )
        validation_metrics = evaluate_model(
            model=student,
            loader=validation_loader,
            device=device,
            criterion=density_criterion,
            amp_enabled=amp_enabled,
            description=f"B3 reliable SKT validation epoch={epoch:03d}",
        )
        scheduler.step(validation_metrics["mae"])

        is_best = validation_metrics["mae"] < best_mae
        if is_best:
            best_mae = validation_metrics["mae"]

        checkpoint_payload = {
            "epoch": epoch,
            "experiment": "B3 quarter-CSRNet with instance-reliable SKT distillation",
            "model_name": "CSRNetStudent",
            "channel_ratio": channel_ratio,
            # 部署模型与训练期适配层分开保存，评估脚本只读取model_state。
            "model_state": student.state_dict(),
            "adapter_state": adapters.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "teacher_checkpoint": str(teacher_checkpoint_path.resolve()),
            "teacher_epoch": teacher_epoch,
            "reliability_configuration": asdict(reliability_config),
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
                "train_total_loss": train_metrics["loss"],
                "train_gt_density_loss": train_metrics["gt_density_loss"],
                "train_output_distillation_loss": train_metrics[
                    "output_distillation_loss"
                ],
                "train_fsp_loss": train_metrics["fsp_loss"],
                "train_cosine_loss": train_metrics["cosine_loss"],
                "train_count_loss": train_metrics["count_loss"],
                "train_object_reliability_mean": train_metrics[
                    "object_reliability_mean"
                ],
                "train_reliability_raw_map_mean": train_metrics[
                    "reliability_raw_map_mean"
                ],
                "train_reliability_normalized_map_mean": train_metrics[
                    "reliability_normalized_map_mean"
                ],
                "train_maximum_normalized_weight": train_metrics[
                    "maximum_normalized_weight"
                ],
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
        "experiment": "B3 quarter-CSRNet with instance-reliable SKT distillation",
        "epochs": total_epochs,
        "channel_ratio": channel_ratio,
        "deployment_student_parameters": student_parameters,
        "training_adapter_parameters": adapter_parameters,
        "teacher_checkpoint": str(teacher_checkpoint_path.resolve()),
        "teacher_epoch": teacher_epoch,
        "reliability_configuration": asdict(reliability_config),
        "count_loss_weight": count_loss_weight,
        "count_loss_offset": count_criterion.offset,
        "count_loss_beta": count_criterion.beta,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_mae": best_mae,
        "best_checkpoint": str((output_dir / "best_mae.pth").resolve()),
        "last_checkpoint": str((output_dir / "last.pth").resolve()),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"B3可靠性SKT训练完成，最佳验证MAE={best_mae:.3f}")


if __name__ == "__main__":
    main()
