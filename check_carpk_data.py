"""快速验证CARPK Dataset、降采样密度积分和批处理尺寸。"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from datasets import CARPKDataset, carpk_collate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="检查CARPK训练DataLoader")
    parser.add_argument(
        "--data-root", default="data/raw/CARPK/CARPK_devkit/data"
    )
    parser.add_argument(
        "--manifest", default="data/manifests/CARPK/train.json"
    )
    parser.add_argument(
        "--target-root", default="data/processed/CARPK/fixed_sigma"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    dataset = CARPKDataset(
        data_root=args.data_root,
        manifest=args.manifest,
        target_root=args.target_root,
        output_stride=8,
        horizontal_flip_probability=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=carpk_collate_fn,
    )
    batch = next(iter(loader))
    density_counts = batch["density"].sum(dim=(1, 2, 3))
    maximum_error = (density_counts - batch["count"]).abs().max().item()

    print(f"dataset length: {len(dataset)}")
    print(f"image shape: {tuple(batch['image'].shape)}")
    print(f"density shape: {tuple(batch['density'].shape)}")
    print(f"weight shape: {tuple(batch['weight'].shape)}")
    print(f"valid mask shape: {tuple(batch['valid_mask'].shape)}")
    print(f"density sums: {density_counts.tolist()}")
    print(f"GT counts: {batch['count'].tolist()}")
    print(f"maximum count error: {maximum_error:.8f}")
    print(
        "weight range: "
        f"[{batch['weight'].min().item():.3f}, {batch['weight'].max().item():.3f}]"
    )
    print(
        "valid mask range: "
        f"[{batch['valid_mask'].min().item():.3f}, "
        f"{batch['valid_mask'].max().item():.3f}]"
    )

    if maximum_error > 1e-3:
        raise RuntimeError("降采样后的密度积分误差超过1e-3")
    print("CARPK DataLoader检查通过。")


if __name__ == "__main__":
    main()
