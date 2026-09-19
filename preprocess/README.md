# CARPK 数据预处理

本目录的程序只读取 `data/raw`，不会修改官方图像、标注或划分文件。

## 文件说明

- `carpk_utils.py`：统一解析路径、官方划分和 `x1 y1 x2 y2 class` 标注。
- `inspect_carpk.py`：检查完整性、统计分布并绘制标注框。
- `make_carpk_splits.py`：保留官方测试集，按拍摄序列建立开发训练集和验证集。
- `generate_carpk_density.py`：生成固定核、自适应核及可选小目标权重HDF5。

## 1. 检查原始数据

```bash
python preprocess/inspect_carpk.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --output outputs/data_check/carpk \
  --num-visualizations 20
```

成功标准：程序最后显示 `完整性错误：0`。重点人工查看：

```text
outputs/data_check/carpk/
├── dataset_report.json
├── per_image_stats.csv
├── distributions.png
└── bbox_visualizations/
```

## 2. 生成固定划分

```bash
python preprocess/make_carpk_splits.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --output data/manifests/CARPK \
  --val-sequence 20161029_NTU
```

得到 `train.json`（827张）、`val.json`（162张）、官方 `test.json`
（459张）和用于最终重训练的 `train_full.json`（989张）。

## 3. 小规模验证密度图代码

```bash
python preprocess/generate_carpk_density.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --mode fixed \
  --fixed-sigma 8 \
  --output outputs/preprocess_smoke_test/fixed_sigma \
  --limit 8 \
  --num-visualizations 8
```

## 4. 生成完整固定核基线

```bash
python preprocess/generate_carpk_density.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --mode fixed \
  --fixed-sigma 8 \
  --output data/processed/CARPK/fixed_sigma \
  --workers 4 \
  --num-visualizations 20
```

## 5. 生成自适应核标签

```bash
python preprocess/generate_carpk_density.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --mode adaptive \
  --adaptive-alpha 0.15 \
  --sigma-min 2 \
  --sigma-max 15 \
  --output data/processed/CARPK/adaptive_sigma \
  --workers 4 \
  --num-visualizations 20
```

## 6. 生成小目标权重实验标签

```bash
python preprocess/generate_carpk_density.py \
  --root data/raw/CARPK/CARPK_devkit/data \
  --mode adaptive \
  --adaptive-alpha 0.15 \
  --sigma-min 2 \
  --sigma-max 15 \
  --with-scale-weight \
  --max-scale-weight 3 \
  --output data/processed/CARPK/adaptive_sigma_weighted \
  --workers 4 \
  --num-visualizations 20
```

脚本支持续跑：配置一致时已有HDF5会被检查并跳过。配置改变时请换输出目录；
只有明确希望重建同一目录时才使用 `--overwrite`。
