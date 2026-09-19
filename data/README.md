# 数据目录

建议结构：

```text
data/
├── raw/          # 官方原始数据，只读使用
├── processed/    # 自动生成的密度图、权重图和掩码
└── manifests/    # 固定的 train/val/test 清单
```

大数据文件不要提交到代码仓库。预处理产物必须能够由原始数据和配置重新生成。

当前CARPK实际路径为：

```text
data/raw/CARPK/CARPK_devkit/data/
├── Images/       # 1448张1280x720 PNG
├── Annotations/  # 1448份TXT，共89774个车辆框
└── ImageSets/    # 官方train 989张、test 459张
```

`PUCPR+_devkit`和作者原始工具保留在 `data/raw/CARPK`，第一阶段训练不读取它们。
