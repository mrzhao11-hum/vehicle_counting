# 数据加载模块

`carpk.py` 提供 `CARPKDataset` 和 `carpk_collate_fn`，联合读取RGB图像、
HDF5密度图、小目标空间权重、有效区域掩码、点和原始边界框。

CSRNet输出步长为8时，Dataset会把密度图下采样到输入的1/8，并重新校正
密度积分，保证下采样前后车辆总数不变。

```python
from torch.utils.data import DataLoader

from datasets import CARPKDataset, carpk_collate_fn

dataset = CARPKDataset(
    data_root="data/raw/CARPK/CARPK_devkit/data",
    manifest="data/manifests/CARPK/train.json",
    target_root="data/processed/CARPK/fixed_sigma",
    output_stride=8,
    horizontal_flip_probability=0.5,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=carpk_collate_fn,
)
```

验证集和测试集必须将 `horizontal_flip_probability` 设为0。
