"""车辆计数数据集读取接口。"""

from .carpk import CARPKDataset, carpk_collate_fn

__all__ = ["CARPKDataset", "carpk_collate_fn"]
