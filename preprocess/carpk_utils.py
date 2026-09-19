"""CARPK 数据集的公共解析与路径工具。

这个模块只负责读取官方原始文件，不生成训练标签，也不会修改原始数据。
将公共逻辑集中在这里，可以保证检查脚本、划分脚本和密度图脚本使用完全
一致的标注解释方式，避免不同脚本把坐标顺序理解错。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


IMAGE_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class BoundingBox:
    """一辆车的轴对齐边界框。

    CARPK 官方标注每行是 ``x1 y1 x2 y2 class_id``。官方坐标位于图像
    范围内，最后一列恒为 1。这里仍保留类别字段，方便以后统一处理其他
    车辆数据集。
    """

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """返回浮点中心坐标 ``(x, y)``，避免偶数宽高框产生整数偏差。"""

        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2, self.class_id]


def resolve_data_root(root: str | Path) -> Path:
    """把用户传入的不同层级路径解析到包含三个官方目录的 ``data``。

    支持以下任意一种输入：

    - ``.../CARPK_devkit/data``
    - ``.../CARPK_devkit``
    - ``.../CARPK``（其下包含 ``CARPK_devkit``）
    """

    root = Path(root).expanduser().resolve()
    candidates = (root, root / "data", root / "CARPK_devkit" / "data")
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("Images", "Annotations", "ImageSets")):
            return candidate
    raise FileNotFoundError(
        "未找到 CARPK 数据目录。传入路径应直接或间接包含 "
        "Images、Annotations、ImageSets 三个目录："
        f"{root}"
    )


def read_annotation(path: str | Path) -> List[BoundingBox]:
    """读取一份 CARPK TXT 标注，并严格检查列数和数据类型。"""

    path = Path(path)
    boxes: List[BoundingBox] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_number} 应有5列，实际得到{len(fields)}列：{line!r}"
                )
            try:
                x1, y1, x2, y2, class_id = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} 含有非整数标注：{line!r}") from exc
            boxes.append(BoundingBox(x1, y1, x2, y2, class_id))
    return boxes


def validate_boxes(
    boxes: Iterable[BoundingBox], width: int, height: int
) -> list[str]:
    """检查框的面积、类别和坐标范围，返回便于汇总的错误文本。"""

    errors: list[str] = []
    for index, box in enumerate(boxes, start=1):
        if box.width <= 0 or box.height <= 0:
            errors.append(f"第{index}个框面积无效：{box.as_list()}")
        if box.x1 < 0 or box.y1 < 0 or box.x2 >= width or box.y2 >= height:
            errors.append(
                f"第{index}个框越界：{box.as_list()}，图像尺寸为{width}x{height}"
            )
        if box.class_id != 1:
            errors.append(f"第{index}个框类别不是CARPK车辆类别1：{box.as_list()}")
    return errors


def read_split(path: str | Path) -> list[str]:
    """读取官方 ImageSets 文本，过滤空行并拒绝重复样本。"""

    path = Path(path)
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"划分文件中存在重复样本：{path}")
    return sample_ids


def find_image(images_dir: str | Path, sample_id: str) -> Path:
    """根据不带扩展名的样本ID寻找图像，并拒绝歧义匹配。"""

    images_dir = Path(images_dir)
    matches = [images_dir / f"{sample_id}{ext}" for ext in IMAGE_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"找不到样本 {sample_id} 的图像：{images_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"样本 {sample_id} 同时匹配多个图像：{matches}")
    return matches[0]


def sequence_name(sample_id: str) -> str:
    """从 ``20161029_NTU_00123`` 提取拍摄序列 ``20161029_NTU``。"""

    parts = sample_id.rsplit("_", maxsplit=1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"无法从样本ID提取序列名：{sample_id}")
    return parts[0]


def official_splits(data_root: str | Path) -> tuple[list[str], list[str]]:
    """返回官方训练和测试样本ID，并检查两者没有交集。"""

    data_root = resolve_data_root(data_root)
    train_ids = read_split(data_root / "ImageSets" / "train.txt")
    test_ids = read_split(data_root / "ImageSets" / "test.txt")
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"官方训练集与测试集有{len(overlap)}个重复样本：{preview}")
    return train_ids, test_ids

