"""快速检查CSRNet教师和轻量学生的结构、输出及中间特征。

该脚本不读取CARPK数据，也不训练模型，适合代码上传服务器后首先执行：

    python check_models.py --device cpu

若只想利用GPU更快完成前向检查，可改为``--device cuda:0``。
"""

from __future__ import annotations

import argparse

import torch

from engine.common import trainable_parameter_count
from models import CSRNetStudent, CSRNetTeacher, TEACHER_FEATURE_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查CSRNet教师/学生结构")
    parser.add_argument("--device", default="cpu", help="cpu或cuda:0")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--channel-ratio", type=int, default=4)
    return parser.parse_args()


def feature_shapes(output: dict[str, object]) -> list[tuple[int, ...]]:
    features = output["features"]
    if not isinstance(features, list):
        raise TypeError("return_features=True时features必须是列表")
    return [tuple(feature.shape) for feature in features]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但当前PyTorch检测不到GPU")
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("为便于明确核验，请让height和width都能被8整除")

    # 教师检查时关闭ImageNet下载；这里只验证结构，不验证预训练参数。
    teacher = CSRNetTeacher(pretrained_frontend=False).to(device).eval()
    student = CSRNetStudent(channel_ratio=args.channel_ratio).to(device).eval()
    image = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.inference_mode():
        teacher_output = teacher(image, return_features=True)
        student_output = student(image, return_features=True)
        plain_student_output = student(image)

    if not isinstance(teacher_output, dict) or not isinstance(student_output, dict):
        raise TypeError("模型的return_features接口未返回字典")
    expected_density_shape = (1, 1, args.height // 8, args.width // 8)
    if tuple(teacher_output["density"].shape) != expected_density_shape:
        raise AssertionError("教师密度图尺寸不正确")
    if tuple(student_output["density"].shape) != expected_density_shape:
        raise AssertionError("学生密度图尺寸不正确")
    if tuple(plain_student_output.shape) != expected_density_shape:
        raise AssertionError("学生普通前向输出尺寸不正确")

    teacher_shapes = feature_shapes(teacher_output)
    student_shapes = feature_shapes(student_output)
    if len(teacher_shapes) != 6 or len(student_shapes) != 6:
        raise AssertionError("教师和学生都必须提供六个SKT中间特征")
    if tuple(shape[1] for shape in teacher_shapes) != TEACHER_FEATURE_CHANNELS:
        raise AssertionError("教师特征通道与SKT约定不一致")
    if tuple(shape[1] for shape in student_shapes) != student.feature_channels:
        raise AssertionError("学生特征通道与模型声明不一致")
    if any(t[2:] != s[2:] for t, s in zip(teacher_shapes, student_shapes, strict=True)):
        raise AssertionError("对应的教师/学生特征空间尺寸不一致")

    teacher_parameters = trainable_parameter_count(teacher)
    student_parameters = trainable_parameter_count(student)
    print(f"输入尺寸：{tuple(image.shape)}")
    print(f"密度图尺寸：{expected_density_shape}")
    print(f"教师特征：{teacher_shapes}")
    print(f"学生特征：{student_shapes}")
    print(f"教师参数：{teacher_parameters:,}")
    print(f"学生参数：{student_parameters:,}")
    print(f"学生/教师参数比例：{student_parameters / teacher_parameters:.4%}")
    print("CSRNet教师/学生结构检查通过。")


if __name__ == "__main__":
    main()
