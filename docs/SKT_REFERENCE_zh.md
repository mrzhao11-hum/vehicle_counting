# 原 SKT 项目参考说明

外部参考项目 `E:\SKT-master\SKT-master` 来自 `Efficient Crowd Counting via Structured Knowledge Transfer`。车辆工程参考其 CSRNet 和蒸馏思路，不在原文件上继续叠加车辆逻辑。

## 文件映射

| 原文件 | 原作用 | 车辆工程中的处理 |
| --- | --- | --- |
| `models/model_vgg.py` | 完整 CSRNet | 参考教师基础结构 |
| `models/model_teacher_vgg.py` | 教师与特征 hook | 参考蒸馏特征位置 |
| `models/model_student_vgg.py` | 通道缩减学生与 1x1 对齐 | 参考 quarter 学生 |
| `models/distillation.py` | 余弦与 FSP 损失 | 重写为可测试的 loss 模块 |
| `dataset.py`、`image.py` | 人群图像与 HDF5 加载 | 车辆工程不直接复用 |
| `SKT_distill.py` | 加载教师后蒸馏学生 | 拆为 trainer 和入口 |
| `test.py` | 计数测试与可视化 | 参考输出格式，统一为 evaluator |
| `utils.py` | checkpoint 和切图工具 | 按职责拆分 |

## 原蒸馏损失

原实现总损失为：

```text
学生与 GT 密度损失
+ 学生与教师输出密度损失
+ lambda_fsp * 跨层 FSP 关系损失
+ lambda_cos * 层内余弦特征损失
```

教师在 `torch.no_grad()` 中前向，学生参数通过 Adam 更新。学生训练状态返回多层对齐特征和最终输出；评估状态只返回密度图。

## 需要修正的研究流程

原入口将 batch、epoch、worker 等写死，并在验证变好时测试测试集。车辆工程将这些参数放入配置，只用验证集选择 checkpoint，实验方案确定后才运行测试集。

原 FSP `gram()` 只取 batch 中第一个样本，因此扩大 batch 前必须重写为完整 batch 矩阵运算。第一阶段可以使用 batch=1 对齐参考结果，但新实现仍应支持一般 batch。

原密度图路径通过字符串替换构造，车辆工程改为 manifest 显式记录。原 `reshape_target()` 使用插值乘 64 近似保持总量，新实现应在变换后核验每张密度图总和。
