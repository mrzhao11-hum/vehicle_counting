# 配置目录

当前CARPK固定核实验配置包括：

- `carpk_teacher_fixed.yaml`：827张开发训练集 + 162张验证集，用验证MAE选模。
- `carpk_teacher_fixed_full.yaml`：官方989张完整训练集，按等优化步数固定训练
  11轮并保存正式模型。
- `carpk_student_fixed.yaml`：B1的1/4-CSRNet学生，使用827张开发训练集和
  162张验证集，不加载教师、不使用知识蒸馏。
- `carpk_student_skt_fixed.yaml`：B2使用相同quarter学生和训练协议，加载
  已冻结的最佳教师，加入输出、Dense-FSP和余弦结构蒸馏。

配置记录：

- 原始数据、manifest和H5目录；
- batch、worker和水平翻转；
- ImageNet前端初始化；
- 学习率、epoch、AMP和学习率调度；
- 输出目录和随机种子。

训练入口会把最终解析的配置复制到实验目录的 `resolved_config.yaml`，保证
checkpoint、指标和实际训练条件可以对应。

B1学生使用原SKT公开训练脚本的Adam、学习率`1e-4`和Kaiming初始化。后续
B2原始SKT蒸馏也应保持相同学生结构、优化器和学习率，使B1与B2之间只有
“是否加入蒸馏监督”这一主要变量。

B2配置中的教师checkpoint固定为FP32、固定学习率教师的验证最佳权重。
训练期1x1适配层只负责对齐特征通道，不进入部署学生的参数量和FLOPs统计。

目前B0教师开发划分的最佳模型为13轮，即约10751次更新；正式配置在989张图上训练
11轮，即约10879次更新。正式训练关闭验证选模和`ReduceLROnPlateau`，
避免把已加入训练的162张图片再次当作验证集。
