# 无人机轻量级密度图车辆计数工程

本目录 `E:\车辆计数` 是后续车辆计数工作的唯一开发目录。另一个项目 `E:\SKT-master\SKT-master` 保留为只读参考实现，不再直接加入车辆数据处理、车辆损失或车辆训练逻辑。

当前状态：CARPK原始数据、密度标签与固定划分已经完成；B0完整CSRNet教师
训练和测试已经跑通。B1的1/4-CSRNet学生模型、无蒸馏训练、独立评估、结构
检查与复杂度统计已经实现；原始SKT蒸馏仍属于下一阶段。

## 1. 项目目标

输入无人机图像或视频帧，模型输出单通道车辆密度图：

```text
图像 -> 密度图模型 -> 预测密度图 -> 对像素求和 -> 当前画面车辆数
```

第一阶段完成瞬时车辆数量估计和车辆分布热力图。模型不输出车辆检测框、车辆 ID、速度或累计通过车辆数。若以后需要测速和轨迹，应另接检测与跟踪模块。

研究主线：

```text
CARPK 跑通标准计数流程
    -> UAVDT 完成道路车辆主实验
    -> bbox 尺度感知密度标签
    -> 小目标空间加权监督
    -> 小目标感知知识蒸馏
    -> 轻量学生模型部署测试
```

## 2. 新旧项目边界

```text
E:/
├── SKT-master/SKT-master/
│   ├── models/、dataset.py、image.py、SKT_distill.py、test.py
│   └── 原人群计数项目，只作为结构和算法参考
│
└── 车辆计数/
    └── 新车辆计数工程，后续车辆相关修改全部放在这里
```

可以参考旧项目的内容：

- CSRNet 前端、空洞卷积后端和 1/8 输出结构。
- quarter-CSRNet 的通道缩减方式。
- 教师中间特征 hook 的位置。
- 层内余弦损失和跨层 FSP 关系蒸馏思想。
- checkpoint 保存与加载的大体格式。

不直接沿用的内容：

- 根据字符串替换猜测 `.h5` 路径的方式。
- ShanghaiTech 专用随机裁剪和训练列表重复四次。
- 固定 batch=1、1000 epochs 和以当前时间作为随机种子。
- 训练期间反复查看测试集结果。
- 只支持人头点标注的数据预处理。
- 把输出日志中的 RMSE 标为 MSE。

## 3. 规划后的目录结构

```text
车辆计数/
├── README_zh.md
├── requirements.txt
├── configs/
│   ├── README.md
│   ├── carpk_baseline.yaml
│   └── uavdt_baseline.yaml
├── data/
│   ├── README.md
│   ├── raw/                  # 原始数据，不修改
│   ├── processed/            # 密度图、权重图、有效区域掩码
│   └── manifests/            # 固定的 train/val/test JSON
├── datasets/
│   ├── carpk.py
│   ├── uavdt.py
│   ├── vehicle_dataset.py
│   └── transforms.py
├── preprocess/
│   ├── carpk_utils.py
│   ├── inspect_carpk.py
│   ├── make_carpk_splits.py
│   ├── generate_carpk_density.py
│   └── parse_uavdt.py              # 后续实现
├── models/
│   └── csrnet.py                  # CSRNet教师、1/4学生和显式中间特征
├── losses/
│   ├── density_loss.py
│   └── density.py                 # 已实现加权/掩码密度MSE
├── engine/
│   ├── trainer.py                 # 已实现教师单epoch训练
│   ├── evaluator.py               # 已实现MAE/RMSE、CSV与可视化
│   └── common.py                  # checkpoint、随机种子和指标工具
├── scripts/
│   └── README.md
├── docs/
│   ├── VEHICLE_COUNTING_ROADMAP_zh.md
│   └── SKT_REFERENCE_zh.md
├── outputs/                  # checkpoint、日志、CSV、可视化
├── check_carpk_data.py            # 已实现DataLoader快速检查
├── train_teacher.py               # 已实现教师训练和8图过拟合
├── evaluate_teacher.py            # 已实现教师独立评估
├── train_student.py              # B1无蒸馏学生训练
├── evaluate_student.py           # B1/B2学生独立评估
├── check_models.py               # 教师/学生结构快速检查
├── analyze_model.py              # 参数、FLOPs、体积和延迟统计
├── train_distill.py
├── evaluate.py
└── infer.py
```

代码不会一次性全部铺开。按“数据核验 -> 教师 -> 学生基线 -> 蒸馏 -> 改进损失”的顺序实现，每一阶段通过检查后再进入下一阶段。

## 4. 各部分职责

### configs

每次实验的唯一配置来源，记录数据路径、类别范围、输入尺寸、密度核、损失系数、学习率、epoch、随机种子和输出目录。配置文件随结果保存，避免事后不知道某个权重怎样训练。

### data

`raw` 只保存官方数据；`processed` 保存由脚本生成的标签；`manifests` 保存数据划分。训练代码只读取 manifest，不在运行时扫描目录并临时随机划分。

### datasets

解析 manifest，联合读取图像、密度图、权重图和有效区域掩码。裁剪、缩放、翻转必须对所有内容同步执行。模型使用 1/8 输出时，密度图下采样必须保持总和。

### preprocess

把 CARPK、UAVDT 不同格式的框统一为内部格式，再生成中心点、密度图和权重图。预处理与模型训练分离，便于检查标签和断点续跑。

### models

教师采用完整 CSRNet；学生先采用通道数约为四分之一的 CSRNet。中间层特征通过明确接口返回，减少对 forward hook 隐式状态的依赖。

### losses

将真实密度监督、教师输出蒸馏、层内特征蒸馏和跨层关系蒸馏分开实现。每项损失独立记录，支持配置开关，方便做消融实验。

### engine

统一训练、验证和测试循环。训练集更新参数；验证集选择最佳 checkpoint；测试集只在实验确定后评估。测试不参与调参。

### utils

统一管理 checkpoint、MAE/RMSE、参数量、延迟测试、随机种子和可视化，避免各入口重复实现且计算口径不一致。

## 5. 数据样本的统一表示

manifest 中每张图片至少记录：

```json
{
  "id": "carpk_000001",
  "dataset": "carpk",
  "split": "train",
  "sequence": "parking_lot_1",
  "image": "data/raw/CARPK/images/000001.png",
  "annotation": "data/raw/CARPK/annotations/000001.txt",
  "target": "data/processed/CARPK/fixed/000001.h5",
  "width": 1280,
  "height": 720,
  "count": 73
}
```

`.h5` 规划保存：

```text
density    单位质量车辆密度图，sum 约等于车辆数
weight     小目标空间权重图，基础实验为全 1
valid_mask 有效监督区域，忽略区域为 0
count      有效车辆数量
```

框标注仍保存在 manifest 指向的标准化标签中，不把 `.h5` 当作唯一真值来源。

## 6. 数据集安排

### CARPK

用途：首先跑通标签、教师、学生和蒸馏；完成与已有车辆计数方法同类的基准实验。

初始计数口径：CARPK 官方有效车辆框全部计数。优先沿用官方 train/test 列表，从 train 中按场景或序列固定留出 validation。不能每次启动训练重新随机划分。

当前实际原始目录：

```text
E:/车辆计数/data/raw/CARPK/CARPK_devkit/data/
├── Images/       # 1448张
├── Annotations/  # 1448份，共89774辆
└── ImageSets/    # train=989，test=459
```

实际名称以下载包为准，解析脚本接受配置路径，不要求用户手动改官方目录名。

### UAVDT

用途：道路车辆主实验，验证密集、小目标、不同视角和相机运动场景。

第一轮跨数据集实验统一为 car-only；道路完整实验再按确认后的官方类别合并 car、truck、bus。必须先核对官方 GT 格式，不能套用 VisDrone 的列定义。

按视频序列划分 train/val/test；相邻帧不能跨集合。可在训练序列按固定间隔抽帧以降低重复，测试帧名单必须固定。

建议原始目录：

```text
E:/车辆计数/data/raw/UAVDT/
├── images_or_sequences/
├── annotations/
└── attributes/
```

## 7. 标签生成流程

### 基础固定核

```text
读取 bbox
-> 校验坐标、类别、有效性
-> 计算中心点
-> 每个中心放置固定 sigma 高斯核
-> 对图像边缘截断后的每个核重新归一化为 sum=1
-> 累加并保存 density
```

基础版本中 `weight=1`，有效区域 `valid_mask=1`；官方忽略区域映射为 0。

### bbox 尺度核

候选定义：

```text
sigma_x = clip(alpha_x * bbox_width, sigma_min, sigma_max)
sigma_y = clip(alpha_y * bbox_height, sigma_min, sigma_max)
```

每辆车的核仍归一化为 1。sigma 只改变空间形状，不改变一辆车对总数量的贡献。

### 小目标权重

按训练集框面积统计参考尺度，小框附近权重大，普通背景权重保持 1，并设置上限防止梯度过大。训练、验证和测试不能共同统计参考尺度。

## 8. 标签生成后的强制检查

预处理完成后不能立即训练，必须通过以下检查：

1. 随机可视化至少 20 张图片的框、中心、密度图和权重图。
2. 检查每张图 `abs(density.sum() - count)` 是否处于数值容差内。
3. 检查越界框、零面积框、边缘框、空图和忽略区域。
4. 检查图像与标签文件一一对应，不多不少。
5. 检查 train/val/test 无同图泄漏。
6. UAVDT 检查同一视频序列没有跨 split。
7. 输出数据统计：图片数、车辆数、每图数量分布、框面积分布和无效框数量。

## 9. 完整训练链路

### 阶段 A：训练教师

```text
训练图像 + 真实密度图
        -> 完整 CSRNet
        -> 预测密度图
        -> 密度损失
        -> 反向传播更新教师
```

教师负责建立车辆计数上限。它可以用 ImageNet VGG16 初始化前端，但必须在目标车辆训练集上训练。原 SKT 人群 checkpoint 不能直接当作车辆教师的最终权重。

开发阶段每个 epoch 后只在 validation 计算 MAE/RMSE，以 validation MAE 保存
best checkpoint。确定训练配置和最佳轮数后，使用官方989张完整训练集固定
轮数重新训练，不再使用validation选模，最后再独立测试。

先运行8张图过拟合检查：

```bash
python train_teacher.py \
  --config configs/carpk_teacher_fixed.yaml \
  --overfit-samples 8 \
  --epochs 200 \
  --output-dir outputs/overfit/carpk_teacher_fixed_8
```

完整训练命令：

```bash
python train_teacher.py --config configs/carpk_teacher_fixed.yaml
```

开发实验确定训练预算后，正式B0使用全部989张官方训练图。开发阶段13轮约
为10751次参数更新；正式阶段采用等优化步数原则训练11轮，约10879次更新：

```bash
python train_teacher.py --config configs/carpk_teacher_fixed_full.yaml
```

正式模型保存为：

```text
outputs/carpk/b0_teacher_fixed_sigma8_full/final.pth
```

### 阶段 B：独立训练学生基线

使用相同数据、标签、增强、epoch 和验证规则，只训练 quarter-CSRNet，不加载教师、不使用蒸馏。它回答“模型缩小后本身能达到什么水平”。

先做结构检查和8张图过拟合：

```bash
python check_models.py --device cpu

python train_student.py \
  --config configs/carpk_student_fixed.yaml \
  --overfit-samples 8 \
  --epochs 200 \
  --output-dir outputs/overfit/b1_student_fixed_8
```

确认小样本能够拟合后，进行B1完整开发训练：

```bash
python train_student.py --config configs/carpk_student_fixed.yaml
```

测试B1最佳验证权重：

```bash
python evaluate_student.py \
  --config configs/carpk_student_fixed.yaml \
  --checkpoint outputs/carpk/b1_student_fixed_sigma8/best_mae.pth \
  --split test \
  --output-dir outputs/carpk/b1_student_fixed_sigma8/test \
  --save-visualizations 20 \
  --save-worst-visualizations 20
```

### 阶段 C：蒸馏训练学生

```text
同一张图像
├── 冻结教师 -> 教师密度图 + 中间特征
└── 学生     -> 学生密度图 + 对齐后的中间特征

总损失 = 真实标签监督
       + 教师输出蒸馏
       + 层内特征蒸馏
       + 跨层关系蒸馏
```

教师必须 `eval()`、关闭梯度且参数不进入优化器；学生和特征对齐层参与更新。每项损失单独写日志，不能只打印总损失。

规划命令：

```bash
python train_distill.py \
  --config configs/carpk_baseline.yaml \
  --teacher outputs/carpk/teacher/best_mae.pth
```

### 阶段 D：加入研究改进

固定基础训练协议后，按顺序增加：

```text
固定核 + 原始 SKT
bbox 尺度核 + 原始 SKT
bbox 尺度核 + 小目标加权密度监督
bbox 尺度核 + 小目标加权输出蒸馏
bbox 尺度核 + 小目标加权局部特征蒸馏
```

每步只改变一个主要因素。跨层 FSP 已经聚合为空间关系矩阵，第一版保持原式，不直接乘二维权重图。

## 10. 测试流程

测试入口只做推理和指标计算，不更新参数：

```text
读取固定 test manifest
-> 加载明确的 best checkpoint
-> model.eval() + no_grad()
-> 每张图输出密度图
-> pred_count = density.sum()
-> 与 gt_count 比较
-> 保存逐图 CSV、热力图和汇总指标
```

开发教师测试命令：

```bash
python evaluate_teacher.py \
  --config configs/carpk_teacher_fixed.yaml \
  --checkpoint outputs/carpk/b0_teacher_fixed_sigma8/best_mae.pth \
  --split test \
  --output-dir outputs/carpk/b0_teacher_fixed_sigma8/test \
  --save-visualizations 20 \
  --save-worst-visualizations 20
```

正式B0测试命令：

```bash
python evaluate_teacher.py \
  --config configs/carpk_teacher_fixed_full.yaml \
  --checkpoint outputs/carpk/b0_teacher_fixed_sigma8_full/final.pth \
  --split test \
  --output-dir outputs/carpk/b0_teacher_fixed_sigma8_full/test \
  --save-visualizations 20 \
  --save-worst-visualizations 20
```

至少输出：

```text
metrics.json
evaluation_report.json
per_image_results.csv
visualizations/
worst_visualizations/
```

开发训练目录另外保存 `resolved_config.yaml`、`history.csv`、`last.pth`、
`best_mae.pth` 和 `training_summary.json`；正式训练目录保存 `final.pth`，并在
逐图CSV中额外记录正密度积分与负密度质量。

指标定义：

```text
MAE  = mean(abs(pred_count - gt_count))
RMSE = sqrt(mean((pred_count - gt_count)^2))
```

不要把 RMSE 写成 MSE。测试时保留浮点预测数量，展示界面可以另行四舍五入。

## 11. 新图片和视频推理

图片推理不需要 GT：

```bash
python infer.py \
  --checkpoint outputs/uavdt/student_skt/best_mae.pth \
  --input demo/road.jpg \
  --output outputs/demo
```

视频推理逐帧得到“画面内车辆数量”。相邻帧出现同一辆车会重复出现在每帧数量中，因此不能把所有帧数量相加当作累计车流量。

## 12. 实验输出目录

每次实验使用独立目录：

```text
outputs/carpk/B2_fixed_skt_seed42/
├── config.yaml
├── checkpoints/
│   ├── last.pth
│   └── best_mae.pth
├── train.csv
├── val.csv
├── test/
│   ├── metrics.json
│   ├── per_image_results.csv
│   └── comparison/
└── environment.txt
```

目录名包含数据集、方法和 seed。checkpoint 中保存模型、优化器、epoch、最佳指标、配置摘要和随机种子，支持断点恢复。

## 13. 最小实验矩阵

| 编号 | 模型 | 标签 | 蒸馏/权重 | 目的 |
| --- | --- | --- | --- | --- |
| B0 | 完整教师 | 固定核 | 无 | 精度上限参考 |
| B1 | quarter 学生 | 固定核 | 无 | 轻量结构基线 |
| B2 | quarter 学生 | 固定核 | 原始 SKT | 蒸馏基线 |
| B3 | quarter 学生 | bbox 核 | 原始 SKT | 检查标签核影响 |
| B4 | quarter 学生 | bbox 核 | 加权密度监督 | 检查小目标监督 |
| B5 | quarter 学生 | bbox 核 | 加权监督和蒸馏 | 候选完整方法 |

先在 CARPK 运行 B0-B5。确认实现和收益后，将关键 B0、B1、B2、B5 迁移到 UAVDT。核心结果建议用至少 3 个随机种子报告均值和标准差。

## 14. 当前原 SKT 项目实际能做什么

外部参考项目 `E:\SKT-master\SKT-master` 当前可以：

- 把 ShanghaiTech/UCF-QNRF 人头标注转换为密度图。
- 读取图片和对应 `.h5` 密度图。
- 定义完整 CSRNet、教师 CSRNet 和 quarter-CSRNet。
- 加载已有教师 checkpoint 蒸馏学生。
- 测试完整或 quarter 模型，计算 MAE/RMSE并保存部分可视化。

外部参考项目当前不能完整完成车辆研究：

- 没有 CARPK/UAVDT 标注解析器。
- 没有从头训练教师的独立命令。
- 没有无蒸馏学生基线入口。
- 没有权重图、忽略掩码或尺度加权损失。
- 没有按视频序列组织 UAVDT 的数据划分。
- 没有车辆视频推理入口。

原项目的蒸馏命令必须已有教师权重：

```bash
python SKT_distill.py A_train.json A_val.json A_test.json \
  --lr 1e-4 \
  -tc checkpoints/teacher_vgg.pth.tar \
  -laf 0.5 \
  -lac 0.5 \
  --out output/skt_shanghai_a \
  --gpu 0
```

这条命令是人群项目参考，不是未来车辆训练命令。

## 15. 实施顺序

接下来严格按以下顺序开发：

1. [已实现] CARPK标注解析器、完整性检查和框可视化。
2. [已实现] 固定核、自适应核、尺度权重密度标签和积分检查。
3. [已实现] 按序列划分manifest和CARPK PyTorch Dataset。
4. [已完成] 教师训练、验证、断点恢复和独立测试；当前FP32固定学习率B0
   测试MAE为8.991、RMSE为10.692。
5. [已完成] 无蒸馏quarter学生B1；测试MAE为16.270、RMSE为20.737，
   部署参数量1.018M。
6. [已实现，待服务器链路检查与训练] B2原始SKT蒸馏，包括输出密度、
   Dense-FSP和余弦特征监督。
7. 实现 bbox 尺度核和小目标权重。
8. 实现加权输出和局部特征蒸馏。
9. 适配 UAVDT，并按序列完成道路实验。
10. 增加单图和视频推理、延迟与参数量评估。

每完成一步，都应先做小数据检查和短训练；标签错误时跑完整训练只会浪费时间。
