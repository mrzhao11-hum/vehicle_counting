# 模型模块

当前已经实现 `csrnet.py` 中的完整 `CSRNetTeacher`：

- VGG16前端，使用前三次最大池化，输出步长为8。
- 六层空洞卷积后端，扩张率为2。
- 单通道密度图输出。
- 可选ImageNet VGG16前端初始化。
- `return_features=True`时显式返回六个中间特征，供后续SKT使用。

固定核教师基线保持原CSRNet行为，输出端不额外添加ReLU。可视化时可以
截断负值，但计算预测数量和指标时必须使用未经修改的原始输出。

同时已经实现B1使用的 `CSRNetStudent`：

- `channel_ratio=4`时，各中间卷积通道为教师的四分之一；
- 最终仍输出单通道、1/8分辨率密度图；
- 采用原SKT学生代码中的Kaiming正态初始化；
- 普通前向只计算计数主干，不包含蒸馏专用的1x1对齐层；
- `return_features=True`时返回六个原始学生特征，供B2单独对齐和蒸馏；
- B1和B2部署时使用同一个学生主干，保证参数量和速度比较公平。

结构快速检查：

```bash
python check_models.py --device cpu
```

复杂度与延迟统计：

```bash
python analyze_model.py --model teacher --device cuda:0 \
  --checkpoint outputs/carpk/b0_teacher_fixed_sigma8/best_mae.pth

python analyze_model.py --model student --device cuda:0 \
  --checkpoint outputs/carpk/b1_student_fixed_sigma8/best_mae.pth \
  --output outputs/carpk/b1_student_fixed_sigma8/complexity.json
```
