# 训练与评估引擎

当前包含：

- `trainer.py`：教师/学生共享的密度监督单epoch训练、AMP和梯度裁剪。
- `skt_trainer.py`：冻结教师，联合训练学生和特征适配层，并分别记录GT、
  输出蒸馏、Dense-FSP和余弦损失。
- `evaluator.py`：FP32计数积分、MAE/RMSE、正负密度诊断、逐图CSV、常规与
  最大误差样本可视化。
- `common.py`：随机种子、指标累计、原子checkpoint和断点读取。

开发配置只使用验证集选择最佳checkpoint，不会在训练过程中运行测试集。
正式配置使用全部官方训练图固定轮数训练，不再使用验证集选模，并输出
`final.pth`。`evaluate_teacher.py`和`evaluate_student.py`分别用于教师与
学生训练结束后的独立测试。

B2训练入口为：

```bash
python check_skt.py --device cuda:0

python train_skt.py --config configs/carpk_student_skt_fixed.yaml
```
