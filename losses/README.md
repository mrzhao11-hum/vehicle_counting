# 损失模块

当前 `DensityMSELoss` 支持：

- 固定核普通密度图监督；
- 空间权重图；
- 有效区域掩码；
- `batch_mean_sum` 和 `weighted_mean` 两种归约方式。

固定核基线使用 `batch_mean_sum`，在batch为1时等价于原CSRNet/SKT的
`MSELoss(reduction="sum")`。后续增加权重实验时，所有对照必须使用一致的
归约方式，避免把梯度量级变化误当成方法收益。

`skt.py`实现B2原始SKT蒸馏所需组件：

- `SKTFeatureAdapters`：六个训练期1x1卷积，把quarter学生通道对齐到教师；
- `cosine_feature_loss`：六层对应特征的逐位置通道余弦距离；
- `dense_fsp_loss`：六个中间特征加最终密度输出，共21个跨层关系矩阵；
- `scale_features_for_fsp`：按原SKT的`[3, 2, 1]`尺度对齐到1/8分辨率。

适配层单独保存在B2 checkpoint的`adapter_state`中，不属于最终部署模型。
