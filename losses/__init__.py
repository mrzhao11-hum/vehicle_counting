"""车辆密度估计损失函数。"""

from .counting import RelativeCountSmoothL1Loss
from .density import DensityMSELoss
from .reliable_distillation import (
    ReliabilityDistillationConfig,
    SUPPORTED_RELIABILITY_MODES,
    build_batch_reliability_maps,
    normalize_reliability_map,
)
from .skt import (
    SKTFeatureAdapters,
    batch_mean_sum_mse,
    cosine_feature_loss,
    dense_fsp_loss,
    scale_features_for_fsp,
)

__all__ = [
    "DensityMSELoss",
    "RelativeCountSmoothL1Loss",
    "ReliabilityDistillationConfig",
    "SUPPORTED_RELIABILITY_MODES",
    "SKTFeatureAdapters",
    "batch_mean_sum_mse",
    "build_batch_reliability_maps",
    "cosine_feature_loss",
    "dense_fsp_loss",
    "normalize_reliability_map",
    "scale_features_for_fsp",
]
