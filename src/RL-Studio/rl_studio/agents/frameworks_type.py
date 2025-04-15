from enum import Enum


class FrameworksType(Enum):
    TF = "TensorFlow"
    PYTORCH = "Pytorch"
    BASELINES = "baselines"
    BASELINES_2 = "baselines_2_networks"
    BASELINES_2_SIMPLE = "baselines_2_simple_networks"
    PARALLEL_BASELINES = "parallel_baselines"

