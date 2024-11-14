import torch
from torch import nn
from models.AASIST import Model

def load_model(model_path):
    model_config={
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    # 加载模型架构
    model = Model(model_config)  # 创建 AASIST 模型实例
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model
