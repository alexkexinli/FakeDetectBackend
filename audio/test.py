import argparse
import json
import torch
from torch import Tensor
from pathlib import Path
from importlib import import_module
import soundfile as sf
import numpy as np

model_config={
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [1, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 获取模型
def get_model(model_config: dict, device: torch.device):
    module = import_module("audio.models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model
# 加载预训练模型
model = get_model(model_config, device)
model.eval()
model.to(device)


def extract_channel(file_path, channel_index=0):
    # 读取音频文件
    data, samplerate = sf.read(file_path)
    
    # 提取指定声道的数据
    channel_data = data[:, channel_index]  # 根据索引选择声道
    
    return channel_data, samplerate

def pad(x, max_len=64600):
    print("x.shape ",x.shape)
    x_len = x.shape[0]

    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


# 评估单个音频文件
def evaluate_audio(audio):
    X_pad = pad(audio, 64600)
    # input_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # 转为张量，并添加 batch 维度
    print("X_pad.shape ",X_pad.shape)
    x_inp = Tensor(X_pad).unsqueeze(0).to(device)
    print("x_inp.shape ",x_inp.shape)
    print(type(audio))
    with torch.no_grad():
        output = model(x_inp)  # 调用模型进行推理

 # 使用 softmax 或 sigmoid 对 logits 进行转换
    # probabilities = torch.softmax(output[1], dim=1)  # 对每个样本进行 softmax 转换
    # 或者：
    probabilities = torch.sigmoid(output[1])  # 如果是二分类，可以直接用 sigmoid
    print("probabilities :",probabilities)
    # 判断类别为 'True'（真实音频）的概率是否大于 0.5
    is_true_audio = probabilities[:, 1] >0.5  # probabilities[1] 是真实音频的概率
    print("is_true_audio:",is_true_audio)
    result = is_true_audio.item() 

    return result,probabilities


    

# 主函数
def evaluate_audio_from_path(audio_file_path: str):

    # 确保音频文件存在
    path = Path(audio_file_path)
    if not path.exists():
        print(f"Audio file {path} not found!")
        return

    audio, sr = sf.read(path)
    print(f"file {sr}")

    # 检查音频是否是双声道
    if len(audio.shape) > 1 and audio.shape[1] == 2:
        # 提取左右声道
        left_channel = audio[:, 0]  # 左声道
        right_channel = audio[:, 1]  # 右声道
        print("Separated stereo audio into left and right channels.")
    else:
        left_channel = audio
        right_channel = audio
        print("Audio is already mono.")

    # 调用模型进行评估
    result_left, probabilities_left = evaluate_audio(left_channel)
    result_right, probabilities_right = evaluate_audio(right_channel)

    # audio=audio.mean(axis=1)
    # result,probabilities = evaluate_audio(audio)
    print("result_left")
    print(result_left)
    print("result_right")
    print(result_right)
    result=False
    if result_left == True and result_right == True:
        result=True
    probabilities=torch.stack((probabilities_left, probabilities_right), dim=0)
    print(f"audio Evaluating {path.name}...")
    print("audio Result:", result)
    return result,probabilities




if __name__ == "__main__":

    # 模型和音频文件路径
    model_path = "./models/weights/AASIST-L.pth"  # 替换为模型路径
    audio_file_path_0 = "./test_audio/testvideo-channel_0.flac"  # 替换为你的 .flac 文件路径
    audio_file_path_1 = "./test_audio/testvideo-channel_1.flac"
    # 调用主函数进行评估
    evaluate_audio_from_path(audio_file_path_0)
    evaluate_audio_from_path(audio_file_path_1)
