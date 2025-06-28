import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import yaml
from .model_audio import RawNet
from torch.nn import functional as F
import librosa
import json
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 加载模型配置
dir_yaml = 'model_config_RawNet.yaml'
dir_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_yaml)
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)

# 加载cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# 初始化模型
model = RawNet(parser1['model'], device)
model =(model).to(device)

# 加载模型权重
model_path = 'librifake_pretrained_lambda0.5_epoch_25.pth'  # 模型权重文件的路径
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

model.eval()

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_sample(sample_path, max_len = 96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    if sr != 24000:
        y = librosa.resample(y, orig_sr = sr, target_sr = 24000)
    if(len(y) <= 96000):
        return [Tensor(pad(y, max_len))]
    for i in range(int(len(y)/96000)):
        if (i+1) ==  range(int(len(y)/96000)):
            y_seg = y[i*96000 : ]
        else:
            y_seg = y[i*96000 : (i+1)*96000]
        y_pad = pad(y_seg, max_len)
        y_inp = Tensor(y_pad)
        y_list.append(y_inp)
    return y_list


def evaluate_model(input_path):
    out_list_multi = []
    out_list_binary = []
    for m_batch in load_sample(input_path):
        m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
        logits, multi_logits = model(m_batch)

        probs = F.softmax(logits, dim=-1)
        probs_multi = F.softmax(multi_logits, dim=-1)
        out_list_multi.append(probs_multi.tolist()[0])
        out_list_binary.append(probs.tolist()[0])

    result_multi = np.average(out_list_multi, axis=0).tolist()
    result_binary = np.average(out_list_binary, axis=0).tolist()

    # 格式化结果
    result = {
        'fake': result_binary[0],
        'real': result_binary[1]
    }

    return result

