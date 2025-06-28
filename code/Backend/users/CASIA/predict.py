import os
import random
import numpy as np  # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

# 模型文件和config的绝对路径
# model_name_or_path = "./CASIA"
model_name_or_path = os.path.dirname(os.path.abspath(__file__))
print(model_name_or_path)

# 音频片段时长和采样率。
duration = 6
sample_rate = 16000

# 配置文件路径
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path,)


def id2class(id):
    if id == 0:
        return "生气"
    elif id == 1:
        return "害怕"
    elif id == 2:
        return "开心"
    elif id == 3:
        return "平静"
    elif id == 4:
        return "悲伤"
    else:
        return "惊讶"

class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
model = HubertForSpeechClassification.from_pretrained( model_name_or_path, config=config, )
model.eval()

def wav_emotion_predict(path):
    speech, sr = librosa.load(path=path, sr=sample_rate)
    speech = processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logit = model(speech)
    score = F.softmax(logit, dim=1).detach().cpu().numpy()[0]
    audio_emotion_confidence = np.max(score)    # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
    id = torch.argmax(logit).cpu().numpy()
    # print(f"file path: {path} \t predict: {id2class(id)} \t score:{score[id]} ")
    # return path, id2class(id), score[id]
    return id2class(id), audio_emotion_confidence


