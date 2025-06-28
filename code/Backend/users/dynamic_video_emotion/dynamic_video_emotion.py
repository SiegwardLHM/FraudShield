import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from .models.Generate_Model import GenerateModel
from .dataloader.video_transform import *
import matplotlib
matplotlib.use('Agg')
import numpy as np
from .models.clip import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from .models.Text import *
import torchvision
from PIL import Image
import cv2

def load_models():
    global model, CLIP_model, class_names, class_names_with_context, class_descriptor
    args = get_args()
    if args.dataset == "FERV39K" or args.dataset == "DFEW":
        number_class = 7
        class_names = class_names_7
        class_names_with_context = class_names_with_context_7
        class_descriptor = class_descriptor_7
    else:
        number_class = 11
        class_names = class_names_11
        class_names_with_context = class_names_with_context_11
        class_descriptor = class_descriptor_11

    # create model and load pre_trained parameters
    CLIP_model, _ = clip.load("ViT-B/32", device='cpu')

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    else:
        input_text = class_descriptor

    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFEW-set1-model.pth")
    state_dict = torch.load(model_dir, map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="DFEW")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
    parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=30)
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default="end")
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--load-and-tune-prompt-learner', type=str, default='False')
    parser.add_argument('--text-type', type=str, default="class_descriptor")
    parser.add_argument('--exper-name', type=str, default="test")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temporal-layers', type=int, default=1)
    args = parser.parse_args([])
    return args

def data_process(video):
    # 处理视频
    # 创建一个视频捕获对象
    cap = cv2.VideoCapture(video)
    # 创建一个预处理管道
    transform = torchvision.transforms.Compose([GroupResize(224),
                                                Stack(),
                                                ToTorchFormatTensor()
                                                ])
    frame_transform = transform
    full_frames = []
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        # 将帧从BGR格式转换为RGB格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将帧转换为RGB格式的PIL图像
        frame = Image.fromarray(frame)
        # frame.show()
        full_frames.append(frame)

    full_num_frames = len(full_frames)
    frames = []
    for i in range(16):
        frame = int(full_num_frames * i / 16)
        frames.append(full_frames[frame])
    frames = frame_transform(frames)
    frames = torch.reshape(
        frames, (-1, 3, 224, 224))
    # 添加一个批量维度
    frames = frames.unsqueeze(0)

    return frames

def dynamic_video_emotion(file_path):
    video = data_process(file_path)
    with torch.no_grad():
        output = model(video)
    index = np.argmax(output)
    emotions = ['开心', '伤心', '中性', '生气', '惊讶', '厌恶', '害怕']
    emo = emotions[index]
    return emo

load_models()

# Example usage
# file_path = './1.mp4'
# emotion = dynamic_video_emotion(file_path)
# print("Predicted Emotion:", emotion)
