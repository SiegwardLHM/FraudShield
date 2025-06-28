# emotion_analysis.py

import os
from ultralytics import YOLO
import cv2
from .ibug.face_detection import RetinaFacePredictor
from torchvision import transforms
from .src.model import Model
import torch


class StaticEmotionArgs():
    def __init__(self):
        self.checkpoint = 'users/static_emotion_detect/src/checkpoint/epoch32_acc0.9041720628738403.pth'
        self.resnet50_path = 'users/static_emotion_detect/src/model/resnet50_ft_weight.pkl'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

emotion_method = 'yolo'  # 暂时先写死，后续需要前端传入选择的方法
yolo = YOLO(model='users/static_emotion_detect/runs/classify/train3/weights/best.pt')
args = StaticEmotionArgs()
EAC_model = Model(args)
checkpoint = torch.load(args.checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EAC_model.load_state_dict(checkpoint)
EAC_model.to(device)
EAC_model.eval()
labels = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
labels_cn = ["惊讶", "害怕", "厌恶", "快乐", "悲伤", "生气", "中立"]
eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_detector = RetinaFacePredictor(threshold=0.8, device='cuda:1',
                model=RetinaFacePredictor.get_model('resnet50'))
unexpected_labels = ["fear", "disgust", "sad", "angry", "sadness"]
unexpected_labels_cn = ["害怕", "厌恶", "悲伤", "愤怒", "悲伤"]


def emotion_video_analysis(file_url):
    cap = cv2.VideoCapture(file_url)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    res, emotions = [],[]
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = face_detector(frame, rgb=False)
        for face in faces:
            left, top, right, bottom = face[:4]
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            face_img = frame[top:bottom, left:right]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            if emotion_method == 'yolo':
                emo = yolo(face_img, verbose=False)[0]
                res.append(emo.names[emo.probs.top1])
            elif emotion_method == 'EAC':
                face_img = eval_transforms(face_img)
                face_img = face_img.unsqueeze(0).to(device)
                with torch.no_grad():
                    output, hm = EAC_model(face_img)
                _, predicted = torch.max(output, 1)
                res.append(labels[predicted.item()])
        cnt += 1
        video_emotion_confidence = -1
        if cnt % fps == 0:
            most_common = max(res, key=res.count)

            # ADD ☆*: .｡. o(≧▽≦)o .｡.:*☆
            most_common_count = res.count(most_common)  # 该标签的出现次数
            total_count = len(res)  # 所有标签的总数
            proportion = most_common_count / total_count
            if proportion > video_emotion_confidence:
                video_emotion_confidence = proportion
            # ADD ☆*: .｡. o(≧▽≦)o .｡.:*☆

            most_common = most_common.lower()
            if most_common in unexpected_labels:
                cn_label = unexpected_labels_cn[unexpected_labels.index(most_common)]
                emotions.append({cnt // fps: cn_label})
            res = []
    cap.release()
    print(emotions)
    return emotions, video_emotion_confidence



def emotion_image_analysis(file_url):
    face_img = cv2.imread(file_url)
    faces = face_detector(face_img, rgb=False)
    if len(faces) == 0:
        return "未检测到人脸"
    for face in faces:
        left, top, right, bottom = face[:4]
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        face_img = face_img[top:bottom, left:right]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        if emotion_method == 'yolo':
            results = yolo(face_img, verbose=False)[0]
            image_emotion_confidence = results.probs[results.probs.top1]    # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
            return results.names[results.probs.top1], image_emotion_confidence
        elif emotion_method == 'EAC':
            face_img = eval_transforms(face_img)
            face_img = face_img.unsqueeze(0).to(device)
            with torch.no_grad():
                output, hm = EAC_model(face_img)
            
            max_prob, predicted = torch.max(output, 1)
            # _, predicted = torch.max(output, 1)

            en_label = labels[predicted.item()]

            image_emotion_confidence = max_prob.item()    # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
            return labels_cn[labels.index(en_label)], image_emotion_confidence
