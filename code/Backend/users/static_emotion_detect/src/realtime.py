import os
import cv2
import torch
import argparse
from copy import deepcopy
from torchvision import transforms
from model import Model
from ibug.face_detection import RetinaFacePredictor
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='/wangrun/nobody/Ada-CM/dataset/RAF-DB/basic', help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='../model/resnet50_ft_weight.pkl', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=2, help='the number of the device')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
args = parser.parse_args()

# 加载模型
print("Model is loading")
model = Model(args)
checkpoint = torch.load("checkpoint/epoch32_acc0.9041720628738403.pth")
model.load_state_dict(checkpoint)

device = torch.device('cuda:{}'.format(args.gpu))
model.to(device)
model.eval()  # 切换到评估模式

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Start Capture")
# 测试单独加载图片
# image_dir = "/wangrun/nobody/POSTER/data/train_class/class002"

# for file in os.listdir(image_dir):
#     frame = cv2.imread(os.path.join(image_dir, file))
    
#     # 检查读取的图像
#     if frame is None:
#         print(f"Failed to read {file}")
#         continue

#     face = deepcopy(frame)
    
#     # 检查图像形状
#     print(f"Original frame shape: {frame.shape}")
    
#     # 转换为 RGB 格式
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # 应用预处理
#     frame = eval_transforms(frame)
#     frame = frame.unsqueeze(0)
#     frame = frame.to(device)
    
#     # 打印预处理后的图像张量
#     print(f"Transformed frame shape: {frame.shape}")

#     # 推理
#     with torch.no_grad():
#         outputs, _ = model(frame)
#         print(f"Model outputs: {outputs}")

#     _, predicts = torch.max(outputs, 1)
#     print(f"Predicted class for {file}: {predicts.item()}")
labels = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
cap = cv2.VideoCapture(0)
detector = RetinaFacePredictor(threshold=0.8, device=device, model=RetinaFacePredictor.get_model('resnet50'))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 检测面部
    faces = detector(frame, rgb=False)
    for face in faces:
        x1, y1, x2, y2 = face[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_img = frame[y1:y2, x1:x2]

        # 转换为 RGB 格式并进行预处理
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_rgb = eval_transforms(face_rgb)
        face_rgb = face_rgb.unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            outputs, _ = model(face_rgb)
        _, predicts = torch.max(outputs, 1)
        predicted_class = predicts.item()
        print(predicted_class)
        # 在帧上显示预测结果
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'Emotion: {labels[predicted_class]}', (x1, y1), font, 1.0, (255, 255, 255), 1)

    #     if count % 10 == 0:
    #         cv2.imwrite(f"./frame_{count}.jpg", frame)
    #     count += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    