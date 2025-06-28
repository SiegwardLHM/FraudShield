import cv2
import os
import shutil

import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from im_backend import settings
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载YOLOv8模型（假设你已经有训练好的模型）
model_path = os.path.join(current_dir, 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = YOLO(model_path, verbose=False)

# 定义异常动作类别和计数器
class_labels = ['斜视', '摸头', '摸鼻子', '摸脖子']
abnormal_action_counts = {label: 0 for label in class_labels}
abnormal_actions_with_timestamps = []

# 加载中文字体
font_path = os.path.join(current_dir, 'SIMHEI.TTF')
font = ImageFont.truetype(font_path, 24)

# 定义异常动作检测逻辑
def is_abnormal_action(detections):
    for detection in detections:
        cls = int(detection[5])  # 类别索引，假设类别索引是最后一个元素
        # print(f'Detected class: {cls}')  # 打印类别索引
        if cls in [0, 1, 2, 3]:  # 如果检测到的类别在异常动作类别中
            abnormal_action_counts[class_labels[cls]] += 1  # 更新计数器
            return True, cls
    return False, None

def draw_text_on_image(image, text, position, font, color=(0, 255, 0)):
    """
    在图像上绘制中文文本
    :param image: 要绘制文本的图像
    :param text: 要绘制的文本
    :param position: 文本的位置
    :param font: 字体
    :param color: 文本颜色，默认为绿色
    :return: 带有文本的图像
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def video_posture(file_path):
    # 定义保存图像的路径
    output_folder = os.path.join(settings.MEDIA_ROOT, 'result')

    # 清空输出目录
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(file_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化帧计数器
    frame_count = 0
    last_saved_second = -1  # 上次保存图像的秒数

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLO模型进行检测
        results = model(frame)

        # 检测结果
        detections = results[0].boxes.data.cpu().numpy()  # 提取检测结果

        # 检测是否存在异常动作
        is_abnormal, cls = is_abnormal_action(detections)
        if is_abnormal:
            # 计算当前帧对应的时间（秒）
            time_in_seconds = frame_count / fps
            current_second = int(time_in_seconds)

            # 检查当前秒是否已经保存过图像
            if current_second != last_saved_second:
                last_saved_second = current_second

                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)

                # 在图像上绘制时间戳和异常动作类别
                timestamp = f'{minutes:02}:{seconds:02}'
                label = class_labels[cls]
                text = f'{timestamp} - {label}'
                frame = draw_text_on_image(frame, text, (10, 30), font)

                # 保存当前帧为图像文件
                image_name = f'{label}_frame_{frame_count}.jpg'
                print(image_name)
                output_path = os.path.join(output_folder, image_name)

                cv2.imwrite(output_path, frame)

                # 记录异常动作和时间戳
                abnormal_actions_with_timestamps.append({
                    'timestamp': current_second,
                    'label': label,
                    'image_path': f'/media/result/{image_name}'  # 使用相对路径生成URL
                })

        frame_count += 1

    cap.release()

    # 导出每个异常动作的数量到文件
    with open(os.path.join(output_folder, 'abnormal_action_counts.txt'), 'w') as f:
        f.write("Abnormal action counts:\n")
        for label, count in abnormal_action_counts.items():
            f.write(f'{label}: {count}\n')

    # 返回结果字典，包括异常动作和时间戳
    return abnormal_action_counts, abnormal_actions_with_timestamps


