import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm
import cv2
from collections import deque
from PIL import Image
import face_alignment
from .data.transforms import NormalizeVideo, ToTensorVideo
from .data.samplers import ConsecutiveClipSampler
from .models.spatiotemporal_net import get_model
from .preprocessing.utils import warp_img, apply_transform, cut_patch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class config:
    checkpoint = "users/video_fake/models/weights/lipforensics_ff.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_per_clip = 25
    batch_size = 1
    num_workers = 0
    transform = Compose(
        [CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    mean_face_landmarks = np.load("users/video_fake/preprocessing/20words_mean_face.npy")
    stable_points = [33, 36, 39, 42, 45]
    std_size = (256, 256)
    crop_width = 96
    crop_height = 96
    start_idx = 48
    stop_idx = 68
    window_margin = 12

class CustomVideoDataset(Dataset):
    def __init__(self, video_path, frames_per_clip, transform):
        print("Loading video clips...")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
        self.clips = self.load_video_clips(video_path, frames_per_clip, transform)
        self.clips_per_video = [len(self.clips) // frames_per_clip]
        self.video_path = video_path
        self.frames_per_clip = frames_per_clip
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        return clip

    def load_video_clips(self, video_path, frames_per_clip, transform):
        cap = cv2.VideoCapture(video_path)
        frames = []
        q_frames, q_landmarks, q_name = deque(), deque(), deque()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.fa.get_landmarks(frame_rgb)

            if landmarks:
                landmarks = landmarks[0]
                q_frames.append(frame_gray)
                q_landmarks.append(landmarks)

                if len(q_frames) == config.window_margin:
                    smoothed_landmarks = np.mean(q_landmarks, axis=0)

                    cur_landmarks = q_landmarks.popleft()
                    cur_frame = q_frames.popleft()

                    trans_frame, trans = warp_img(
                        smoothed_landmarks[config.stable_points, :],
                        config.mean_face_landmarks[config.stable_points, :],
                        cur_frame, config.std_size
                    )

                    trans_landmarks = trans(cur_landmarks)

                    cropped_frame = cut_patch(
                        trans_frame,
                        trans_landmarks[config.start_idx: config.stop_idx],
                        config.crop_height // 2,
                        config.crop_width // 2,
                    )

                    frames.append(cropped_frame)

        cap.release()
        while q_frames:
            cur_frame = q_frames.popleft()
            cur_landmarks = q_landmarks.popleft()
            trans_frame = apply_transform(trans, cur_frame, config.std_size)
            trans_landmarks = trans(cur_landmarks)
            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[config.start_idx: config.stop_idx],
                config.crop_height // 2,
                config.crop_width // 2
            )
            frames.append(cropped_frame)
        clips = []
        for i in tqdm(range(0, len(frames), frames_per_clip)):
            clip = frames[i: i + frames_per_clip]
            if len(clip) < frames_per_clip:
                continue
            clip = np.array(clip).astype(np.float32) / 255
            clip = torch.from_numpy(clip).unsqueeze(0)
            clip = transform(clip)
            clips.append(clip)

        return clips

# 加载模型，只加载一次
model = get_model(config.checkpoint, device=config.device)
print("Loading model video_fake")
model.eval()

def detect_fake_video(video_path):
    dataset = CustomVideoDataset(video_path, config.frames_per_clip, config.transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    video_to_logits = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(config.device)
            logits = model(batch, lengths=[config.frames_per_clip] * batch.size(0))
            video_to_logits.extend(torch.sigmoid(logits))
        preds = []

        for logit in video_to_logits:
            pred = np.where(np.array(logit.cpu()) >= 0.3, 1, 0)
            preds.append(pred.item())

        all = preds.count(0) + preds.count(1)   # ADD ☆*: .｡. o(≧▽≦)o .｡.:*☆
        if max(preds, key=preds.count):
            video_fake_confidence = max(video_to_logits)[0]
            return "视频疑似伪造", video_fake_confidence.tolist()
        else:
            video_fake_confidence = preds.count(0)/ all
            return "视频为真", video_fake_confidence.tolist()

if __name__ == "__main__":
    # 示例视频路径
    video_path = "./test/5.mp4"
    detect_fake_video(video_path)
