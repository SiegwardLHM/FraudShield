import subprocess

import ffmpeg

input_file = 'D:/Dp/demo2-backend/users/video_fake/test/fake1.mp4'
output_file = 'D:/Dp/demo2-backend/users/video_fake/test/fake1-1.mp4'

command = [
    'ffmpeg',
    '-i', input_file,
    '-an',  # 去除音频
    '-vcodec', 'copy',  # 复制视频流
    output_file
]

try:
    subprocess.run(command, check=True)
    print(f"Video without audio has been saved to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
