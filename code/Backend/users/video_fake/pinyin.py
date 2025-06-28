import subprocess

video_file = 'D:/Dp/demo2-backend/users/video_fake/test/version4.mp4'
audio_file = 'D:/Dp/demo2-backend/users/video_fake/test/audio-recording.wav'
output_file = 'D:/Dp/demo2-backend/users/video_fake/test/pin1.mp4'

command = [
    'ffmpeg',
    '-i', video_file,
    '-i', audio_file,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-strict', 'experimental',
    output_file
]

try:
    subprocess.run(command, check=True)
    print(f"Video and audio have been merged into {output_file} successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
