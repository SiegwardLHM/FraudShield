import os
import requests
import datetime
import subprocess
import threading
import queue
from pydub import AudioSegment
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ..models import RealtimeAnalysisHistory
from ..audio_fake.utils_audio import evaluate_model
from ..text.text_model_realtime import analyze_text_file
from ..audio_transfer.robot_transfer import transfer_audio
from ..dynamic_video_emotion.dynamic_video_emotion import dynamic_video_emotion
from ..video_posture.locate import video_posture
from ..video_fake.realtime import detect_fake_video

User = get_user_model()


class RealTimeVideoAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Unauthorized"}, status=401)

        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file provided"}, status=400)

        user_id = request.user.id
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        session_folder = f'user_{user_id}/video_sessions/{current_time}'
        session_directory = os.path.join(settings.MEDIA_ROOT, session_folder)
        if not os.path.exists(session_directory):
            os.makedirs(session_directory)

        webm_file_name = file.name
        webm_file_path = os.path.join(session_directory, webm_file_name)
        with open(webm_file_path, 'wb') as webm_file:
            webm_file.write(file.read())

        mp4_file_name = f'{os.path.splitext(webm_file_name)[0]}.mp4'
        mp4_file_path = os.path.join(session_directory, mp4_file_name)
        try:
            print(f"开始转换 {webm_file_path} 到 {mp4_file_path}")
            subprocess.run(['ffmpeg', '-i', webm_file_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
                            mp4_file_path], check=True)
            print("转换成功")
        except subprocess.CalledProcessError as e:
            return JsonResponse({"error": f"Error converting video: {str(e)}"}, status=500)

        result, text_class = self.analyze(mp4_file_path, session_directory)

        formatted_result_warn = self.format_warning_message(result, text_class)
        result = formatted_result_warn + result

        print("final")
        print(result)

        # 确保 result['images'] 是一个列表
        screenshot_url = None
        if 'images' in result and isinstance(result['images'], list) and len(result['images']) > 0:
            screenshot_url = request.build_absolute_uri(result['images'][0])
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')

        RealtimeAnalysisHistory.objects.create(
            user=request.user,
            result=result,
            screenshot=screenshot_url
        )

        return JsonResponse({
            "result": result,
            "timestamp": timestamp,
            "screenshot": screenshot_url
        }, status=200)

    def analyze(self, file_path, session_directory):
        result,text_class = analyze_file(file_path, session_directory)
        print(result)
        print(text_class)
        return result, text_class

    def format_warning_message(self, result, text_class):
        def is_fraudulent(fraud_result):
            return (
                    isinstance(fraud_result, dict) and
                    'fake' in fraud_result and
                    'real' in fraud_result and
                    fraud_result['fake'] >= fraud_result['real']
            )

        is_fraud_detected = False
        if 'fraud' in result and isinstance(result['fraud'], dict):
            is_fraud_detected = any(is_fraudulent(fraud_res) for fraud_res in result['fraud'].values())

        if ('video_fake' in result and result['video_fake'] == '视频疑似伪造') or \
                is_fraud_detected or \
                (text_class != '') or \
                ('dynamic_emotion' in result and result['dynamic_emotion']) or \
                ('posture' in result and result['posture']):

            if ('video_fake' in result and result['video_fake'] == '视频疑似伪造') or \
                    is_fraud_detected or \
                    ((text_class != '不是诈骗') and (text_class != '垃圾信息')):
                warning_message = "**<span style='color: red;'>疑似诈骗！</span>**\n"
                warning_message += "\n 概要：\n"
            else:
                warning_message = "**<span style='color: green;'>不是诈骗！</span>**\n"
                warning_message += "\n 概要：\n"

            if 'video_fake' in result:
                if result['video_fake'] == '视频疑似伪造':
                    warning_message += "疑似存在视频伪造，"
                else:
                    warning_message += "视频为真，"

            if 'fraud' in result:
                if is_fraud_detected:
                    warning_message += "语音疑似伪造，"
                else:
                    warning_message += "音频为真，"

            if text_class != '不是诈骗':
                warning_message += f"通话内容存在<span style='color:blue;'>{text_class}</span>"
            else:
                warning_message += "通话内容不存在引导诈骗"

            if 'dynamic_emotion' in result and result['dynamic_emotion'] or \
                    'posture' in result and result['posture']:
                warning_message += "，对方疑似存在异常情绪"
            else:
                warning_message += "，对方未显露异常情绪"

            warning_message += "\n"
        else:
            warning_message = "未检测到诈骗行为"

        return warning_message


def analyze_file(file_path, baseurl):
    local_file_path = file_path
    result = {}

    wav_path = convert_mp4_to_wav(local_file_path)
    result['video_fake'] = detect_fake_video(local_file_path)
    result['dynamic_emotion'] = []
    result['images'] = []
    result['posture'], abnormal_actions_with_timestamps = video_posture(local_file_path)
    for action in abnormal_actions_with_timestamps:
        image_path = os.path.join(settings.MEDIA_ROOT, 'result', action['image_path'])
        if 'images' not in result:
            result['images'] = []
        image_url = "http://localhost:8000/" + 'media/result/' + os.path.basename(image_path)
        result['images'].append(image_url)

    result['fraud'] = format_fraud_result(evaluate_model(wav_path))
    result['transfer'] = transfer_audio(wav_path)
    text_path = save_text_to_file(result['transfer'], wav_path)
    text_class, text_reason = analyze_text_file(text_path)
    result['text'] = f"分类：{text_class} \n 原因：{text_reason}\n"

    dynamic_emotion = dynamic_video_emotion(local_file_path)
    result['dynamic_emotion'].append({datetime.datetime.now().strftime('%H:%M:%S'): dynamic_emotion})

    print("result:")
    print(result)

    return format_result(result), text_class

def format_result(result):
    formatted_result = ""

    if 'video_fake' in result:
        formatted_result += (f"\n 视频是否伪造：{result['video_fake']} \n")

    if 'fraud' in result:
        if isinstance(result['fraud'], dict):
            fraud_entries = []
            if 'fake' in result['fraud'] and 'real' in result['fraud']:
                if result['fraud']['fake'] >= result['fraud']['real']:
                    formatted_result += f"\n疑似音频伪造, 可能性：{result['fraud']['fake']}"
                else:
                    formatted_result += f"\n音频是否伪造：音频为真"
            else:
                for time_point, fraud_res in result['fraud'].items():
                    if isinstance(fraud_res, dict) and 'fake' in fraud_res and 'real' in fraud_res:
                        if fraud_res['fake'] >= fraud_res['real']:
                            fraud_entries.append(f"\n时间点 {time_point}: 疑似音频伪造, 可能性：{fraud_res['fake']}")
                    elif isinstance(fraud_res, dict):
                        fraud_entries.append(f"\n{time_point}: {format_fraud_result(fraud_res)}")
                    else:
                        fraud_entries.append(f"\n{time_point}: {fraud_res}")

                if fraud_entries:
                    formatted_result += "\n音频伪造分析结果：" + ''.join(fraud_entries)
                else:
                    formatted_result += f"\n音频是否伪造：音频为真"
        else:
            if result['fraud']['fake'] >= result['fraud']['real']:
                formatted_result += f"\n疑似音频伪造, 可能性：{result['fraud']['fake']}"
            else:
                formatted_result += f"\n音频是否伪造：音频为真"

    if 'text' in result:
        formatted_result += f"\n\n文本分析结果: {result['text']}"

    if 'dynamic_emotion' in result:
        unique_dynamic_emotions = {}
        for res in result['dynamic_emotion']:
            for timestamp, emotion in res.items():
                if emotion in ['生气', '惊讶', '厌恶', '害怕']:
                    if timestamp not in unique_dynamic_emotions:
                        unique_dynamic_emotions[timestamp] = []
                    unique_dynamic_emotions[timestamp].append(emotion)
                if unique_dynamic_emotions:
                    formatted_result += "\n视频情感分析结果:"
                    for timestamp, emotions in unique_dynamic_emotions.items():
                        formatted_result += f"\n时间点 {timestamp}: {emotions}"

    # if 'transfer' in result:
    #     formatted_result += f"\n\n语音转换结果: {result['transfer']}\n"
    #     print(formatted_result)

    if 'posture' in result:
        formatted_result += "\n姿势检测结果:\n"
        for label, count in result['posture'].items():
            formatted_result += f'{label}: {count}\n'

    if 'images' in result:
        formatted_result += "\n\n异常动作检测图片:\n"
        for image_path in result['images']:
            formatted_result += f'<img src="{image_path}" style="max-width: 280px; max-height: 300px; width: auto; height: auto;">\n'

    if formatted_result =="":
        formatted_result = "没有发现异常或未知分析结果"

    return formatted_result

def convert_mp4_to_wav(mp4_path):
    import ffmpeg
    wav_path = os.path.splitext(mp4_path)[0] + '.wav'
    try:
        (
            ffmpeg
            .input(mp4_path)
            .output(wav_path)
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"音频已提取并保存到: {wav_path}")
    except ffmpeg.Error as e:
        print(f"提取音频失败: {e.stderr.decode('utf-8')}")
        raise
    return wav_path

def save_text_to_file(text, audio_file_path):
    text_file_path = os.path.splitext(audio_file_path)[0] + "_transcription.txt"
    with open(text_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    return text_file_path

def format_fraud_result(fraud_result):
    if isinstance(fraud_result, float):
        return {'fake': fraud_result, 'real': 1 - fraud_result}
    if isinstance(fraud_result, dict):
        if 'fake' in fraud_result and 'real' in fraud_result:
            return fraud_result
        formatted_fraud_result = {}
        for key, value in fraud_result.items():
            formatted_fraud_result[key] = format_fraud_result(value)
        return formatted_fraud_result
    return fraud_result
