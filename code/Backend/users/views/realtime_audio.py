import os
import json
import datetime
import logging
from django.conf import settings
from django.http import JsonResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from ..models import RealtimeAnalysisHistory
from ..audio_fake.utils_audio import evaluate_model
from ..audio_transfer.robot_transfer import transfer_audio
from ..text.text_model_realtime import analyze_text_file
from ..CASIA.predict import wav_emotion_predict

logger = logging.getLogger(__name__)


class RealTimeAudioAnalysisView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Unauthorized"}, status=401)

        file = request.FILES.get('file')
        timestamp = request.POST.get('timestamp')
        if not file:
            return JsonResponse({"error": "No file provided"}, status=400)

        # 创建对应目录
        user_id = request.user.id
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        session_folder = f'user_{user_id}/audio_sessions/{current_time}'
        session_directory = os.path.join(settings.MEDIA_ROOT, session_folder)
        os.makedirs(session_directory, exist_ok=True)

        # 保存文件
        wav_file_name = file.name
        wav_file_path = os.path.join(session_directory, wav_file_name)
        with open(wav_file_path, 'wb') as wav_file:
            wav_file.write(file.read())

        # 确保文件有效
        file_size = os.path.getsize(wav_file_path)
        logger.debug(f"Saved file size: {file_size} bytes")
        if file_size == 0:
            return JsonResponse({"error": "Uploaded file is empty"}, status=400)

        # 开始分析
        try:
            result, text_class = self.analyze(wav_file_path, session_directory)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return JsonResponse({"error": f"Error during analysis: {str(e)}"}, status=500)

        RealtimeAnalysisHistory.objects.create(
            user=request.user,
            result=result
        )

        # 格式化结果
        formatted_result = self.format_warning_message(result, text_class)
        if formatted_result != "":
            formatted_result += self.format_result(result)

        return JsonResponse({"result": formatted_result, "timestamp": timestamp}, status=200)

    def analyze(self, file_path, session_directory):
        result = {}

        # 转文字并分析
        result['transfer'] = transfer_audio(file_path)
        text_path = save_text_to_file(result['transfer'], file_path)
        text_class, text_reason = analyze_text_file(text_path)
        result['text'] = f"分类：{text_class} \n 原因：{text_reason}\n"

        # 音频情绪
        result['audio_emotion'] = filter_audio_emotion(wav_emotion_predict(file_path))

        # 伪造检测
        fraud_result = evaluate_model(file_path)
        result['fraud'] = format_fraud_result(fraud_result)
        print(result['fraud'])

        # 保存结果
        self.update_session_results(result, session_directory)

        return result, text_class

    def update_session_results(self, result, session_directory):
        results_file = os.path.join(session_directory, 'results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as file:
                session_results = json.load(file)
        else:
            session_results = []

        session_results.append(result)

        with open(results_file, 'w', encoding='utf-8') as file:
            json.dump(session_results, file, ensure_ascii=False, indent=4)

    def format_result(self, result):
        formatted_result = ""

        # 音频伪造
        if 'fraud' in result:
            if isinstance(result['fraud'], dict):
                if 'fake' in result['fraud'] and 'real' in result['fraud']:
                    if result['fraud']['fake'] >= result['fraud']['real']:
                        formatted_result += f"\n 疑似音频伪造, 可能性：{result['fraud']['fake']}\n"
                    else:
                        formatted_result += "\n 音频为真\n"
                else:
                    formatted_result += "音频伪造分析结果："
                    for time_point, fraud_res in result['fraud'].items():
                        if isinstance(fraud_res, dict) and 'fake' in fraud_res and 'real' in fraud_res:
                            if fraud_res['fake'] >= fraud_res['real']:
                                formatted_result += f"\n时间点 {time_point}: 疑似音频伪造, 可能性：{fraud_res['fake']}\n"
                            else:
                                formatted_result += f"\n时间点 {time_point}: 音频为真\n"
            else:
                if result['fraud']['fake'] >= result['fraud']['real']:
                    formatted_result += f"疑似音频伪造, 可能性：{result['fraud']['fake']}\n"
                else:
                    formatted_result += "音频为真\n"

        # 文本分析
        if 'text' in result:
            formatted_result += f"\n\n文本分析结果: {result['text']}\n"

        # 语音情绪
        if 'audio_emotion' in result:
            formatted_result += f"\n语音情绪分析结果: {result['audio_emotion']}\n"

        if not formatted_result:
            formatted_result = "没有发现异常或未知分析结果"

        return formatted_result

    def format_warning_message(self, result, text_class):
        if ('fraud' in result and any(
            fraud_res['fake'] >= fraud_res['real'] for fraud_res in result['fraud'].values() if
            isinstance(fraud_res, dict) and 'fake' in fraud_res and 'real' in fraud_res)) or \
            (text_class != '') or ('audio_emotion' in result and result['audio_emotion']):

            if ('fraud' in result and any(
            fraud_res['fake'] >= fraud_res['real'] for fraud_res in result['fraud'].values() if
            isinstance(fraud_res, dict) and 'fake' in fraud_res and 'real' in fraud_res)) or \
                    ((text_class != '不是诈骗') and (text_class != '垃圾信息')):
                warning_message = "**<span style='color: red;'>疑似诈骗！</span>**\n"
                warning_message += "\n 概要：\n"
            else:
                warning_message = "**<span style='color: green;'>不是诈骗！</span>**\n"
                warning_message += "\n 概要：\n"

            if 'fraud' in result:
                if 'fake' in result['fraud'] and 'real' in result['fraud']:
                    if result['fraud']['fake'] >= result['fraud']['real']:
                        warning_message += "语音疑似伪造，"
                    else:
                        warning_message += "音频为真，"
                else:
                    warning_message += ""
            else:
                warning_message += ""

            if 'text' in result:
                if text_class != '不是诈骗':
                    warning_message += f"通话内容存在<span style='color:blue;'>{text_class}</span>"
                else:
                    warning_message += "通话内容不存在引导诈骗"
            else:
                warning_message += ""

            if 'audio_emotion' in result and result['audio_emotion']:
                warning_message += "，对方疑似存在异常情绪"
            elif not any(key in result for key in ['audio_emotion']):
                warning_message += ""
            else:
                warning_message += "，对方未显露异常情绪"

            warning_message += "\n"
        else:
            warning_message = ""

        return warning_message


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

def filter_audio_emotion(audio_emotion):
    # 筛选出异常情绪
    abnormal_emotions = ['害怕', '惊讶', '生气', '悲伤']
    filtered_emotion = ''
    if audio_emotion in abnormal_emotions:
        filtered_emotion = audio_emotion
    return filtered_emotion