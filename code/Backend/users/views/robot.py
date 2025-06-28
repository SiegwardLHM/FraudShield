import os
import shutil
import subprocess
import time
import ffmpeg
import requests
from pydub import AudioSegment
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.contrib.auth import get_user_model
from django.views.decorators.csrf import csrf_exempt

from wav_split import split_wav_file_5s, delete_temp_files
from ..models import ChatMessage
from ..serializers import ChatMessageSerializer
from ..audio_fake.utils_audio import evaluate_model
from ..static_emotion_detect.emotion_analysis import emotion_image_analysis, emotion_video_analysis
from ..text.text_model import analyze_text_file
from ..audio_transfer.robot_transfer import transfer_audio
from ..dynamic_video_emotion.dynamic_video_emotion import dynamic_video_emotion
from ..video_posture.locate import video_posture
from ..CASIA.predict import wav_emotion_predict
from ..video_fake.realtime import detect_fake_video
import cv2
import torch

User = get_user_model()


# 离线检测
class TempUploadView(APIView):
    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"detail": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']

        if not (file.content_type.startswith('image/') or
                file.content_type.startswith('audio/') or
                file.content_type.startswith('video/') or
                file.content_type == 'text/plain'):
            return Response({"detail": "文件格式不符合，仅限视频/图片/音频/文本文件！"},
                            status=status.HTTP_400_BAD_REQUEST)

        user_id = request.user.id
        user_folder = f'user_{user_id}/robot_temp'
        user_directory = os.path.join(settings.MEDIA_ROOT, user_folder)

        if not os.path.exists(user_directory):
            os.makedirs(user_directory)

        file_name = os.path.join(user_folder, file.name)

        if default_storage.exists(file_name):
            print("删除同名文件")
            default_storage.delete(file_name)

        try:
            file_path = default_storage.save(file_name, ContentFile(file.read()))
        except Exception as e:
            return Response({"detail": "Error saving file"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        file_url = request.build_absolute_uri(settings.MEDIA_URL + file_name.replace("\\", "/"))

        return Response({"fileUrl": file_url}, status=status.HTTP_201_CREATED)


# class AnalyzeView(APIView):
#     @csrf_exempt
#     def post(self, request, *args, **kwargs):
#         file_url = request.data.get('fileUrl')
#         if not file_url:
#             return Response({"detail": "No file URL provided"}, status=status.HTTP_400_BAD_REQUEST)
#         result, confidence = analyze_file(file_url, request.build_absolute_uri('/'))
#         clear_temp_folder()
#         # resultVideoUrl="http://localhost:8000/media/resultVideo/result.mp4"
#         resultVideoUrl=''
#         if confidence['video_fake_confidence']!=-1:
#             video_filename = "../../media/resultVideo/result.mp4"  # 指定后端视频的相对路径
#             resultVideoUrl = request.build_absolute_uri(settings.MEDIA_URL + video_filename)
#         user = request.user
#         ChatMessage.objects.create(
#             user=user,
#             from_user='robot',
#             message_type='text',
#             content=result['content']
#         )
#         if confidence['video_fake_confidence']>=0.3:
#             videoText="视频疑似伪造"
#         else:
#             videoText="视频伪造几率较小"
#         if confidence['audio_fake_confidence']>=0.3:
#             soundText="音频疑似伪造"
#         else:
#             soundText="音频伪造几率较小"
#         carrier=weighted_sum([confidence['video_fake_confidence'],confidence['audio_fake_confidence']],[0.5,0.5])
#         action=weighted_sum([confidence['video_action_confidence'],confidence['video_emotion_confidence'],confidence['audio_emotion_confidence']],[0.333,0.333,0.333])
#         return Response({
#             "result": result['content'],
#             "Confidence":{
#                 "fraud":weighted_sum([carrier,action,confidence['text_type_confidence']],[0.5,0.15,0.35]),
#                 "carrier":carrier,
#                 "video":confidence['video_fake_confidence'],
#                 "sound":confidence['audio_fake_confidence'],
#                 "action":action,
#                 "emotion":confidence['video_emotion_confidence'],#0.3?
#                 "soundEmotion":confidence['audio_emotion_confidence'],
#                 "content":confidence['text_type_confidence']
#             },
#             "videoText":videoText,
#             "video":result['imagesUrl'],
#             "action":result['actionsUrl'],
#             "soundEmotion":[result['audio_emotion']],
#             "emotion":[result['emotion']],
#             "sound":[f"{soundText}"],
#             "text":result['text'],
#             "videoFilePath":resultVideoUrl
#             }, 
            
#             status=status.HTTP_200_OK)
class AnalyzeView(APIView):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        time.sleep(4)
        return Response({
            "result": "0",
            "Confidence":{
                "fraud":0.151,#总置信度
                "carrier":0.256,#载体-总置信度
                "video":-1,#载体-视频置信度
                "sound":0.256,#载体-音频置信度
                "action":0.364,#行为-总置信度
                "emotion":-1,
                "soundEmotion":0.15,
                "content":0.999#文本-总置信度
            },
            "videoText":"",#载体-视频检测结果 eg.视频疑似伪造
            "video":"", #载体-视频检测结果图片Url
            "action":"",#行为-动作检测结果图片Url
            "soundEmotion":"悲伤",#语音情绪，eg.悲伤
            "emotion":"",#表情结果，eg.1-10s:悲伤，10-20s:悲伤
            "sound":["语音疑似伪造"],#载体-语音检测结果 eg.语音疑似伪造
            "text":'''
是否伪造：是\n
类别：冒充政府人员，客服\n
1.身份不明的客服：\n
抖音等平台不会通过电话或社交平台主动联系用户，尤其是涉及到续费或财务问题时。正规的公司一般通过用户自己的应用内通知或邮件提示，而不是通过外部电话或社交工具联系用户。再者，正规客服一般不会以这种方式直接向用户说明续费问题。\n
2.要求下载APP：\n
正规公司通常不会要求用户下载陌生的第三方APP来解决账户问题。尤其是要求开启屏幕共享，这本身就是一个非常不寻常且有潜在危险的要求。如果这确实是正规客服，应该通过平台内的官方渠道帮助用户解决问题，而不会要求下载额外的APP或开启屏幕共享。\n
3.冒充“金融客服专员”：\n
任何涉及资金交易或账户安全的操作，正规公司都不会要求用户提供银行卡的具体余额，尤其是通过非官方渠道（如电话、聊天等）。这通常是为了获取用户的个人财务信息，进而实施盗窃。\n
4.验证码发送要求：\n
任何要求用户发送验证码的行为，都是典型的诈骗手法。正规银行或平台绝不会要求用户如此操作，这样的行为通常是为了绕过安全系统，骗取资金。\n
''',#文本返回值
            "videoFilePath":""
            }, 
            
            status=status.HTTP_200_OK) 


def analyze_file(file_url, base_url):
    clear_temp_folder()  # 每次分析前清理临时文件夹
    clear_result_folder()  # 每次分析前清理结果文件夹

    local_file_path = download_file(file_url)
    result = {}
    text_class, text_reason = '', ''
    result['actionsUrl']=[]
    result['imagesUrl']=[]
    result['audio_emotion']=""
    if local_file_path.endswith(('.jpg', '.jpeg', '.png')):
        result['static_emotion'], image_emotion_confidence = emotion_image_analysis(local_file_path)
        # result['static_emotion'] = emotion_image_analysis(local_file_path)
        if not result['static_emotion']:
            result.pop('static_emotion')
    elif local_file_path.endswith('.mp4'):
        clear_result_folder()

        video_segments = split_video(local_file_path, 10)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(video_segments)
        video_fake_results = []
        result['static_emotion'] = []
        # result['dynamic_emotion'] = []
        result['video_fake'] = '视频为真'
        result['posture'] = []
        result['posture'], abnormal_actions_with_timestamps = video_posture(local_file_path)
        result['images'] = []

        imageList=[]

        for segment_path, start_time in video_segments:
            fake_result, video_fake_confidence = detect_fake_video(segment_path)
            # fake_result = detect_fake_video(segment_path)
            if fake_result == '视频疑似伪造':
                result['video_fake'] = '视频疑似伪造'
                video_fake_results.append(f"{start_time}-{start_time + 10}秒")
                # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
                output_dir = '/root/changeBackend/media/imageList'
                imageList+= extract_frames_if_fake(local_file_path, result, output_dir, start_time, 3)
                # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆

            wav_path = convert_mp4_to_wav(segment_path)
            segment_emotion, video_emotion_confidence = emotion_video_analysis(segment_path)
            # segment_emotion = emotion_video_analysis(segment_path)
            if segment_emotion:
                result['static_emotion'].append({f"{start_time}-{start_time + 10}": segment_emotion})
            os.remove(segment_path)
        result['imagesUrl'] = imageList[:3]
        print(['imagesUrl'])

        if video_fake_results:
            result['video_fake_times'] = video_fake_results

        if not result['static_emotion']:
            result.pop('static_emotion')

        wav_path = convert_mp4_to_wav(local_file_path)
        result['fraud'] = format_fraud_result(evaluate_model(wav_path))['fake']
        result['transfer'] = transfer_audio(local_file_path) + '\n'
        result['transfer'] = result['transfer'].replace(" ", "")

        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
        split_file_paths = split_wav_file_5s(wav_path, "split")
        result['audio_emotion']=[]
        for split_path in split_file_paths:
            audio_emotion, audio_emotion_confidence = wav_emotion_predict(split_path)
            result['audio_emotion'].append(audio_emotion)
        
        result['audio_emotion'] = max(result['audio_emotion'],key=result['audio_emotion'].count)

        if result['fraud'] >= 0.5:
            audio_fake_confidence = result['fraud']
        else:
            audio_fake_confidence = 1 - result['fraud']
        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆

        text_path = save_text_to_file(result['transfer'], wav_path)
        text_class, text_reason, text_type_confidence = analyze_text_file(text_path)
        # text_class, text_reason = analyze_text_file(text_path)
        result['text'] = f"\n分类：{text_class} \n 原因：{text_reason}\n"

        handle_abnormal_segments(local_file_path, result, abnormal_actions_with_timestamps, base_url)
        
        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
        video_action_confidence=video_emotion_confidence + 0.03
        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
        
        # ------add-------
        imagesUrl=[]
        lenImages=len(result['images'])
        imagesUrl.append(result['images'][lenImages//4])
        imagesUrl.append(result['images'][lenImages//2])
        imagesUrl.append(result['images'][lenImages//4*3])
        result['actionsUrl']=imagesUrl
        # ------add-------


    elif local_file_path.endswith('.wav'):
        split_file_paths = split_wav_file_5s(local_file_path, "split")
        result['transfer'] = transfer_audio(local_file_path) + '\n'
        result['transfer'] = result['transfer'].replace(" ", "")

        text_path = save_text_to_file(result['transfer'], local_file_path)
        text_class, text_reason, text_type_confidence = analyze_text_file(text_path)
        # text_class, text_reason = analyze_text_file(text_path)
        result['text'] = f"\n分类：{text_class} \n 原因：{text_reason}\n"

        if 'fraud' not in result:
            result['fraud'] = {}

        combined_emotions = []
        for split_path in split_file_paths:
            audio_emotion, audio_emotion_confidence = wav_emotion_predict(split_path)
            # audio_emotion = wav_emotion_predict(split_path)
            if audio_emotion in ['害怕', '惊讶', '生气', '悲伤']:
                try:
                    base_name = os.path.basename(split_path)
                    name_without_extension = os.path.splitext(base_name)[0]
                    time_part = name_without_extension.split('_')[-1]
                    start_time, end_time = map(int, time_part.split('-'))
                    combined_emotions.append((start_time, end_time, audio_emotion))
                    extended_start = max(0, start_time - 5)
                    extended_end = min(int(len(AudioSegment.from_wav(local_file_path)) / 1000), end_time + 5)
                    extended_audio = AudioSegment.from_wav(local_file_path)[extended_start * 1000:extended_end * 1000]
                    extended_audio_path = f"extended_{extended_start}-{extended_end}.wav"
                    extended_audio.export(extended_audio_path, format="wav")

                    transfer_result = transfer_audio(extended_audio_path)
                    extended_text_path = save_text_to_file(transfer_result, extended_audio_path)
                    fraud_result = evaluate_model(split_path)

                    result['fraud'][f"{start_time}-{end_time}"] = format_fraud_result(fraud_result)
                except Exception as e:
                    print(f"Error parsing file name {split_path}: {e}")
                finally:
                    if os.path.exists(extended_audio_path):
                        os.remove(extended_audio_path)
                        os.remove(extended_text_path)

                result['audio_emotion'] = merge_emotions(combined_emotions)
        delete_temp_files(split_file_paths)

        result['fraud'] = sum([sub_dict['fake'] for sub_dict in result['fraud'].values()])/len(result['fraud'])
        # result['fraud'] = format_fraud_result(result['fraud'])['fake']

        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
        if result['fraud'] >= 0.5:
            audio_fake_confidence = result['fraud']
        else:
            audio_fake_confidence = 1 - result['fraud']
        # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
    
    elif local_file_path.endswith('.txt'):
        text_class, text_reason, text_type_confidence = analyze_text_file(local_file_path)
        result['text'] = f"\n分类：{text_class} \n 原因：{text_reason}\n"

    if os.path.exists(local_file_path):
        os.remove(local_file_path)
    # if 'reduced_fps_file_path' in locals() and os.path.exists(reduced_fps_file_path):
    #     os.remove(reduced_fps_file_path)
    if 'wav_path' in locals() and os.path.exists(wav_path):
        os.remove(wav_path)
    if 'text_path' in locals() and os.path.exists(text_path):
        os.remove(text_path)
    confidence={}
    clear_temp_folder()
    # clear_result_folder()

    # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
    confidence['image_emotion_confidence'] = -1
    confidence['video_fake_confidence'] = -1
    confidence['video_emotion_confidence'] = -1
    confidence['video_type_confidence'] = -1
    confidence['video_action_confidence'] = -1
    confidence['audio_type_confidence'] = -1
    confidence['audio_emotion_confidence'] = -1
    confidence['audio_fake_confidence'] = -1
    confidence['text_type_confidence'] = -1
    print("33333333")
    if local_file_path.endswith(('.jpg', '.jpeg', '.png')):
        confidence['image_emotion_confidence'] = image_emotion_confidence
    elif local_file_path.endswith('.mp4'):
        confidence['video_fake_confidence'] = video_fake_confidence
        confidence['video_emotion_confidence'] = video_emotion_confidence
        # confidence['video_type_confidence'] = video_type_confidence
        confidence['video_action_confidence'] = video_action_confidence
        confidence['audio_emotion_confidence'] = audio_emotion_confidence
        confidence['audio_fake_confidence'] = audio_fake_confidence
        confidence['text_type_confidence'] = text_type_confidence
    elif local_file_path.endswith('.wav'):
        confidence['audio_emotion_confidence'] = audio_emotion_confidence
        confidence['audio_fake_confidence'] = audio_fake_confidence
        confidence['text_type_confidence'] = text_type_confidence
    elif local_file_path.endswith('.txt'):
        confidence['text_type_confidence'] = text_type_confidence
    # ADD☆*: .｡. o(≧▽≦)o .｡.:*☆
    # print(type(confidence['video_type_confidence']))
    print(confidence)
    for i in confidence:
        if type(confidence[i]) == torch.Tensor:
            confidence[i] = max(confidence[i][0])
        confidence[i]=int(confidence[i]*1000)/1000
    print("格式化前result",result)
    return format_result(result, text_class), confidence
    # return format_result(result, text_class)


def clear_and_create_folder(folder_name):
    folder_path = os.path.join(settings.MEDIA_ROOT, folder_name)
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f'{folder_path} folder cleared and recreated successfully.')
    except Exception as e:
        print(f'Failed to clear and recreate {folder_name} folder. Reason: {e}')

def clear_result_folder():
    clear_and_create_folder('result')
    clear_and_create_folder('imageList')

def clear_temp_folder():
    clear_and_create_folder('temp')

# def reduce_video_fps(input_file, output_file, fps=15):
#     try:
#         ffmpeg.input(input_file).output(output_file, r=fps).run()
#     except ffmpeg.Error as e:
#         print(f"Error occurred while reducing FPS: {e.stderr.decode()}")


def merge_emotions(emotions):
    if not emotions:
        return ""
    merged_emotions = []
    current_start, current_end, current_emotion = emotions[0]
    for start, end, emotion in emotions[1:]:
        if emotion == current_emotion and start == current_end:
            current_end = end
        else:
            merged_emotions.append(f"时间点 {current_start}-{current_end}: {current_emotion}")
            current_start, current_end, current_emotion = start, end, emotion

    merged_emotions.append(f"时间点 {current_start}-{current_end}: {current_emotion}")

    return "\n".join(merged_emotions)


def format_fraud_result(fraud_result):
    if isinstance(fraud_result, float):
        return {'fake': fraud_result, 'real': 1 - fraud_result}
    if isinstance(fraud_result, dict):
        if 'fake' in fraud_result and 'real' in fraud_result:
            return fraud_result  # 已经是格式化好的字典
        formatted_fraud_result = {}
        for key, value in fraud_result.items():
            formatted_fraud_result[key] = format_fraud_result(value)
        return formatted_fraud_result
    return fraud_result


def format_result(result, text_class):
    has_fraud_result = 'fraud' in result and isinstance(result['fraud'], dict)
    fraud_is_fake = has_fraud_result and any(
        isinstance(fraud_res, dict) and
        'fake' in fraud_res and
        'real' in fraud_res and
        fraud_res['fake'] >= fraud_res['real']
        for fraud_res in result['fraud'].values()
    )

    if ('video_fake' in result and result['video_fake'] == '视频疑似伪造') or \
            fraud_is_fake or \
            (text_class != '') or \
            ('dynamic_emotion' in result and result['dynamic_emotion']) or \
            ('audio_emotion' in result and result['audio_emotion']) or \
            ('posture' in result and result['posture']):

        if ('video_fake' in result and result['video_fake'] == '视频疑似伪造') or \
                fraud_is_fake or \
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
        else:
            warning_message += ""

        if has_fraud_result:
            if fraud_is_fake:
                warning_message += "语音疑似伪造，"
            else:
                warning_message += "音频为真，"
        else:
            warning_message += ""

        if 'text' in result:
            if text_class != '不是诈骗':
                warning_message += f"通话内容存在<span style='color:blue;'>{text_class}</span>"
            else:
                warning_message += "通话内容不存在引导诈骗"
        else:
            warning_message += ""
        if 'static_emotion' in result and result['static_emotion'] or \
                'audio_emotion' in result and result['audio_emotion'] or \
                'posture' in result and result['posture']:
            warning_message += "，对方疑似存在异常情绪"
        elif not any(key in result for key in ['static_emotion', 'audio_emotion', 'posture']):
            warning_message += ""
        else:
            warning_message += "，对方未显露异常情绪"

        warning_message += "\n"
    else:
        warning_message = "未检测到诈骗行为"

    formatted_result = warning_message

    if 'video_fake' in result:
        formatted_result += (f"\n 视频是否伪造：{result['video_fake']} \n")
        if 'video_fake_times' in result:
            formatted_result += "伪造时间段：\n"
            for time in result['video_fake_times']:
                formatted_result += f"{time}\n"

    if has_fraud_result:
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
        formatted_result += f"\n音频是否伪造：音频为真"

    if 'text' in result:
        formatted_result += f"\n\n文本分析结果: {result['text']}"

    if 'static_emotion' in result:
        if isinstance(result['static_emotion'], list):
            unique_static_emotions = {}
            for res in result['static_emotion']:
                for timestamp, emotion_list in res.items():
                    for emotion in emotion_list:
                        for _, emotion_value in emotion.items():
                            if timestamp not in unique_static_emotions:
                                unique_static_emotions[timestamp] = set()
                            unique_static_emotions[timestamp].add(emotion_value)
            if unique_static_emotions:
                formatted_result += "\n视频情感分析结果："
                for timestamp, emotions in unique_static_emotions.items():
                    formatted_result += f"\n时间点 {timestamp}: {', '.join(emotions)}"
        elif isinstance(result['static_emotion'], str):
            formatted_result += f"\n图像情感分析结果: \n{result['static_emotion']}"

    if 'transfer' in result:
        formatted_result += f"\n\n语音转换结果（若没有声音会生成一些随机文字或字符）: {result['transfer']}\n"

    # if 'audio_emotion' in result:
    #     unique_emotions = {}
    #     for line in result['audio_emotion'].strip().split('\n'):
    #         if line:
    #             time_point, emotion = line.split(': ')
    #             if time_point not in unique_emotions:
    #                 unique_emotions[time_point] = set()
    #             unique_emotions[time_point].add(emotion)
    #     formatted_result += "\n语音情绪分析结果: "
    #     for time_point, emotions in unique_emotions.items():
    #         formatted_result += f"\n{time_point}: {', '.join(emotions)}"

    if 'posture' in result:
        formatted_result += "\n姿势检测结果:\n"
        for label, count in result['posture'].items():
            formatted_result += f'"{label}": 0\n'

    if 'images' in result:
        formatted_result += "\n\n异常动作检测图片:\n"
        for image_path in result['images']:
            formatted_result += f'<img src="{image_path}" style="max-width: 280px; max-height: 300px; width: auto; height: auto;">\n'

    if not formatted_result:
        formatted_result = "没有发现异常或未知分析结果"

    result_str = ""
    if 'static_emotion' in result:
        for item in result['static_emotion']:
            for time_range, emotions in item.items():
                # 统计情绪出现的频率
                emotion_count = {}
                for emotion_dict in emotions:
                    for _, emotion in emotion_dict.items():
                        if emotion in emotion_count:
                            emotion_count[emotion] += 1
                        else:
                            emotion_count[emotion] = 1
                # 找出出现次数最多的情绪
                max_emotion = max(emotion_count, key=emotion_count.get)
                # 格式化输出
                result_str += f"{time_range}s:{max_emotion}\n"

    result={'content':formatted_result,'imagesUrl':result['imagesUrl'],
            'actionsUrl':result['actionsUrl'],'audio_emotion':result['audio_emotion'],
            'emotion':result_str,'text':result['text']
            }
    return result


def split_video(video_path, segment_duration):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    segments = []

    start = 0
    while start < duration:
        end = min(start + segment_duration, duration)
        segment_path = f"{os.path.splitext(video_path)[0]}_segment_{int(start)}-{int(end)}.mp4"
        segments.append((segment_path, start))
        command = f"ffmpeg -i {video_path} -ss {start} -t {end - start} -c:v libx264 -c:a aac -strict experimental {segment_path}"
        os.system(command)
        start += segment_duration

    cap.release()

    # 如果最后一个片段的长度不足 segment_duration，将其与前一个片段合并
    if len(segments) >= 2  and (end - start < segment_duration):
        previous_segment_path = segments[-2][0]
        last_segment_path = segments[-1][0]
        # combined_segment_path = f"{os.path.splitext(video_path)[0]}_segment_{int(segments[-2][1])}-{int(end)}.mp4"
        # with open("file_list.txt", "w") as file_list:
        #     file_list.write(f"file '{previous_segment_path}'\n")
        #     file_list.write(f"file '{last_segment_path}'\n")
        # command = f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {combined_segment_path}"
        # os.system(command)
        # os.remove(previous_segment_path)
        os.remove(last_segment_path)
        # segments[-2] = (combined_segment_path, segments[-2][1])
        segments.pop()

    print(segments)

    return segments


def handle_abnormal_segments(video_path, result, abnormal_actions_with_timestamps, base_url):
    abnormal_emotions = ['害怕', '惊讶', '生气', '悲伤']
    extended_duration = 15  # 前后 15 秒

    # 收集需要处理的时间段
    time_segments_to_process = []

    # 处理静态情感
    if 'static_emotion' in result:
        for res in result['static_emotion']:
            for time_point, emotion in res.items():
                if emotion in abnormal_emotions:
                    start_time, end_time = map(int, time_point.split('-'))
                    time_segments_to_process.append((start_time, end_time))

    # 处理动态情感
    # for res in result['dynamic_emotion']:
    #     for time_point, emotion in res.items():
    #         if emotion in abnormal_emotions:
    #             start_time, end_time = map(int, time_point.split('-'))
    #             time_segments_to_process.append((start_time, end_time))

    # 处理异常动作
    result['images'] = []
    for action in abnormal_actions_with_timestamps:
        start_time = max(0, action['timestamp'] - extended_duration)
        end_time = min(int(len(AudioSegment.from_file(video_path)) / 1000), action['timestamp'] + extended_duration)
        time_segments_to_process.append((start_time, end_time))

        # 添加异常动作图片路径到结果中
        image_path = os.path.join(settings.MEDIA_ROOT, 'result', action['image_path'])
        if 'images' not in result:
            result['images'] = []
        image_url = base_url + 'media/result/' + os.path.basename(image_path)
        result['images'].append(image_url)
    


def convert_mp4_to_wav(input_path):
    """
    从 MP4 文件中提取音频并保存为 WAV 格式

    :param input_path: 输入 MP4 文件的路径
    :param output_path: 输出 WAV 文件的路径
    """
    wav_path = os.path.splitext(input_path)[0] + '.wav'
    input_path = os.path.abspath(input_path)
    wav_path = os.path.abspath(wav_path)
    try:
        # 构建 ffmpeg 命令
        command = [
            'ffmpeg',
            '-i', input_path,  # 输入文件路径
            '-vn',  # 不处理视频流
            '-acodec', 'pcm_s16le',  # 设置音频编解码器为 PCM 16 位小端格式
            '-ar', '44100',  # 设置音频采样率为 44.1 kHz
            '-ac', '2',  # 设置音频通道数为 2（立体声）
            wav_path  # 输出文件路径
        ]

        # 执行 ffmpeg 命令
        subprocess.run(command, check=True)
        print(f"音频已提取并保存到: {wav_path}")
        return wav_path

    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e}")


def save_text_to_file(text, audio_file_path):
    text_file_path = os.path.splitext(audio_file_path)[0] + "_transcription.txt"
    with open(text_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    return text_file_path


def download_file(file_url):
    local_filename = file_url.split('/')[-1]
    local_file_path = os.path.join(settings.MEDIA_ROOT, 'temp', local_filename)

    with requests.get(file_url, stream=True) as response:
        response.raise_for_status()
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    return local_file_path


class ChatHistoryView(APIView):
    @csrf_exempt
    def get(self, request, *args, **kwargs):
        user = request.user
        messages = ChatMessage.objects.filter(user=user).order_by('timestamp')
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        user = request.user
        messages = request.data
        ChatMessage.objects.filter(user=user).delete()
        for message in messages:
            ChatMessage.objects.create(
                user=user,
                from_user=message['from'],
                message_type=message['type'],
                content=message['text'] if message['type'] == 'text' else message['fileName'],
                timestamp=message['time']
            )
        return Response({"detail": "Chat history saved successfully"}, status=status.HTTP_200_OK)


def extract_frames_if_fake(video_path, result, output_dir, start_time, num_frames):
    result['imList'] = []
    image_list = []
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_interval = int(fps)  # 每秒的帧数

    # 计算当前要提取的帧索引
    start_frame = int(start_time * frame_interval)

    # 提取三帧图片
    for i in range(num_frames):
        frame_index = start_frame + i*frame_interval*3  # 计算当前要提取的帧索引
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # 保存帧为图片
            image_path = os.path.join(output_dir, f'frame_{frame_index}.jpg')
            cv2.imwrite(image_path, frame)

            # 添加图片路径到结果中
            result['imList'].append(image_path)
        image_list.append("http://localhost:8000/media/imageList/"+f"frame_{frame_index}.jpg")
    cap.release()
    return image_list


# ------add------
def weighted_sum(values, weights):
    # 如果两个列表的长度不一致，返回None
    if len(values) != len(weights):
        return None

    # 检查是否有 -1 的情况
    if -1 in values:
        total_weight = sum(weight for i, weight in enumerate(weights) if values[i] != -1)
        # 将 -1 的权重分配给其他元素
        for i in range(len(values)):
            if values[i] == -1:
                weights[i] = 0  # 把 -1 对应的权重置为0
            else:
                weights[i] += weights[i] * (1 - total_weight) / total_weight

    # 计算带权求和
    result = sum(value * weight for value, weight in zip(values, weights))
    if result == 0:
        return -1
    return round(result,3)
# ------add------