#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', cache_dir='./robot_model')

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online', cache_dir='./realtime_model')