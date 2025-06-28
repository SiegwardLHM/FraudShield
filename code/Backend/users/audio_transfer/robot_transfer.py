from funasr import AutoModel
import os

model_dir = "robot_model/iic/paraformer-zh"
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
model = AutoModel(model=model_dir, device="cuda")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def transfer_audio(file_path):
    # 进行语音转文字
    print(file_path)
    res = model.generate(input=file_path)

    # 返回结果和用时
    return res[0]['text']


# result = transcribe_audio(file_path)
