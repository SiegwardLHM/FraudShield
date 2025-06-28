import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from bert import Bert_Classifier
from llama import Llama
from cnsenti import Emotion


class_data = ['不是诈骗', '中奖', '网络刷单', '假冒领导，熟人', '虚假招聘', '网络交易', '绑架', '代购', '网络交友',
              '贷款信用', '炒股理财', '冒充公检法及政府机构', '保险', '垃圾信息']

class TextModel:
    def __init__(self):
        self.bert = Bert_Classifier()
        self.bert.load_best_model()
        self.llama = Llama()
        self.llama.load('best_int4')

    def predict(self, text):
        return self.bert.predict(text)

    def explain(self, text, label):
        return self.llama.chat(text, label)


model = TextModel()


def analyze_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    label, label_confidence = model.predict(content)
    # label = model.predict(content)
    if(is_nervous(content)):
        nervous = "紧张"
    else:
        nervous = "不紧张"
    reason = ''
    if label != 0 and label != 13:
        reason = model.explain(content, label)
    classification = f"{class_data[label]}"
    print(classification)
    print(reason)
    return classification, reason,label_confidence


def is_nervous(text):
    emotion = Emotion()
    result = emotion.emotion_count(text)
    if result['怒'] + result['惧'] + result['惊'] + result['恶'] > 0:
        return True
    return False
