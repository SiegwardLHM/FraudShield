import os
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件所在目录
bert_path = os.path.join(current_dir, "model/base")
best_path = os.path.join(current_dir, "model/best.pth")
tokenizer = BertTokenizer.from_pretrained(bert_path)


class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=14):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # 池化后的输出
        logit = self.fc(out_pool)
        return logit


class Bert_Classifier():
    def __init__(self):
        self.model = Bert_Model(bert_path).to(DEVICE)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(best_path))

    def predict(self, text):
        encode_dict = tokenizer.encode_plus(text=text, max_length=512,
                                            padding='max_length', truncation=True)

        self.model.eval()
        with torch.no_grad():
            input_ids = torch.LongTensor(encode_dict['input_ids']).unsqueeze(0).to(DEVICE)
            input_types = torch.LongTensor(encode_dict['token_type_ids']).unsqueeze(0).to(DEVICE)
            input_masks = torch.LongTensor(encode_dict['attention_mask']).unsqueeze(0).to(DEVICE)

            logits = self.model(input_ids, input_masks, input_types)
            probabilities = F.softmax(logits, dim=1)  # 计算 softmax，得到概率值
            label = torch.argmax(probabilities, dim=1).item()  # 获取预测结果的标签值

        return label, probabilities



