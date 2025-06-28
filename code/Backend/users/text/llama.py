from transformers import AutoModelForCausalLM, AutoTokenizer
import os
path = os.path.dirname(os.path.realpath(__file__))


class Llama:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load(self, model):
        model_path = os.path.join(path, 'model', model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                              trust_remote_code=True).eval()

    def chat(self, text, class_):
        class_data = ['不是诈骗', '中奖', '网络刷单', '假冒领导，熟人', '虚假招聘', '网络交易', '绑架', '代购', '网络交友',
              '贷款信用', '炒股理财', '冒充公检法及政府机构', '保险', '垃圾信息']

        messages = [
            {"role": "user", "content": f"请阅读以下对话或者段落 ： “{text}”。请分析该对话或段落，具体详细的解释清楚其为什么属于{class_data[class_]}类型的诈骗"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        reason = self.tokenizer.decode(response, skip_special_tokens=True)
        return reason
