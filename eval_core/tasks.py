# eval_core/tasks.py
import json
from abc import ABC, abstractmethod
from .metrics import compute_exact_match, compute_f1, compute_accuracy
import re

class BaseTask(ABC):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
    
    def load_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
        return data

    @abstractmethod
    def build_messages(self, sample):
        pass
    
    @abstractmethod
    def get_ground_truth(self, sample):
        pass
    
    @abstractmethod
    def post_process(self, prediction):
        """
        严格模式下，这里只做必要的提取，不做清洗。
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, gold, pred):
        pass


# ========================================================
# 1. WikiTableQuestions (Strict Mode)
#    要求：模型必须完全听话，只输出答案实体，多一个字都算错。
# ========================================================
class WTQTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        # 你的严格指令
        self.instruction = (
            "Read the table and answer the question. "
            "Output ONLY the exact answer entity (e.g., a number, date, or name). "
            "DO NOT output a full sentence. "
            "DO NOT provide explanations or context. "
            "Just the answer."
        )

    def get_ground_truth(self, sample):
        if 'answer' in sample: return sample['answer']
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        return assistant_msg['content'] if assistant_msg else ""
    
    def build_messages(self, sample):
        content = ""
        if 'question' in sample and 'table' in sample:
            content = f"Table:\n{sample['table']}\nQuestion: {sample['question']}"
        else:
            messages = sample.get('messages', [])
            user_msg = next((m for m in messages if m['role'] == 'user'), None)
            if user_msg: content = user_msg['content']
        
        if not content: return []

        return [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question concisely."},
            {"role": "user", "content": content + "\n\n" + self.instruction}
        ]
    
    def post_process(self, prediction):
        # [严格模式]
        # 只去掉首尾空格，不做任何正则替换。
        # 如果模型输出了 "The answer is 5"，这里返回 "The answer is 5"
        # 进而导致和 Gold "5" 匹配失败。这是符合预期的。
        if not prediction: return ""
        return prediction.strip()

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        f1 = compute_f1(gold, pred)
        return {"em": em, "f1": f1}


import re
from .metrics import compute_accuracy


class TabFactTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.instruction = (
            "\n\nEnd your response with 'Answer: entailed' or 'Answer: refuted'."
        )

    def get_ground_truth(self, sample):
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                return msg['content'].strip().lower().replace('.', '')
        return ""

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        if not user_content:
            return []

        # 加上换行符，自然拼接
        sep = "\n" if not user_content.endswith("\n") else ""
        final_content = f"{user_content}{sep}{self.instruction}"
        
        return [
            {"role": "user", "content": final_content}
        ]

    def post_process(self, prediction):
        """
        后处理保持严格模式：
        既然指令里要求了 End your response with 'Answer: ...'
        我们就只提取这个格式。
        """
        if not prediction:
            return "unknown"
            
        pred_lower = prediction.lower()
        
        # 严格匹配 Answer: entailed/refuted
        # 允许 Answer 后面有空格
        match = re.search(r'answer:\s*(entailed|refuted)', pred_lower)
        
        if match:
            return match.group(1)
            
        # 如果模型没按格式输出，直接判错 (unknown)
        return "unknown"

    def compute_metrics(self, gold, pred):
        acc = compute_accuracy(gold, pred)
        return {"accuracy": acc}


# Factory
def get_task(task_name, data_path):
    name = task_name.lower().strip()
    if name == "wtq": return WTQTask(data_path)
    elif name == "tabfact": return TabFactTask(data_path)
    raise ValueError(f"Unknown task: {task_name}")