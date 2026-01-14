import json
import re
from abc import ABC, abstractmethod
from .metrics import compute_exact_match, compute_f1, compute_accuracy

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
        处理模型输出，提取核心答案用于计算 Metric
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, gold, pred):
        pass


# ========================================================
# 1. WikiTableQuestions (Strict / Direct Answer Mode)
#    旧模式：强制模型只输出答案，不许废话。
# ========================================================
class WTQTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.instruction = (
            "Read the table and answer the question. "
            "Output ONLY the exact answer entity (e.g., a number, date, or name). "
            "DO NOT output a full sentence. "
            "DO NOT provide explanations or context. "
            "Just the answer."
        )

    def get_ground_truth(self, sample):
        if 'answer' in sample: return sample['answer']
        # 兼容 SFT 数据格式读取
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg:
            # 如果是 CoT 数据作为 GT，可能需要提取 Answer 后面的部分，
            # 但如果是评测集通常会有专门的 answer 字段。
            # 这里假设作为 GT 时，我们尽量取纯净答案。
            content = assistant_msg['content']
            if "Answer:" in content:
                return content.split("Answer:")[-1].strip()
            return content
        return ""
    
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
        if not prediction: return ""
        return prediction.strip()

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        f1 = compute_f1(gold, pred)
        return {"em": em, "f1": f1}


# ========================================================
# 2. [重点] WikiTableQuestions (CoT Mode)
#    用于测试 SFT 效果以及和 Qwen/Llama 进行公平对比
# ========================================================
class WTQCoTTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        # 结合了 "Step by step" (激发推理) 和 "Format Constraint" (方便测评)
        self.instruction = (
            "Let's think step by step. "
            "Analyze the table and the question, then provide the answer. "
            "At the end of your response, output the final result in this format: Answer: <result>"
        )

    def get_ground_truth(self, sample):
        # 优先取标准数据集里的 answer 字段
        if 'answer' in sample: return sample['answer']
        
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg:
            content = assistant_msg['content']
            # 如果 GT 数据本身就是 CoT 格式，提取 Answer 后面的部分
            match = re.search(r'(?i)answer:\s*(.*)', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return content
        return ""
    
    def build_messages(self, sample):
        content = ""
        # 优先处理标准 WTQ 格式
        if 'question' in sample and 'table' in sample:
            content = f"Table:\n{sample['table']}\nQuestion: {sample['question']}"
        # 兼容 SFT 格式
        else:
            messages = sample.get('messages', [])
            user_msg = next((m for m in messages if m['role'] == 'user'), None)
            if user_msg: content = user_msg['content']
        
        if not content: return []

        # 使用 \n\n 分隔，确保指令清晰
        return [
            {"role": "user", "content": content + "\n\n" + self.instruction}
        ]

    def post_process(self, prediction):
        """
        核心提取逻辑：寻找 'Answer:' 之后的内容
        """
        if not prediction: return ""
        
        # 1. 正则提取
        match = re.search(r'(?i)answer:\s*(.*)', prediction, re.DOTALL)
        
        if match:
            extracted = match.group(1).strip()
            
            # 2. 清洗：有时候模型会在答案末尾加句号 "Answer: 5." -> "5"
            # 这一步对于 EM 计算很重要
            if extracted.endswith('.'):
                extracted = extracted[:-1]
                
            return extracted.strip()
        
        # 如果没有 Answer: 标记，返回空字符串（视为格式错误/回答错误）
        return "" 

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        f1 = compute_f1(gold, pred)
        return {"em": em, "f1": f1}


# ========================================================
# 3. TabFact (Strict Classification)
# ========================================================
class TabFactTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.instruction = (
            "\n\nEnd your response with 'Answer: entailed' or 'Answer: refuted'."
        )

    def get_ground_truth(self, sample):
        # 优先从 label 字段获取（如果有）
        if 'label' in sample:
            return "entailed" if sample['label'] == 1 else "refuted"
            
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'].strip().lower()
                # 尝试提取 GT 中的标签
                if 'entailed' in content: return 'entailed'
                if 'refuted' in content: return 'refuted'
        return ""

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
        # 尝试从 messages 获取，或者直接从 key 构造
        if not messages and 'statement' in sample and 'table' in sample:
             user_content = f"Table:\n{sample['table']}\nStatement: {sample['statement']}"
        else:
            for msg in messages:
                if msg['role'] == 'user':
                    user_content = msg['content']
                    break
        
        if not user_content: return []

        sep = "\n" if not user_content.endswith("\n") else ""
        final_content = f"{user_content}{sep}{self.instruction}"
        
        return [
            {"role": "user", "content": final_content}
        ]

    def post_process(self, prediction):
        if not prediction: return "unknown"
        pred_lower = prediction.lower()
        match = re.search(r'answer:\s*(entailed|refuted)', pred_lower)
        if match: return match.group(1)
        return "unknown"

    def compute_metrics(self, gold, pred):
        acc = compute_accuracy(gold, pred)
        return {"accuracy": acc}


# ========================================================
# Factory
# ========================================================
def get_task(task_name, data_path):
    name = task_name.lower().strip()
    if name == "wtq": 
        # 默认 WTQ 还是 Strict 模式，保持兼容
        return WTQTask(data_path)
    elif name == "wtq-cot":
        # 新增的 CoT 模式
        return WTQCoTTask(data_path)
    elif name == "tabfact": 
        return TabFactTask(data_path)
    raise ValueError(f"Unknown task: {task_name}")