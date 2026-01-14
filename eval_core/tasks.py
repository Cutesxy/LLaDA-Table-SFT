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
# 2. WikiTableQuestions (CoT Mode)
#    新增模式：允许模型进行链式思考，但强制输出格式。
# ========================================================  
class WTQCoTTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        pass 
    
    def get_ground_truth(self, sample):
        # 1. 优先取标准数据集里的 answer 字段
        if 'answer' in sample: return sample['answer']
        
        # 2. 兼容 SFT 格式 (从 assistant 消息中提取)
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg:
            content = assistant_msg['content']
            # 尝试提取末尾的 Answer: [...]
            match = re.search(r'(?i)Answer\s*[:：]?\s*(?:\[)?(.*?)(?:\])?$', content, re.MULTILINE)
            if match:
                return match.group(1).strip()
            return content
        return ""
    
    def build_messages(self, sample):
            # 1. 尝试提取标准字段
            context = sample.get('table', "")
            question = sample.get('question', "")
            
            # 2. 准备核心输入内容
            core_input = ""
            if context and question:
                core_input = f"# Table Context\n{context}\n\n# Question\n{question}\n\n"
            else:
                messages = sample.get('messages', [])
                user_msg = next((m for m in messages if m['role'] == 'user'), None)
                if user_msg:
                    core_input = user_msg['content'] + "\n\n"
            
            if not core_input: return []
            
            # =======================================================
            # [修复版 Prompt] 两段式指令
            # 1. Analysis Section: 明确授权可以写句子
            # 2. Answer Section: 明确约束只针对最后的方括号
            # =======================================================
            instruction = (
                f"# Instruction\n"
                f"You are a precise data analyst. Follow these two steps:\n\n"
                
                f"Step 1: Analysis (Optional but recommended)\n"
                f"You may briefly analyze the table data to derive the answer. "
                f"Full sentences and reasoning are ALLOWED in this step.\n\n"
                
                f"Step 2: Final Answer\n"
                f"At the very end, output the result in this exact format:\n"
                f"Answer: [The Result]\n\n"
                
                f"Constraints for Step 2 ONLY (Inside the brackets):\n"
                f"1. Put ONLY the exact answer entity inside [ ].\n"
                f"2. DO NOT include explanations inside [ ].\n"
                f"3. Example: Answer: [50]"
            )
            
            final_user_content = core_input + instruction

            return [
                {"role": "system", "content": "You are a precise data analyst."},
                {"role": "user", "content": final_user_content}
            ]

    def post_process(self, prediction):
        """
        [精准提取逻辑]
        配合 Strict Constraints，优先抓取方括号内的实体。
        """
        if not prediction: return ""
        pred_clean = prediction.strip()
        
        # ==========================================================
        # 策略 1: 优先提取 Answer: [...] (最高优先级)
        # ==========================================================
        # 使用 finditer 取最后一次出现的答案
        matches = list(re.finditer(r'(?i)Answer\s*[:：]?\s*\[(.*?)\]', pred_clean, re.DOTALL))
        if matches:
            return self._clean_result(matches[-1].group(1))

        # ==========================================================
        # 策略 2: 提取 Answer: ... (无括号备选)
        # ==========================================================
        # 防止模型虽然答对了，但忘了加方括号
        matches = list(re.finditer(r'(?i)Answer\s*[:：]?\s*([^\n]+)', pred_clean))
        if matches:
            content = matches[-1].group(1).strip()
            # 行内截断清洗：防止 "50 because..."
            if ". " in content: content = content.split(". ")[0]
            if "(" in content: content = content.split("(")[0]
            return self._clean_result(content)

        # ==========================================================
        # 策略 3: 暴力兜底 (最后一行极短文本)
        # ==========================================================
        # 如果模型直接扔了一个数字在最后一行，没有写 "Answer:"
        lines = pred_clean.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line: continue
            
            # 排除包含 "Analysis", "Step" 等词的行
            if any(k in line.lower() for k in ["analysis", "step", "reasoning", "therefore"]):
                break 

            # 如果是纯数字或很短的实体 (长度<30)
            if len(line) < 30:
                return self._clean_result(line)
            break
            
        return ""

    def _clean_result(self, text):
        """
        最终清洗：去除格式残留
        """
        if not text: return ""
        text = str(text).strip()
        
        # 去除方括号 (双重保险)
        text = text.replace('[', '').replace(']', '')
        
        # 去除 Markdown 和 LaTeX
        text = text.replace('**', '').replace('`', '').replace('$', '').replace('\\boxed', '')
        
        # 去除末尾句号 (仅当它不是缩写的一部分时)
        # 例如保留 "U.S." 但去除 "50."
        if text.endswith('.'):
             if not re.search(r'[A-Z]\.$', text):
                text = text[:-1]
                
        return text.strip()

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