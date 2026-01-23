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
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg:
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
# ========================================================  
class WTQCoTTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        pass 
    
    def get_ground_truth(self, sample):
        if 'answer' in sample: return sample['answer']
        messages = sample.get('messages', [])
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg:
            content = assistant_msg['content']
            match = re.search(r'(?i)Answer\s*[:：]?\s*(?:\[)?(.*?)(?:\])?$', content, re.MULTILINE)
            if match:
                return match.group(1).strip()
            return content
        return ""
    
    def build_messages(self, sample):
            context = sample.get('table', "")
            question = sample.get('question', "")
            
            core_input = ""
            if context and question:
                core_input = f"# Table Context\n{context}\n\n# Question\n{question}\n\n"
            else:
                messages = sample.get('messages', [])
                user_msg = next((m for m in messages if m['role'] == 'user'), None)
                if user_msg:
                    core_input = user_msg['content'] + "\n\n"
            
            if not core_input: return []
            
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
        if not prediction: return ""
        pred_clean = prediction.strip()
        matches = list(re.finditer(r'(?i)Answer\s*[:：]?\s*\[(.*?)\]', pred_clean, re.DOTALL))
        if matches:
            return self._clean_result(matches[-1].group(1))

        matches = list(re.finditer(r'(?i)Answer\s*[:：]?\s*([^\n]+)', pred_clean))
        if matches:
            content = matches[-1].group(1).strip()
            if ". " in content: content = content.split(". ")[0]
            if "(" in content: content = content.split("(")[0]
            return self._clean_result(content)

        lines = pred_clean.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line: continue
            if any(k in line.lower() for k in ["analysis", "step", "reasoning", "therefore"]):
                break 
            if len(line) < 30:
                return self._clean_result(line)
            break
            
        return ""

    def _clean_result(self, text):
        if not text: return ""
        text = str(text).strip()
        text = text.replace('[', '').replace(']', '')
        text = text.replace('**', '').replace('`', '').replace('$', '').replace('\\boxed', '')
        if text.endswith('.'):
             if not re.search(r'[A-Z]\.$', text):
                text = text[:-1]
        return text.strip()

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        f1 = compute_f1(gold, pred)
        return {"em": em, "f1": f1}

# ========================================================
# 3. WikiTableQuestions (Completion / Infilling Mode)
#    [NEW] 专门针对 LLaDA 扩散模型设计的填空模式
#    去除 Chat Template，直接输出 Table + Question + Answer:
# ========================================================
class WTQCompletionTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        # 不需要 system instruction，格式本身就是指令
    
    def get_ground_truth(self, sample):
        # 1. 优先取标准字段
        if 'answer' in sample: return sample['answer']
        
        # 2. 从 SFT 格式的 assistant 消息中提取
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                return msg['content'].strip()
        return ""
    
    def build_messages(self, sample):
        """
        试卷风格 (Exam Style) 的 Prompt 构造
        """
        # 1. 提取原始 User 输入
        messages = sample.get('messages', [])
        raw_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                raw_content = msg['content']
                break
        
        if not raw_content:
            return ""

        # 2. 数据清洗
        tab_start = raw_content.find("[TAB]")
        if tab_start == -1:
            return raw_content + "\nAnswer:"
            
        clean_content = raw_content[tab_start:] 
        
        # 3. 分离表格和问题
        parts = clean_content.rsplit("\n\n", 1)
        
        if len(parts) == 2:
            table_part = parts[0].strip()
            question_part = parts[1].strip()
            
            # ==========================================
            # [核心修改] 构造试卷风格的 Prompt
            # ==========================================
            prompt = (
                "== Table Analysis Task ==\n\n"     # 试卷标题
                "[Reference Data]\n"                # 数据区标题
                f"{table_part}\n\n"                 # 表格内容
                "-------------------\n\n"           # 分隔线 (视觉辅助)
                "[Question]\n"                      # 问题区标题
                f"{question_part}\n\n"              # 问题内容
                "[Answer]\n"                        # 答题区 (留给模型填空)
            )
            # 注意：Prompt 到 "[Answer]\n" 结束，
            # 后面紧接着的就是 Diffusion 模型要生成的 [MASK] (即那个"空")
            
        else:
            # 兜底逻辑
            prompt = (
                "[Data]\n"
                f"{clean_content}\n\n"
                "[Answer]\n"
            )

        return prompt
    
    def post_process(self, prediction):
        """
        简单清洗：取第一行，因为扩散生成可能会有多余的 padding 或乱码
        (Models.py 中已经做了 EOS 截断，这里做二次保险)
        """
        if not prediction: return ""
        return prediction.strip().split('\n')[0]

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        return {"em": em}


# ========================================================
# 4. TabFact (Strict Classification)
# ========================================================
class TabFactTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.instruction = (
            "\n\nEnd your response with 'Answer: entailed' or 'Answer: refuted'."
        )

    def get_ground_truth(self, sample):
        if 'label' in sample:
            return "entailed" if sample['label'] == 1 else "refuted"
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'].strip().lower()
                if 'entailed' in content: return 'entailed'
                if 'refuted' in content: return 'refuted'
        return ""

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
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
        return WTQTask(data_path)
    elif name == "wtq-cot":
        return WTQCoTTask(data_path)
    elif name == "wtq-completion": # <--- 新增的注册入口
        return WTQCompletionTask(data_path)
    elif name == "tabfact": 
        return TabFactTask(data_path)
    raise ValueError(f"Unknown task: {task_name}")