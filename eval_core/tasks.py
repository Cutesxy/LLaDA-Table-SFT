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
# ========================================================
class WTQCompletionTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
    
    def get_ground_truth(self, sample):
        if 'answer' in sample: return sample['answer']
        
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                return msg['content'].strip()
        return ""
    
    def build_messages(self, sample):
        messages = sample.get('messages', [])
        raw_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                raw_content = msg['content']
                break
        
        if not raw_content:
            return ""

        tab_start = raw_content.find("[TAB]")
        if tab_start == -1:
            return raw_content + "\nAnswer:"
            
        clean_content = raw_content[tab_start:] 
        
        parts = clean_content.rsplit("\n\n", 1)
        
        if len(parts) == 2:
            table_part = parts[0].strip()
            question_part = parts[1].strip()
            
            prompt = (
                "== Table Analysis Task ==\n\n"
                "[Reference Data]\n"
                f"{table_part}\n\n"
                "-------------------\n\n"
                "[Question]\n"
                f"{question_part}\n\n"
                "[Answer]\n"
            )
        else:
            prompt = (
                "[Data]\n"
                f"{clean_content}\n\n"
                "[Answer]\n"
            )
        return prompt
    
    def post_process(self, prediction):
        if not prediction: return ""
        return prediction.strip().split('\n')[0]

    def compute_metrics(self, gold, pred):
        em = compute_exact_match(gold, pred)
        return {"em": em}

# ========================================================
# 4. TabFact (Strict Classification) - Chat Mode
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
# 5. TabFact (Completion / Infilling Mode)
#    [NEW] Zero-shot 纯填空模式 (Regex Enhanced)
# ========================================================
class TabFactCompletionTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
    
    def get_ground_truth(self, sample):
        # 兼容 TabFact 原始数据格式 (label=1/0)
        if 'label' in sample:
            return "entailed" if sample['label'] == 1 else "refuted"
            
        # 兼容 SFT 数据格式 (messages)
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'].strip().lower()
                if 'entailed' in content: return 'entailed'
                if 'refuted' in content: return 'refuted'
        return "refuted" # 兜底
    
    def build_messages(self, sample):
        """
        构造 TabFact 专属的 Zero-shot 试卷填空格式
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

        # 2. 精准解析 (Regex Parsing)
        # A. 提取表格 (Table)
        table_raw = ""
        tab_match = re.search(r'\[TAB\](.*?)(\n\nThe statement is:)', raw_content, re.DOTALL | re.IGNORECASE)
        
        if tab_match:
            table_raw = tab_match.group(1).strip()
        else:
            parts = raw_content.split('[TAB]')
            if len(parts) > 1:
                table_raw = parts[1].split('\n\n')[0].strip()

        # B. 提取陈述 (Statement)
        statement_raw = ""
        stmt_match = re.search(r'The statement is:\s*<(.*?)>', raw_content, re.DOTALL | re.IGNORECASE)
        
        if stmt_match:
            statement_raw = stmt_match.group(1).strip()
        else:
            fallback_match = re.search(r'The statement is:\s*(.*?)(?=\.\s*Is it)', raw_content, re.DOTALL | re.IGNORECASE)
            if fallback_match:
                statement_raw = fallback_match.group(1).strip()
            else:
                statement_raw = "Unknown Statement"

        # 3. 构造 Prompt (Zero-shot)
        
        # ------------------------------------------------------------------
        # [STRATEGY 1: Standard Append Layout] (已注释 - 备份)
        # 之前的策略：Title -> Data -> Statement -> [Answer] 标签
        # 预期输出样式：
        # == Table Verification Task ==
        # [Reference Data]
        # | col | ... |
        # -------------------
        # [Statement]
        # 5 designation hds be send ...
        # [Answer]
        # ------------------------------------------------------------------
        prompt = (
            "== Table Verification Task ==\n\n"
            "[Reference Data]\n"                
            f"{table_raw}\n\n"
            "-------------------\n\n"           
            "[Statement]\n"                     
            f"{statement_raw}\n\n"
            "[Answer]\n"                        
        )

        # ------------------------------------------------------------------
        # [STRATEGY 2: Embedded/Infilling Layout] (当前激活)
        # 新策略：将判断任务嵌入到自然语言句子中，利用 Diffusion 的补全能力。
        # 预期输出样式：
        # == Fact Checking Task ==
        # [Reference Data]
        # | col | ... |
        # -------------------
        # Based on the table above, the statement "5 designation hds be send ..." is
        # ------------------------------------------------------------------
        
        # prompt = (
        #     "== Fact Checking Task ==\n\n"      # 稍微改个更自然的标题
        #     "[Reference Data]\n"
        #     f"{table_raw}\n\n"
        #     "-------------------\n\n"
        #     # 下面是关键修改：构建一个未完成的句子
        #     f'Based on the table above, the statement "{statement_raw}" is' 
        #     # 注意：这里不需要手动加 [MASK]，LLaDA 推理代码会自动在末尾生成
        #     # 模型应当接续生成 " entailed" 或 " refuted" 或 " correct" 等词
        # )
        
        return prompt
    
    def post_process(self, prediction):
        """
        清洗 LLaDA 的输出
        """
        if not prediction: return "unknown"
        
        # 1. 取第一行，转小写
        text = prediction.strip().split('\n')[0].lower()
        
        # 2. 关键词匹配 (TabFact 的金标准)
        if "entailed" in text: return "entailed"
        if "refuted" in text: return "refuted"
        
        # 3. 常见同义词映射 (适应自然语言补全可能出现的词)
        # 因为 Prompt 变成了 "... is X", 模型可能会填 "true", "correct", "wrong" 等
        if "true" in text or "yes" in text or "correct" in text or "right" in text: return "entailed"
        if "false" in text or "no" in text or "wrong" in text or "incorrect" in text: return "refuted"
        
        return "unknown"

    def compute_metrics(self, gold, pred):
        acc = compute_accuracy(gold, pred)
        return {"accuracy": acc}

# ========================================================
# 6. Feverous (Generalization Test - 3-Class Optimized)
#    [OPTIMIZED] 支持 Supports/Refutes/NotEnoughInfo 三分类
# ========================================================
class FeverousCompletionTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
    
    def get_ground_truth(self, sample):
        # 1. 尝试从 assistant message 获取
        messages = sample.get('messages', [])
        raw_label = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                raw_label = msg['content'].strip().lower()
                break
        
        # 2. 如果没找到，尝试找 label 字段
        if not raw_label and 'label' in sample:
            raw_label = str(sample['label']).lower()

        # [关键修改] 三分类映射 (Label Alignment)
        if "supports" in raw_label: return "entailed"
        if "refutes" in raw_label: return "refuted"
        if "not enough" in raw_label or "unknown" in raw_label: return "unknown"
        
        return "unknown" # 现在的兜底逻辑改为 unknown，这更符合逻辑（不确定就是未知）
    
    def build_messages(self, sample):
        messages = sample.get('messages', [])
        raw_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                raw_content = msg['content']
                break
        
        if not raw_content: return ""

        # Regex 解析
        table_raw = ""
        tab_match = re.search(r'\[TAB\](.*?)(\n\nThe statement is:)', raw_content, re.DOTALL | re.IGNORECASE)
        if tab_match:
            table_raw = tab_match.group(1).strip()
        else:
            parts = raw_content.split('[TAB]')
            if len(parts) > 1:
                table_raw = parts[1].split('\n\n')[0].strip()

        statement_raw = ""
        stmt_match = re.search(r'The statement is:\s*<(.*?)>', raw_content, re.DOTALL | re.IGNORECASE)
        if stmt_match:
            statement_raw = stmt_match.group(1).strip()
        else:
            statement_raw = "Unknown Statement"

        # [关键修改] Prompt 注入 (Prompt Injection)
        # 我们需要在 prompt 里明确告诉模型，'unknown' 是一个合法的选项。
        # 否则受 TabFact SFT 影响，它可能不敢输出 unknown。
        prompt = (
            "== Table Verification Task ==\n\n"
            "[Reference Data]\n"                
            f"{table_raw}\n\n"
            "-------------------\n\n"           
            "[Statement]\n"                     
            f"{statement_raw}\n\n"

            "[Answer]\n"                        
        )
        return prompt
    
    def post_process(self, prediction):
        if not prediction: return "unknown"
        text = prediction.strip().split('\n')[0].lower()
        
        # [关键修改] 增加 unknown 的捕获
        if "unknown" in text or "not enough" in text: return "unknown"
        
        if "entailed" in text: return "entailed"
        if "refuted" in text: return "refuted"
        
        # 兼容性映射
        if "supports" in text or "true" in text or "correct" in text: return "entailed"
        if "false" in text or "wrong" in text: return "refuted"
        
        return "unknown"

    def compute_metrics(self, gold, pred):
        # 这里计算的就是标准的 3 分类 Accuracy
        acc = compute_accuracy(gold, pred)
        return {"accuracy": acc}
    
# ========================================================
# 7. TabFact RL (CoT Mode) - 适配队友的 <think> 格式
# ========================================================
class TabFactRLTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.system_prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            "Respond in the following format:\n"
            "<think>\n"
            "reasoning process here\n"
            "</think>\n"
            "answer here."
        )

    def get_ground_truth(self, sample):
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                return msg['content'].strip()
        if 'label' in sample:
            return "entailed" if sample['label'] == 1 else "refuted"
        return "unknown"

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        if not user_content: return []

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

    def post_process(self, prediction):
        """
        清洗逻辑：提取 </think> 后面的内容，并模糊匹配 refuted/entailed
        """
        if not prediction: return "unknown"
        
        # 1. 转小写
        text = prediction.strip().lower()
        
        # 2. 剥离 <think>，只看推理之后的部分
        if "</think>" in text:
            # 取 </think> 之后的内容
            check_content = text.split("</think>")[-1]
        else:
            # 如果没闭合，查全文
            check_content = text
            
        # 3. 关键词扫描 (只要出现了就算)
        if "refuted" in check_content:
            return "refuted"
        if "entailed" in check_content:
            return "entailed"
            
        return "unknown"

    def compute_metrics(self, gold, pred):
        # [核心修复]：强制在这里再清洗一次！
        # 无论外面传进来的是原始文本还是处理过的，这里都确保拿到的是核心答案
        clean_pred = self.post_process(pred)
        
        # 确保 gold 也是干净的 (防止 gold 里有空格)
        clean_gold = gold.strip().lower() if gold else ""
        
        return {"accuracy": compute_accuracy(clean_gold, clean_pred)}

# ========================================================
# 8. WTQ RL (CoT Mode) - 适配队友的 <think> 格式
# ========================================================
class WTQRLTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        # 复用队友的 CoT System Prompt
        self.system_prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            "Respond in the following format:\n"
            "<think>\n"
            "reasoning process here\n"
            "</think>\n"
            "answer here."
        )

    def get_ground_truth(self, sample):
        # 1. 优先直接取 answer 字段 (WTQ 标准格式通常是 list，但这里兼容 string)
        if 'answer' in sample: 
            return sample['answer']
        
        # 2. 兼容 messages 格式 (从 assistant 回复中提取)
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                # 如果 GT 本身也有 <think> (RL 训练数据)，尝试剥离只取答案
                if "</think>" in content:
                    return content.split("</think>")[-1].strip()
                return content.strip()
        return ""

    def build_messages(self, sample):
        # 1. 尝试从 messages 获取
        messages = sample.get('messages', [])
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        # 2. 如果没有 messages，尝试从 table/question 字段构建 (WTQ 原始格式)
        if not user_content and 'question' in sample and 'table' in sample:
            user_content = f"Table:\n{sample['table']}\nQuestion: {sample['question']}"

        if not user_content: return []

        # 3. 拼接 System Prompt
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

    def post_process(self, prediction):
        """
        核心清洗逻辑：剥离 <think>，并去掉常见的 "The answer is" 前缀
        """
        if not prediction: return ""
        text = prediction.strip()
        
        # 1. 剥离 <think> 部分
        if "</think>" in text:
            # 取 </think> 之后的内容
            text = text.split("</think>")[-1].strip()
        
        # 2. 清洗引导词 (RL 模型可能会输出 "The answer is 5" 而不是 "5")
        # 使用正则去掉开头可能的 "Answer:", "The answer is", "Result:"
        # flags=re.IGNORECASE 忽略大小写
        text = re.sub(r'^(the\s+)?answer\s+(is\s+)?|^(the\s+)?result\s+(is\s+)?|^[:：]\s*', '', text, flags=re.IGNORECASE).strip()
        
        # 3. 去掉末尾的句号 (如果答案是实体，通常不带句号，除非是缩写如 U.S.)
        # 简单的规则：如果结尾是句号，且倒数第二个字符不是大写字母(避免误删缩写)，则删掉
        if text.endswith('.') and not re.search(r'[A-Z]\.$', text):
            text = text[:-1]
            
        return text.strip()

    def compute_metrics(self, gold, pred):
        # [核心修复]：强制在这里再清洗一次！
        clean_pred = self.post_process(pred)
        
        # WTQ 的 Metric 计算 (EM 和 F1)
        return {
            "em": compute_exact_match(gold, clean_pred),
            "f1": compute_f1(gold, clean_pred)
        }

class WTQRobustTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
        # [修改点] 采纳你的建议：基于表格分析问题
        # 英文：Analyze the question based on the table and answer it.
        # 这样既强调了 Source (Table)，也强调了 Action (Analyze & Answer)
        self.instruction = "Analyze the question based on the table and answer it. End your response with 'Answer: <result>'."

    def get_ground_truth(self, sample):
        if 'answer' in sample: 
            return sample['answer']
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                if "Answer:" in content:
                    return content.split("Answer:")[-1].strip()
                return content.strip()
        return ""

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        if not user_content and 'question' in sample:
             user_content = f"Table:\n{sample['table']}\nQuestion: {sample['question']}"
             
        if not user_content: return []

        # 简单的拼接，给模型自由度，但指令明确要求 "Based on the table"
        return [
            {"role": "user", "content": user_content + "\n\n" + self.instruction}
        ]
    
    def post_process(self, prediction):
        """
        核心优化：不仅提取 Answer，还负责清洗 '废话'
        """
        if not prediction: return ""
        text = prediction.strip()
        
        # 1. 提取 Answer: 后的内容
        match = re.search(r'(?i)answer\s*[:：]\s*(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # 兜底：取最后一行
            lines = text.split('\n')
            if len(lines) > 1:
                text = lines[-1].strip()
        
        # 2. 移除常见的“废话”前缀
        # 针对: "The answer is 17" -> "17"
        text = re.sub(r'^(the\s+)?answer\s+(is\s+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(it\s+is\s+)|^(the\s+result\s+is\s+)', '', text, flags=re.IGNORECASE)
        
        # 3. 去掉句号 (除非是缩写如 U.S.)
        if text.endswith('.') and not re.search(r'[A-Z]\.$', text): 
            text = text[:-1]
            
        # 4. 去掉 Markdown (**17**)
        text = text.replace('**', '').replace('`', '').strip()
        
        return text

    def compute_metrics(self, gold, pred):
        clean_pred = self.post_process(pred)
        return {
            "em": compute_exact_match(gold, clean_pred),
            "f1": compute_f1(gold, clean_pred)
        }

class TabFactRobustTask(BaseTask):
    def __init__(self, data_path):
        super().__init__(data_path)
    
    def get_ground_truth(self, sample):
        if 'label' in sample:
            return "entailed" if sample['label'] == 1 else "refuted"
        
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'].lower()
                if "refuted" in content: return "refuted"
                if "entailed" in content: return "entailed"
        return "refuted" # 兜底

    def build_messages(self, sample):
        messages = sample.get('messages', [])
        user_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        if not user_content: return []
        
        # 可以在这里加一句引导，强化格式
        instruction = "\n\nAnalyze the statement based on the table. End your response with 'Answer: entailed' or 'Answer: refuted'."
        
        return [
            {"role": "user", "content": user_content + instruction}
        ]

    def post_process(self, prediction):
        """
        核心优化：不依赖固定格式，只要最后部分包含关键词即可
        """
        if not prediction: return "unknown"
        text = prediction.lower().strip()
        
        # 1. 优先找 Answer: 之后的
        if "answer:" in text:
            ans_part = text.split("answer:")[-1]
            if "entailed" in ans_part: return "entailed"
            if "refuted" in ans_part: return "refuted"
            
        # 2. 如果没找到 Answer:，或者 Answer 后没关键词，搜索全文的最后 50 个字符
        # 防止前面的推理中包含 "refuted" 但结论是 "entailed"
        last_part = text[-100:] 
        
        if "entailed" in last_part: return "entailed"
        if "refuted" in last_part: return "refuted"
        
        # 3. 兼容常见同义词 (防止模型变异)
        if "true" in last_part or "correct" in last_part: return "entailed"
        if "false" in last_part or "wrong" in last_part: return "refuted"

        return "unknown"

    def compute_metrics(self, gold, pred):
        clean_pred = self.post_process(pred)
        return {"accuracy": compute_accuracy(gold, clean_pred)}

def get_task(task_name, data_path):
    name = task_name.lower().strip()
    
    # === WTQ 系列 ===
    if name == "wtq": 
        return WTQTask(data_path)
    elif name == "wtq-cot":
        return WTQCoTTask(data_path)
    elif name == "wtq-completion":
        return WTQCompletionTask(data_path)
    # [新增] WTQ RL 任务
    elif name == "wtq_rl":
        return WTQRLTask(data_path)
        
    # === TabFact 系列 ===
    elif name == "tabfact": 
        return TabFactTask(data_path)
    elif name == "tabfact-completion": 
        return TabFactCompletionTask(data_path)
    elif name == "tabfact_rl":
        return TabFactRLTask(data_path)
        
    # === Feverous 系列 ===
    elif name == "feverous": 
        return FeverousCompletionTask(data_path)
    
    # === [推荐] 针对你的数据优化过的 Robust 任务 ===
    elif name == "wtq-robust":   # 对应你的 WTQ 数据
        return WTQRobustTask(data_path)
    elif name == "tabfact-robust": # 对应你的 TabFact 数据
        return TabFactRobustTask(data_path)
        
    raise ValueError(f"Unknown task: {task_name}")