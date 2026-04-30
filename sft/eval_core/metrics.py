# eval_core/metrics.py
import re
import string
import collections

# ----------------------------
# 基础清洗工具
# ----------------------------
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

# ----------------------------
# 通用指标计算函数
# ----------------------------
def compute_exact_match(a_gold, a_pred):
    """计算 EM (Exact Match)"""
    return 1 if normalize_answer(a_gold) == normalize_answer(a_pred) else 0

def compute_f1(a_gold, a_pred):
    """计算 Bag-of-Words F1 Score"""
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0: return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_accuracy(a_gold, a_pred):
    """计算简单的 Accuracy (适用于分类任务如 TabFact)"""
    # 这里做简单的全等判断，也可以加上 normalize
    return 1 if str(a_gold).lower().strip() == str(a_pred).lower().strip() else 0