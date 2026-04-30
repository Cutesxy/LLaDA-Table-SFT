import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from sacrebleu import sentence_chrf
except Exception:
    sentence_chrf = None

try:
    import bert_score
except Exception:
    bert_score = None


_METRIC_CACHE: Dict[Tuple[str, str, str], float] = {}
_BERT_SCORER = None


def _normalize_cell(text: str) -> str:
    text = str(text).replace("\\|", "|").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return "|" in s and s.count("|") >= 2


def _split_markdown_row(line: str) -> List[str]:
    s = line.strip()
    if not s.startswith("|"):
        s = "|" + s
    if not s.endswith("|"):
        s = s + "|"
    cells = [c.strip() for c in s[1:-1].split("|")]
    return cells


def _is_alignment_row(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    for c in cells:
        if not re.fullmatch(r":?-{2,}:?", c.strip()):
            return False
    return True


def _parse_table_block(lines: Sequence[str]) -> Optional[List[List[str]]]:
    rows: List[List[str]] = []
    for line in lines:
        if not _is_table_line(line):
            continue
        rows.append(_split_markdown_row(line))

    if not rows:
        return None

    if len(rows) >= 2 and _is_alignment_row(rows[1]):
        rows = [rows[0]] + rows[2:]

    if not rows:
        return None

    n_col = len(rows[0])
    if n_col == 0:
        return None

    fixed_rows: List[List[str]] = []
    for row in rows:
        if len(row) < n_col:
            row = row + [""] * (n_col - len(row))
        elif len(row) > n_col:
            row = row[:n_col]
        fixed_rows.append([_normalize_cell(x) for x in row])
    return fixed_rows


def _extract_all_markdown_tables(text: str) -> List[List[List[str]]]:
    lines = str(text).replace("\r\n", "\n").split("\n")
    blocks: List[List[str]] = []
    cur: List[str] = []

    for line in lines:
        if _is_table_line(line):
            cur.append(line)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    tables: List[List[List[str]]] = []
    for block in blocks:
        table = _parse_table_block(block)
        if table is not None:
            tables.append(table)
    return tables


def extract_table_by_name_markdown(text: str, table_name: Optional[str]) -> Optional[List[List[str]]]:
    text = str(text).replace("\r\n", "\n")
    if not table_name:
        tables = _extract_all_markdown_tables(text)
        return tables[0] if tables else None

    target = table_name.strip().lower()
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        heading = None
        if line.startswith("###"):
            heading = line[3:].strip().strip(":").lower()
        elif line.endswith(":"):
            heading = line[:-1].strip().lower()

        if heading and target in heading:
            j = i + 1
            while j < len(lines) and not _is_table_line(lines[j]):
                if lines[j].strip().startswith("###"):
                    break
                j += 1
            block: List[str] = []
            while j < len(lines) and _is_table_line(lines[j]):
                block.append(lines[j])
                j += 1
            table = _parse_table_block(block)
            if table is not None:
                return table
        i += 1

    tables = _extract_all_markdown_tables(text)
    return tables[0] if tables else None


def _is_empty_table(table: Optional[List[List[str]]], row_header: bool, col_header: bool) -> bool:
    if table is None or len(table) == 0:
        return True
    row = len(table)
    col = len(table[0]) if row > 0 else 0
    if row_header and col < 2:
        return True
    if col_header and row < 2:
        return True
    return False


def _table_to_sets(
    table: List[List[str]],
    row_header: bool,
    col_header: bool,
) -> Tuple[set, set, set]:
    if _is_empty_table(table, row_header=row_header, col_header=col_header):
        return set(), set(), set()

    row_headers = list(row[0] for row in table) if row_header else []
    col_headers = list(table[0]) if col_header else []

    if row_header and col_headers:
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    relations = []
    i_start = 1 if col_header else 0
    j_start = 1 if row_header else 0
    for i in range(i_start, len(table)):
        for j in range(j_start, len(table[0])):
            val = table[i][j]
            if val == "":
                continue
            rel = []
            if row_header:
                rel.append(table[i][0])
            if col_header:
                rel.append(table[0][j])
            rel.append(val)
            relations.append(tuple(rel))

    return set(row_headers), set(col_headers), set(relations)


def _cell_similarity(a: str, b: str, metric: str) -> float:
    key = (a, b, metric)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]

    if metric == "E":
        score = 1.0 if a == b else 0.0
    elif metric == "c":
        if sentence_chrf is not None:
            score = sentence_chrf(b, [a]).score / 100.0
        else:
            score = SequenceMatcher(None, a, b).ratio()
    elif metric == "BS-scaled":
        global _BERT_SCORER
        if bert_score is not None:
            if _BERT_SCORER is None:
                _BERT_SCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
            score = _BERT_SCORER.score([b], [a])[2].item()
            score = max(0.0, min(1.0, score))
        elif sentence_chrf is not None:
            score = sentence_chrf(b, [a]).score / 100.0
        else:
            score = SequenceMatcher(None, a, b).ratio()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    _METRIC_CACHE[key] = float(score)
    return float(score)


def _tuple_similarity(a: tuple, b: tuple, metric: str) -> float:
    if len(a) != len(b):
        return 0.0
    score = 1.0
    for aa, bb in zip(a, b):
        score *= _cell_similarity(str(aa), str(bb), metric)
    return score


def _calc_prf(tgt_items: Sequence, pred_items: Sequence, metric: str) -> Tuple[float, float, float]:
    if len(tgt_items) == 0 and len(pred_items) == 0:
        return 1.0, 1.0, 1.0
    if len(tgt_items) == 0 or len(pred_items) == 0:
        return 0.0, 0.0, 0.0

    tgt_list = list(tgt_items)
    pred_list = list(pred_items)
    sim = []
    for tgt in tgt_list:
        row = []
        for pred in pred_list:
            if isinstance(tgt, tuple):
                row.append(_tuple_similarity(tgt, pred, metric))
            else:
                row.append(_cell_similarity(str(tgt), str(pred), metric))
        sim.append(row)

    pred_best = [max(sim_i[j] for sim_i in sim) for j in range(len(pred_list))]
    tgt_best = [max(row) for row in sim]
    precision = sum(pred_best) / len(pred_best)
    recall = sum(tgt_best) / len(tgt_best)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_text2table_metrics(
    pred_text: str,
    gold_text: str,
    row_header: bool,
    col_header: bool,
    table_name: Optional[str] = None,
    metric: str = "E",
    prefix: str = "",
) -> Dict[str, float]:
    pred_table = extract_table_by_name_markdown(pred_text, table_name)
    gold_table = extract_table_by_name_markdown(gold_text, table_name)

    if _is_empty_table(gold_table, row_header=row_header, col_header=col_header):
        # Target empty: skip this sample by returning NaN-like sentinel.
        return {f"{prefix}valid": 0.0}

    out: Dict[str, float] = {f"{prefix}valid": 1.0}
    wrong_format = 1.0 if pred_table is None else 0.0
    out[f"{prefix}wrong_format"] = wrong_format

    if pred_table is None or _is_empty_table(pred_table, row_header=row_header, col_header=col_header):
        if row_header:
            out[f"{prefix}row_precision"] = 0.0
            out[f"{prefix}row_recall"] = 0.0
            out[f"{prefix}row_f1"] = 0.0
        if col_header:
            out[f"{prefix}col_precision"] = 0.0
            out[f"{prefix}col_recall"] = 0.0
            out[f"{prefix}col_f1"] = 0.0
        out[f"{prefix}cell_precision"] = 0.0
        out[f"{prefix}cell_recall"] = 0.0
        out[f"{prefix}cell_f1"] = 0.0
        return out

    pred_row, pred_col, pred_rel = _table_to_sets(pred_table, row_header=row_header, col_header=col_header)
    gold_row, gold_col, gold_rel = _table_to_sets(gold_table, row_header=row_header, col_header=col_header)

    if row_header:
        p, r, f = _calc_prf(gold_row, pred_row, metric=metric)
        out[f"{prefix}row_precision"] = p
        out[f"{prefix}row_recall"] = r
        out[f"{prefix}row_f1"] = f
    if col_header:
        p, r, f = _calc_prf(gold_col, pred_col, metric=metric)
        out[f"{prefix}col_precision"] = p
        out[f"{prefix}col_recall"] = r
        out[f"{prefix}col_f1"] = f

    p, r, f = _calc_prf(gold_rel, pred_rel, metric=metric)
    out[f"{prefix}cell_precision"] = p
    out[f"{prefix}cell_recall"] = r
    out[f"{prefix}cell_f1"] = f
    return out

