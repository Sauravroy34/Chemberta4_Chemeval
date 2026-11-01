#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Evaluation script for ChemEval MULTIMODAL benchmark.

Reads model-output JSONL files (one per run), routes each row to the correct
scorer by its file_path, and reports per-task and aggregate metrics.

Tasks are grouped by evaluation method:
  - Code-graded: classification, regression, molecule conversion, SIDER,
                  entity extraction, reagent selection, etc.
  - LLM-judged:  fill-in-blank, short answer, calculation, multiple choice,
                  pathway recommendation, intermediate derivation, spectroscopy
                  description tasks, etc.

Usage:
    python evaluate.py --inputs run1.jsonl --skip-judge --out res   # code-graded only
    python evaluate.py --inputs run1.jsonl --judge-model gpt-4o --out res
    python evaluate.py --inputs run1.jsonl --inspect                # routing check
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


from dotenv import load_dotenv

load_dotenv(override=True)
LANGFUSE = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    LANGFUSE = get_client()
else:
    print("WARNING: Langfuse keys not set - nothing will be logged to Langfuse")

HERE = Path(__file__).resolve().parent
# Re-use repo_constants.json from text_eval (same judge templates)
TEXT_EVAL = HERE.parent / "text_eval"
log = logging.getLogger("chemeval_mm")


# ---------------------------------------------------------------------------
# 0.  Task routing — map file_path -> evaluation category
# ---------------------------------------------------------------------------
# Each entry: (category, regex matching the file_path)
ROUTING_RULES = [
    # --- Classification (accuracy) ---
    ("classification", r"BBBP|ClinTox|HIV|极性"),
    # --- Regression (NRMSE etc.) ---
    ("regression", r"ESOL|HOMO|LUMO|Lipo|沸点|熔点|合成难度评估"),
    # --- Regression with mean-of-numbers extraction ---
    ("regression_avg", r"反应温度推荐|反应时间推荐"),
    # --- Yield prediction (classification: Yes/No -> high/low yield) ---
    ("classification", r"产物产率预测"),
    # --- Molecule conversion: SMILES-based (Tanimoto) ---
    ("molecule_smiles", r"SELFIES转SMILES|IUPACE?转SMILES|2D分子识别"),
    ("molecule_smiles_to_selfies", r"SMILES转SELFIES"),
    # --- Molecule conversion: formula (L2/L1/cosine) ---
    ("molecule_formula", r"SMILES转分子式|IUPAC转分子式"),
    # --- Molecule conversion: IUPAC (BLEU/edit distance) ---
    ("molecule_iupac", r"SMILES转IUPAC"),
    # --- SIDER (per-label accuracy) ---
    ("sider", r"SIDER"),
    # --- Reagent / Catalyst / Ligand / Solvent / Substrate / Product prediction (F1) ---
    ("reagent", r"催化剂推荐|配体推荐|溶剂推荐|试剂推荐|反应底物推荐|反应产物预测|合成路径解析"),
    # --- Judge: Multiple choice ---
    ("judge_choice", r"选择题"),
    # --- Judge: Fill-in-blank ---
    ("judge_fillblank", r"填空题"),
    # --- Judge: Short answer ---
    ("judge_shortanswer", r"简答题"),
    # --- Judge: Calculation ---
    ("judge_calculation", r"计算题"),
    # --- Judge: Single-step pathway ---
    ("judge_single_step", r"单步合成路径推荐"),
    # --- Judge: Multi-step pathway ---
    ("judge_multi_step", r"多步合成路径推荐"),
    # --- Judge: Intermediate derivation ---
    ("judge_intermediate", r"反应中间体推导"),
    # --- Judge: Spectroscopy / diagram description ---
    ("judge_description", r"红外光谱|拉曼光谱|紫外可见|XRD|衍射谱图|质谱图|核磁图谱|状态图|势能图|动力学行为图|描述分子的物理化学性质"),
    # --- Judge: Molecular image recognition ---
    ("judge_description", r"化学分子式_|化学反应_|化学多模态图表解析|统计图问答"),
]


def route(file_path: str) -> Optional[str]:
    """Return the evaluation category for a given file_path."""
    for cat, pat in ROUTING_RULES:
        if re.search(pat, file_path):
            return cat
    return None


# Paper-level task code for reporting
TASK_CODES = [
    ("MCTask", r"选择题"),
    ("FBTask", r"填空题"),
    ("SATask", r"简答题"),
    ("CalcTask", r"计算题"),
    ("MolPC_BBBP", r"BBBP"), ("MolPC_ClinTox", r"ClinTox"),
    ("MolPC_HIV", r"HIV"), ("MolPC_Polar", r"极性"),
    ("MolPR_ESOL", r"ESOL"), ("MolPR_HOMO", r"HOMO"),
    ("MolPR_LUMO", r"LUMO"), ("MolPR_Lipo", r"Lipo"),
    ("MolPR_BP", r"沸点"), ("MolPR_MP", r"熔点"),
    ("SynDE", r"合成难度评估"),
    ("YPred", r"产物产率预测"),
    ("SIDER", r"SIDER"),
    ("TempRec", r"反应温度推荐"), ("TimeRec", r"反应时间推荐"),
    ("IUPAC2SMILES", r"IUPACE?转SMILES"),
    ("S2MF", r"SMILES转分子式"), ("I2MF", r"IUPAC转分子式"),
    ("S2IUPAC", r"SMILES转IUPAC"),
    ("SELFIES2S", r"SELFIES转SMILES"),
    ("S2SELFIES", r"SMILES转SELFIES"),
    ("CatRec", r"催化剂推荐"), ("LRec", r"配体推荐"),
    ("SolvRec", r"溶剂推荐"), ("RRec", r"试剂推荐"),
    ("SubRec", r"反应底物推荐"), ("PPred", r"反应产物预测"),
    ("PathRec1", r"单步合成路径推荐"), ("PathRecN", r"多步合成路径推荐"),
    ("IMDer", r"反应中间体推导"),
    ("MolRec", r"2D分子识别"),
    ("PathParse", r"合成路径解析"),
    ("IR", r"红外光谱"), ("Raman", r"拉曼光谱"),
    ("UV", r"紫外可见"), ("XRD", r"XRD|衍射谱图"),
    ("MS", r"质谱图"), ("NMR", r"核磁图谱"),
    ("StateDiag", r"状态图"), ("PES", r"势能图"),
    ("Kinetics", r"动力学行为图"),
    ("MolFormula", r"化学分子式_"),
    ("ChemRxn", r"化学反应_"),
    ("ChartQA", r"化学多模态图表解析"),
    ("StatChart", r"统计图问答"),
    ("DiffSpec", r"基于衍射谱图"), ("MassSpec", r"基于质谱图"),
    ("NMRSpec", r"基于核磁图谱"),
]
TASK_ORDER = [t for t, _ in TASK_CODES]


def paper_task(file_path: str) -> str:
    for code, pat in TASK_CODES:
        if re.search(pat, file_path):
            return code
    return Path(file_path).stem


# ---------------------------------------------------------------------------
# 1.  Answer cleaning and extraction (shared with text_eval logic)
# ---------------------------------------------------------------------------
def repo_preprocess(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, (list, tuple)):
        ans = ans[0] if ans else None
    if isinstance(ans, dict):
        ans = json.dumps(ans, ensure_ascii=False)
    if not isinstance(ans, str):
        ans = str(ans)
    ans = ans.replace("<ret>", "\n").replace("<end>", "")
    i = ans.find("</think>")
    if i != -1:
        ans = ans[i + len("</think>"):]
    elif "<think>" in ans:
        return None
    return ans


def clean_model_text(ans: Any) -> Optional[str]:
    t = repo_preprocess(ans)
    if t is None:
        return None
    t = re.sub(r"```[A-Za-z0-9_+-]*", "", t)
    t = t.strip()
    return t or None


def _balanced_objects(text: str, max_starts: int = 200):
    n, starts = len(text), 0
    i = text.find("{")
    while i != -1 and starts < max_starts:
        starts += 1
        depth, quote, esc, end = 0, None, False, -1
        for j in range(i, n):
            c = text[j]
            if quote:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    quote = None
            elif c == '"':
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end != -1:
            yield text[i:end + 1]
        i = text.find("{", i + 1)


def _parse_obj(s: str):
    cands = [s]
    t = (s.replace("{{", "{").replace("}}", "}").replace("\u201c", '"').replace("\u201d", '"')
         .replace("\u2018", "'").replace("\u2019", "'"))
    t = re.sub(r",\s*([}\]])", r"\1", t)
    cands.append(t)
    for c in cands:
        for fn in (json.loads, ast.literal_eval):
            try:
                return fn(c)
            except Exception:
                pass
    return None


_PLACEHOLDER = re.compile(
    r"^\s*(the answer you judge|your answer|answer here|valid\s+(smiles|selfies|iupac)\b.*|\.\.\.)\s*$", re.I)


def _is_placeholder(v: Any) -> bool:
    if isinstance(v, str):
        return bool(_PLACEHOLDER.match(v))
    if isinstance(v, list) and v:
        return all(_is_placeholder(x) for x in v)
    return False


def extract_answer_value(text: Optional[str]):
    if not text:
        return None, "none"
    best = None
    for s in _balanced_objects(text):
        obj = _parse_obj(s)
        if isinstance(obj, dict):
            key = next((k for k in obj if str(k).strip().lower() == "answer"), None)
            if key is not None and not _is_placeholder(obj[key]):
                best = obj[key]
    if best is not None:
        return best, "json"
    ms = [m for m in re.finditer(r'["\']?answer["\']?\s*[:：]\s*"(.*?)"\s*[,}]', text, re.I | re.S)
          if not _is_placeholder(m.group(1))]
    if ms:
        return ms[-1].group(1), "regex"
    ms = list(re.finditer(r'["\']?answer["\']?\s*[:：]\s*(\[.*?\])', text, re.I | re.S))
    if ms:
        raw = ms[-1].group(1)
        try:
            return ast.literal_eval(raw), "regex"
        except Exception:
            return [x.strip(" \"'") for x in raw[1:-1].split(",") if x.strip(" \"'")], "regex"
    ms = list(re.finditer(r"\banswer\s*(?:is|are|:|：)\s*(.+)", text, re.I))
    if ms:
        return ms[-1].group(1).strip().strip("\"'`*. "), "phrase"
    return text.strip(), "raw"


def get_value(raw_answer: Any):
    return extract_answer_value(clean_model_text(raw_answer))


def value_to_text(v: Any, sep: str = ", ") -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v
    elif isinstance(v, (list, tuple)):
        s = sep.join(x for x in (value_to_text(i, sep) for i in v) if x)
    elif isinstance(v, dict):
        s = sep.join(x for x in (value_to_text(i, sep) for i in v.values()) if x)
    else:
        s = str(v)
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s or None


def first(g: Any) -> Any:
    return g[0] if isinstance(g, (list, tuple)) and g else g


def last_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace("−", "-")
    m = re.search(r"[-+]?\d*\.?\d+(?=\D*$)", s)
    return float(m.group()) if m else None


def numbers_avg(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d)\s*[-~]\s*(?=\d)", " ", s)
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", s)]
    return sum(nums) / len(nums) if nums else None


def gold_number(g: Any) -> Optional[float]:
    g = first(g)
    if g is None:
        return None
    s = str(g).replace("−", "-").strip()
    m = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*(?:~|–|to|-)\s*(\d*\.?\d+)\s*", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return last_number(s)


# ---------------------------------------------------------------------------
# 2.  Chemistry helpers (lazy imports)
# ---------------------------------------------------------------------------
_RD: dict = {}
SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def rd():
    if not _RD:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        _RD.update(Chem=Chem, DS=DataStructs,
                   gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    return _RD


def canon_smiles(smiles: str) -> Optional[str]:
    Chem = rd()["Chem"]
    out = []
    for part in smiles.split(";"):
        if part == "":
            return None
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        for a in mol.GetAtoms():
            a.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True))
    return ".".join(sorted(out))


def tanimoto(a: str, b: str) -> float:
    R = rd()
    ma, mb = R["Chem"].MolFromSmiles(a), R["Chem"].MolFromSmiles(b)
    if ma is None or mb is None:
        return 0.0
    return R["DS"].TanimotoSimilarity(R["gen"].GetFingerprint(ma), R["gen"].GetFingerprint(mb))


def selfies_to_smiles(s: str) -> Optional[str]:
    import selfies as sf
    try:
        out = sf.decoder(s)
        return out or None
    except Exception:
        return None


def smiles_candidate(v: Any, how: str) -> Optional[str]:
    t = value_to_text(v, ".")
    if not t:
        return None
    t = re.sub(r"</?SMILES>", "", t, flags=re.I).strip()
    if how in ("raw", "phrase") and re.search(r"\s", t):
        Chem = rd()["Chem"]
        toks = [x.strip(".,;:()\"\\'`*") for x in re.split(r"\s+", t)]
        ok = [x for x in toks if len(x) > 1 and Chem.MolFromSmiles(x) is not None]
        if ok:
            return ok[-1]
    return t.replace(" ", "").replace("\n", "") or None


# ---------------------------------------------------------------------------
# 3.  Data containers
# ---------------------------------------------------------------------------
@dataclass
class Item:
    idx: int
    query: str
    gold: Any
    answer: Any
    file_path: str


@dataclass
class FileScore:
    n: int
    metrics: dict
    item_vals: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    category: str = ""
    task: str = ""


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _json_rate(hows):
    return sum(h == "json" for h in hows) / len(hows) if hows else 0.0


# ---------------------------------------------------------------------------
# 4.  Code-graded scorers
# ---------------------------------------------------------------------------
def norm_yes_no(txt: Optional[str], how: str) -> Optional[str]:
    if not txt:
        return None
    t = txt.lower().replace("正确", "yes").replace("错误", "no")
    if how in ("raw", "phrase"):
        m = re.match(r"\W*(yes|no)\b", t)
        if m:
            return m.group(1).capitalize()
    hits = set(re.findall(r"\b(yes|no)\b", t))
    if len(hits) == 1:
        return hits.pop().capitalize()
    return None


def s_classification(items, ctx):
    corr, valid, rows, hows = [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v)
        gold = str(first(it.gold)).replace("正确", "Yes").replace("错误", "No").strip()
        pred = norm_yes_no(txt, how)
        ok = pred is not None and pred.lower() == gold.lower()
        corr.append(1.0 if ok else 0.0)
        valid.append(1.0 if pred else 0.0)
        rows.append({"idx": it.idx, "pred": pred, "gold": gold, "correct": ok})
    return FileScore(len(items), {"accuracy": _mean(corr), "valid": _mean(valid),
                                  "json_rate": _json_rate(hows)},
                     {"accuracy": corr, "valid": valid}, rows)


def _reg_stats(g, p):
    g, p = list(g), list(p)
    out = {"nrmse": None, "rmse": None, "rae": None, "r2": None}
    if not g:
        return out
    n = len(g)
    mg = sum(g) / n
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(g, p)) / n)
    rng = max(g) - min(g)
    base = sum(abs(a - mg) for a in g)
    tss = sum((a - mg) ** 2 for a in g)
    out["rmse"] = rmse
    out["nrmse"] = rmse / rng if rng else None
    out["rae"] = sum(abs(a - b) for a, b in zip(g, p)) / base if base else None
    out["r2"] = 1 - sum((a - b) ** 2 for a, b in zip(g, p)) / tss if tss else None
    return out


def s_regression(items, ctx, use_avg=False):
    g_ok, p_ok, valid, rows, hows = [], [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v, " ")
        pred = numbers_avg(txt) if use_avg else last_number(txt)
        g = gold_number(it.gold)
        valid.append(0.0 if pred is None else 1.0)
        if pred is not None and g is not None:
            g_ok.append(g)
            p_ok.append(pred)
        rows.append({"idx": it.idx, "pred": pred, "gold": g})
    m = _reg_stats(g_ok, p_ok)
    m.update(valid=_mean(valid), json_rate=_json_rate(hows))
    return FileScore(len(items), m, {"valid": valid}, rows)


def s_smiles(items, ctx):
    sims, valid, rows, hows = [], [], [], []
    for it in items:
        gold = str(first(it.gold)).replace(" ", "").replace(";", ".")
        v, how = get_value(it.answer)
        hows.append(how)
        pred = smiles_candidate(v, how)
        gold_c = canon_smiles(gold) if gold else None
        if gold_c is None:
            rows.append({"idx": it.idx, "skipped": "gold not parseable"})
            continue
        pred_c = canon_smiles(pred) if pred else None
        sim = tanimoto(pred_c, gold_c) if pred_c else 0.0
        sims.append(sim)
        valid.append(1.0 if pred_c else 0.0)
        rows.append({"idx": it.idx, "pred": pred_c, "gold": gold_c, "tanimoto": sim})
    return FileScore(len(sims), {"tanimoto": _mean(sims), "valid": _mean(valid),
                                  "json_rate": _json_rate(hows)},
                     {"tanimoto": sims, "valid": valid}, rows)


def s_smiles_to_selfies(items, ctx):
    """SMILES -> SELFIES: convert both to SMILES then compare with Tanimoto."""
    sims, valid, rows, hows = [], [], [], []
    for it in items:
        gold = str(first(it.gold)).replace(" ", "")
        gold_smi = selfies_to_smiles(gold)
        v, how = get_value(it.answer)
        hows.append(how)
        pred_raw = smiles_candidate(v, how)
        pred_smi = selfies_to_smiles(pred_raw) if pred_raw else None
        gold_c = canon_smiles(gold_smi) if gold_smi else None
        if gold_c is None:
            rows.append({"idx": it.idx, "skipped": "gold SELFIES not parseable"})
            continue
        pred_c = canon_smiles(pred_smi) if pred_smi else None
        sim = tanimoto(pred_c, gold_c) if pred_c else 0.0
        sims.append(sim)
        valid.append(1.0 if pred_c else 0.0)
        rows.append({"idx": it.idx, "pred": pred_c, "gold": gold_c, "tanimoto": sim})
    return FileScore(len(sims), {"tanimoto": _mean(sims), "valid": _mean(valid),
                                  "json_rate": _json_rate(hows)},
                     {"tanimoto": sims, "valid": valid}, rows)


def parse_formula(s: str) -> dict:
    d: dict = defaultdict(int)
    for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", s):
        d[el] += int(cnt) if cnt else 1
    return dict(d)


def s_formula(items, ctx):
    import numpy as np
    l2s, l1s, coss, valid, rows, hows = [], [], [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v, "")
        if txt and how in ("raw", "phrase"):
            c = re.findall(r"\b(?:[A-Z][a-z]?\d*){2,}\b|\b[A-Z][a-z]?\d+\b", txt.translate(SUBS))
            txt = c[-1] if c else txt
        pred = txt.translate(SUBS) if txt else None
        gv = parse_formula(str(first(it.gold)).translate(SUBS))
        pv = parse_formula(pred) if pred else None
        if not pv:
            l2s.append(0.0); l1s.append(0.0); coss.append(0.0); valid.append(0.0)
        else:
            atoms = sorted(set(pv) | set(gv))
            a = np.array([pv.get(x, 0) for x in atoms], float)
            b = np.array([gv.get(x, 0) for x in atoms], float)
            l2s.append(1 / (1 + float(np.linalg.norm(a - b))))
            l1s.append(1 / (1 + float(np.abs(a - b).sum())))
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            coss.append(float(a @ b) / den if den else 0.0)
            valid.append(1.0)
        rows.append({"idx": it.idx, "pred": pred, "gold": first(it.gold), "l2": l2s[-1]})
    return FileScore(len(items), {"l2": _mean(l2s), "l1": _mean(l1s), "cosine": _mean(coss),
                                  "valid": _mean(valid), "json_rate": _json_rate(hows)},
                     {"l2": l2s, "l1": l1s, "cosine": coss, "valid": valid}, rows)


def s_iupac(items, ctx):
    import Levenshtein
    from nltk.tokenize.destructive import NLTKWordTokenizer
    from nltk.translate.bleu_score import sentence_bleu
    tok = NLTKWordTokenizer().tokenize
    ws = lambda s: re.sub(r"\s+", " ", s).strip()
    em, bl, ed, rows, hows = [], [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        pred = value_to_text(v, ";")
        pred = ws(pred) if pred else None
        gold = ws(str(first(it.gold)))
        if not pred:
            em.append(0.0); bl.append(0.0); ed.append(float(len(gold)))
        else:
            em.append(1.0 if pred.lower() == gold.lower() else 0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bl.append(sentence_bleu([tok(gold.lower())], tok(pred.lower()), weights=(0.25,) * 4))
            ed.append(float(Levenshtein.distance(gold, pred)))
        rows.append({"idx": it.idx, "pred": pred, "gold": gold})
    return FileScore(len(items), {"exact_match": _mean(em), "bleu": _mean(bl),
                                  "edit_distance": _mean(ed), "json_rate": _json_rate(hows)},
                     {"exact_match": em, "bleu": bl, "edit_distance": ed}, rows)


def s_sider(items, ctx):
    labels = ctx.consts.get("sider_labels", [])
    tot_c, tot_n, accs, rows, hows = 0, 0, [], [], []
    for it in items:
        gold = it.gold
        if isinstance(gold, str):
            gold = _parse_obj(gold) or {}
        v, how = get_value(it.answer)
        pred = v if isinstance(v, dict) else None
        if pred is None:
            for s in _balanced_objects(clean_model_text(it.answer) or ""):
                o = _parse_obj(s)
                if isinstance(o, dict) and any(k in o for k in labels):
                    pred = o
        hows.append("json" if pred is not None else "none")
        norm = lambda x: str(x).strip().lower()
        if isinstance(gold, dict):
            c = sum(1 for k, gv in gold.items()
                    if pred is not None and k in pred and norm(pred[k]) == norm(gv))
            n = len(gold) or 1
        else:
            c, n = 0, 1
        tot_c += c; tot_n += n
        accs.append(c / n)
        rows.append({"idx": it.idx, "correct": c, "of": n})
    return FileScore(len(items), {"accuracy": tot_c / tot_n if tot_n else None,
                                  "json_rate": _json_rate(hows)},
                     {"accuracy": accs}, rows)


def _f1_sets(t, p):
    t, p = set(t), set(p)
    tp = len(t & p)
    pr = tp / len(p) if p else 0
    rc = tp / len(t) if t else 0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            r = ast.literal_eval(x)
            return r if isinstance(r, list) else None
        except Exception:
            return None
    return None


def s_reagent(items, ctx):
    Chem = rd()["Chem"]
    f1s, rows, hows = [], [], []

    def canon_frag(m):
        mol = Chem.MolFromSmiles(m) if m else None
        return Chem.MolToSmiles(mol) if mol is not None else None

    for it in items:
        g = first(it.gold) if not isinstance(it.gold, list) else it.gold
        if isinstance(g, str):
            try:
                g = ast.literal_eval(g)
            except Exception:
                pass
        gs = ".".join(map(str, g)) if isinstance(g, list) else str(g)
        gs = gs.replace(" ", "").replace("and", ".").replace("\n", "").replace(">", ".")
        mol = Chem.MolFromSmiles(gs)
        if mol is not None:
            gold_set = set(Chem.MolToSmiles(mol).split("."))
            exact = True
        else:
            gold_set = set(x for x in gs.split(".") if x)
            exact = False
        v, how = get_value(it.answer)
        hows.append(how)
        lst = _as_list(v) if isinstance(v, str) and v.strip().startswith("[") else v
        ptxt = ".".join(map(str, lst)) if isinstance(lst, list) else (value_to_text(lst, ".") or "")
        ptxt = ptxt.replace(" ", "").replace("and", ".").replace("\n", "").replace(">", ".")
        pset = set()
        for frag in ptxt.split("."):
            if not frag:
                continue
            c = canon_frag(frag)
            if c is not None:
                pset.add(c)
            elif not exact:
                pset.add(frag)
        s = _f1_sets(gold_set, pset)
        f1s.append(s)
        rows.append({"idx": it.idx, "pred": sorted(pset), "gold": sorted(gold_set), "f1": s})
    return FileScore(len(items), {"f1": _mean(f1s), "json_rate": _json_rate(hows)},
                     {"f1": f1s}, rows)


CODE_SCORERS = {
    "classification": s_classification,
    "regression": lambda items, ctx: s_regression(items, ctx, use_avg=False),
    "regression_avg": lambda items, ctx: s_regression(items, ctx, use_avg=True),
    "molecule_smiles": s_smiles,
    "molecule_smiles_to_selfies": s_smiles_to_selfies,
    "molecule_formula": s_formula,
    "molecule_iupac": s_iupac,
    "sider": s_sider,
    "reagent": s_reagent,
}


# ---------------------------------------------------------------------------
# 5.  LLM-as-judge
# ---------------------------------------------------------------------------
JUDGE_TEMPLATE_MAP = {
    "judge_choice": "template_choice",
    "judge_fillblank": "template_fillblank",
    "judge_shortanswer": "template_score",
    "judge_calculation": "template_calculate",
    "judge_single_step": "template_danbulujing",
    "judge_multi_step": "template_duobulujing",
    "judge_intermediate": "template_fanyingzhongjianti",
    "judge_description": "template_xingzhi",  # general expert scoring
}


def fillblank_question(query: str) -> str:
    for marker in ("接下来是需要你回答的问题：", "Please answer the following questions:"):
        pos = query.rfind(marker)
        if pos != -1:
            return query[pos:].replace(marker, "")
    return query.replace("请回答下列问题\n", "")


def build_judge_prompt(cat: str, it: Item, ctx) -> str:
    T = ctx.consts["templates"]
    ans = repo_preprocess(it.answer)
    ans = "None" if ans is None else ans
    gold = first(it.gold)
    gold = gold if isinstance(gold, str) else str(gold)
    tpl_key = JUDGE_TEMPLATE_MAP[cat]
    t = T[tpl_key]
    if cat == "judge_choice":
        return t.format(ans)
    if cat in ("judge_fillblank",):
        return t.format(fillblank_question(it.query or ""), ans, gold)
    # All other templates: {answer} {gold}
    return t.format(ans, gold)


class Judge:
    def __init__(self, model, cache_path, workers=8, max_tokens=2048, temperature=0.0, retries=6):
        self.model, self.workers, self.max_tokens = model, workers, max_tokens
        self.temperature, self.retries = temperature, retries
        self.cache_path, self.cache, self.lock = Path(cache_path), {}, threading.Lock()
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r["text"]
                except Exception:
                    pass

    def _key(self, prompt):
        return hashlib.sha256((self.model + "\x00" + prompt).encode("utf-8")).hexdigest()

    def _one(self, prompt):
        import litellm
        litellm.drop_params = True

        litellm.success_callback = ["langfuse"] if LANGFUSE else []
        litellm.failure_callback = ["langfuse"] if LANGFUSE else []
        for a in range(self.retries):
            try:
                r = litellm.completion(model=self.model,
                                       messages=[{"role": "user", "content": prompt}],
                                       temperature=self.temperature, max_tokens=self.max_tokens)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                log.warning("judge error %d/%d: %s", a + 1, self.retries, str(e)[:200])
                time.sleep(min(2 ** a, 30))
        return None

    def run(self, prompts):
        todo = {}
        for p in prompts:
            k = self._key(p)
            if k not in self.cache and k not in todo:
                todo[k] = p
        if todo:
            log.info("judge: %d new calls (%d cached)", len(todo), len(prompts) - len(todo))

            def work(kp):
                k, p = kp
                text = self._one(p)
                if text is not None:
                    with self.lock:
                        self.cache[k] = text
                        with open(self.cache_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": k, "text": text}, ensure_ascii=False) + "\n")

            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(work, todo.items()))
        return [self.cache.get(self._key(p)) for p in prompts]


def _after(text: str, *tags) -> str:
    best = -1
    for t in tags:
        i = text.rfind(t)
        if i != -1 and i + len(t) > best:
            best = i + len(t)
    return text[best:] if best != -1 else text


def parse_score(text: Optional[str], lo: float, hi: float) -> Optional[float]:
    if not text:
        return None
    seg = _after(text, "<最终得分>")
    nums = re.findall(r"(?<![\d.])\d+(?:\.\d+)?", seg)
    if not nums:
        return None
    v = float(nums[0] if "<最终得分>" in text else nums[-1])
    return v if lo <= v <= hi else None


def parse_choice(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    seg = _after(text, "<最终选项>", "<最终答案>").replace("*", " ")
    if re.fullmatch(r"\W*none\W*", seg, re.I):
        return ""
    return "".join(sorted(set(re.findall(r"[A-Z]", seg))))


def norm_gold_choice(g: Any) -> str:
    return "".join(sorted(set(re.findall(r"[A-Z]", str(first(g))))))


def parse_tf(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    seg = _after(text, "<最终答案>", "<最终选项>").replace("不正确", "错误")
    t, f = "正确" in seg, "错误" in seg
    return None if (t == f) else ("正确" if t else "错误")


def rule_extract(cat: str, raw: Any) -> str:
    """Deterministic alternative to the judge for choice (--mc-tf-extractor rule)."""
    v, how = get_value(raw)
    t = value_to_text(v, "") or ""
    if cat == "judge_choice":
        return "".join(sorted(set(re.findall(r"[A-Z]", t)))) if re.fullmatch(r"[A-Z,、\s和and]+", t.strip()) or how == "json" else ""
    return t


def score_judged(cat: str, items: list, outs: list, ctx) -> FileScore:
    vals, rows = [], []
    unparsed = 0
    rule = getattr(ctx, "mc_tf_extractor", "judge") == "rule" and cat == "judge_choice"
    for it, out in zip(items, outs):
        if cat == "judge_choice":
            pred = rule_extract(cat, it.answer) if rule else parse_choice(out)
            if pred is None:
                unparsed += 1
            vals.append(1.0 if pred is not None and pred == norm_gold_choice(it.gold) else 0.0)
            rows.append({"idx": it.idx, "judge_extracted": pred,
                         "gold": norm_gold_choice(it.gold), "correct": vals[-1]})
        elif cat == "judge_fillblank":
            s = parse_score(out, 0.0, 1.0)
            if s is None:
                unparsed += 1
            vals.append(s)
            rows.append({"idx": it.idx, "judge_score": s, "judge_output": out})
        else:
            # All other judge tasks: 1-5 scale -> normalise to 0-1
            s = parse_score(out, 1.0, 5.0)
            if s is None:
                unparsed += 1
            norm = None if s is None else (s - 1) / 4
            if norm is None:
                norm = 0.0  # missing -> minimum score
            vals.append(norm)
            rows.append({"idx": it.idx, "judge_score_1_5": s, "judge_output": out})
    name = "accuracy" if cat == "judge_choice" else "score"
    return FileScore(len(items), {name: _mean(vals), "unparsed": unparsed}, {name: vals}, rows,
                     [f"{unparsed} judge outputs could not be parsed"] if unparsed else [])


# ---------------------------------------------------------------------------
# 6.  Evaluation orchestrator
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    consts: dict
    mc_tf_extractor: str = "judge"


def load_run(path: Path, limit=None):
    """Load JSONL -> dict[file_path -> list[Item]]"""
    by_file: dict = defaultdict(list)
    n = empty = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fp = r.get("file_path") or ""
            lst = by_file[fp]
            if limit and len(lst) >= limit:
                continue
            ans = r.get("answer")
            empty += 1 if (ans is None or (isinstance(ans, str) and not ans.strip())) else 0
            lst.append(Item(len(lst), r.get("query", ""), r.get("target"), ans, fp))
            n += 1
    if empty:
        log.warning("%s: %d/%d rows have an empty answer", path.name, empty, n)
    return by_file


def evaluate_run(by_file: dict, ctx: Ctx, judge: Optional[Judge], dump_prompts=None):
    results: dict = {}
    judge_jobs = []  # (file_path, cat, items)

    for fp, items in by_file.items():
        cat = route(fp)
        if cat is None:
            log.warning("UNROUTED file_path skipped: %s", fp)
            continue

        if cat in CODE_SCORERS:
            try:
                fs = CODE_SCORERS[cat](items, ctx)
            except Exception as e:
                log.exception("scoring failed for %s: %s", fp, e)
                continue
            fs.category, fs.task = cat, paper_task(fp)
            results[fp] = fs
        else:
            # It's a judge task
            judge_jobs.append((fp, cat, items))

    if judge_jobs:
        # Build all judge prompts
        all_prompts = []
        for fp, cat, items in judge_jobs:
            prompts = [build_judge_prompt(cat, it, ctx) for it in items]
            all_prompts.append((fp, cat, items, prompts))

        if dump_prompts is not None:
            for fp, cat, items, ps in all_prompts:
                for it, p in zip(items, ps):
                    dump_prompts.write(json.dumps({"file_path": fp, "cat": cat, "idx": it.idx,
                                                   "query": p, "target": str(first(it.gold))}, ensure_ascii=False) + "\n")

        outs_map = {}
        if judge is not None:
            flat = [p for _, _, _, ps in all_prompts for p in ps]
            outs = judge.run(flat)
            pos = 0
            for fp, cat, items, ps in all_prompts:
                outs_map[(fp, cat)] = outs[pos:pos + len(ps)]
                pos += len(ps)

        for fp, cat, items, ps in all_prompts:
            rule = ctx.mc_tf_extractor == "rule" and cat == "judge_choice"
            if judge is None and not rule:
                continue
            item_outs = [None] * len(items) if rule else outs_map.get((fp, cat), [None] * len(items))
            fs = score_judged(cat, items, item_outs, ctx)
            fs.category, fs.task = cat, paper_task(fp)
            results[fp] = fs

    return results


# ---------------------------------------------------------------------------
# 7.  Aggregation and reporting
# ---------------------------------------------------------------------------
PCT = {"accuracy", "f1", "tanimoto", "score", "overlap", "exact_match", "bleu", "valid", "json_rate"}


def scale(metric, v):
    return None if v is None else (v * 100 if metric in PCT else v)


def task_level(file_scores: list) -> dict:
    out, n = {}, sum(fs.n for fs in file_scores)
    names = []
    for fs in file_scores:
        for m in fs.metrics:
            if m not in names:
                names.append(m)
    for m in names:
        if all(m in fs.item_vals for fs in file_scores):
            out[m] = _mean([x for fs in file_scores for x in fs.item_vals[m]])
        elif m == "json_rate":
            out[m] = sum(fs.metrics[m] * fs.n for fs in file_scores) / n if n else None
        elif m in ("unparsed", "no_answer"):
            out[m] = sum(fs.metrics.get(m) or 0 for fs in file_scores)
        else:
            out[m] = _mean([fs.metrics.get(m) for fs in file_scores])
    out["n"] = n
    return out


def aggregate(runs_results: list):
    per_file = defaultdict(lambda: defaultdict(list))
    per_task = defaultdict(lambda: defaultdict(list))
    for res in runs_results:
        tasks = defaultdict(list)
        for fn, fs in res.items():
            for m, v in fs.metrics.items():
                per_file[fn][m].append(scale(m, v))
            per_file[fn]["n"].append(fs.n)
            tasks[fs.task].append(fs)
        for t, fss in tasks.items():
            for m, v in task_level(fss).items():
                per_task[t][m].append(scale(m, v) if m != "n" else v)
    return per_file, per_task


def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def fmt(m, vals):
    mean, sd = ms(vals)
    if mean is None:
        return "n/a"
    s = f"{mean:.4f}" if m in ("l2", "l1", "cosine", "nrmse", "rmse", "rae", "r2") else f"{mean:.2f}"
    if len(vals) > 1 and sd is not None:
        s += f" ±{sd:.2f}"
    return s


def report(per_file, per_task, out_dir: Path, nruns: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    tasks = sorted(per_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    print(f"\n{'=' * 22} MULTIMODAL  ({nruns} run{'s' if nruns > 1 else ''}) {'=' * 22}")
    print(f"{'task':14s}{'n':>6s}  metrics")
    for t in tasks:
        d = per_task[t]
        summary[t] = {m: {"mean": ms(v)[0], "std": ms(v)[1], "runs": v} for m, v in d.items()}
        shown = "  ".join(f"{m}={fmt(m, v)}" for m, v in d.items() if m != "n")
        print(f"{t:14s}{int(ms(d['n'])[0]):6d}  {shown}")
        for m, v in d.items():
            rows.append([t, m, ms(v)[0], ms(v)[1], nruns])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pf = {fn: {m: {"mean": ms(v)[0], "std": ms(v)[1], "runs": v} for m, v in d.items()}
          for fn, d in per_file.items()}
    (out_dir / "per_file.json").write_text(
        json.dumps(pf, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task", "metric", "mean", "std", "runs"])
        w.writerows(rows)
    print(f"\nwrote {out_dir}/summary.json, per_file.json, summary.csv")


def inspect(by_file):
    print(f"{'n':>5s}  {'category':25s} {'task':14s} file_path")
    unrouted = 0
    for fp, items in sorted(by_file.items()):
        cat = route(fp)
        if cat is None:
            unrouted += 1
        print(f"{len(items):5d}  {str(cat):25s} {paper_task(fp):14s} {fp}")
    total = sum(len(v) for v in by_file.values())
    print(f"\ntotal rows: {total}   unrouted files: {unrouted}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="model-output JSONL files; one file = one run")
    ap.add_argument("--out", default="chemeval_mm_results")
    ap.add_argument("--constants", default=str(TEXT_EVAL / "repo_constants.json"),
                    help="path to repo_constants.json (from text_eval)")
    ap.add_argument("--judge-model", default="gpt-4o-2024-08-06")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--judge-max-tokens", type=int, default=2048)
    ap.add_argument("--skip-judge", action="store_true",
                    help="only score code-graded tasks (no API cost)")
    ap.add_argument("--dump-judge-prompts", action="store_true",
                    help="write judge prompts to JSONL, do not call the judge")
    ap.add_argument("--mc-tf-extractor", choices=["judge", "rule"], default="judge",
                    help="multiple-choice: let the judge extract option or use rule-based extractor")
    ap.add_argument("--limit-per-file", type=int, default=None)
    ap.add_argument("--no-item-dump", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="show routing per file_path and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    paths = [Path(p) for p in a.inputs]
    runs = [load_run(p, a.limit_per_file) for p in paths]

    if a.inspect:
        inspect(runs[0])
        return

    consts = json.loads(Path(a.constants).read_text(encoding="utf-8"))
    ctx = Ctx(consts, a.mc_tf_extractor)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = None
    if not a.skip_judge and not a.dump_judge_prompts:
        judge = Judge(a.judge_model, out_dir / "judge_cache.jsonl",
                      a.judge_workers, a.judge_max_tokens)
    dump = open(out_dir / "judge_prompts.jsonl", "w", encoding="utf-8") if a.dump_judge_prompts else None

    all_res = []
    for k, (p, by_file) in enumerate(zip(paths, runs), 1):
        log.info("evaluating run %d/%d: %s", k, len(runs), p.name)
        res = evaluate_run(by_file, ctx, judge, dump)
        all_res.append(res)
        if not a.no_item_dump and not a.dump_judge_prompts:
            with open(out_dir / f"items_run{k}.jsonl", "w", encoding="utf-8") as f:
                for fn, fs in res.items():
                    for r in fs.rows:
                        f.write(json.dumps({"file": fn, **r},
                                           ensure_ascii=False, default=str) + "\n")

    if dump:
        dump.close()
        print(f"wrote {out_dir}/judge_prompts.jsonl")
        return

    per_file, per_task = aggregate(all_res)
    report(per_file, per_task, out_dir, len(runs))

    for fn, fs in all_res[0].items():
        for note in fs.notes:
            log.warning("%s: %s", fn, note)


if __name__ == "__main__":
    main()
