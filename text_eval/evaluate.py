#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chemeval_text_eval.py - faithful, robust re-implementation of ChemEval's TEXTUAL evaluation.

What it does
------------
* Reads one or more model-output JSONL files (one file == one run), each line:
      {"query": ..., "target": ..., "filename": ..., "answer": <raw model output>, ["shot": "0shot"|"3shot"]}
* Splits the rows into 0-shot and 3-shot and evaluates the two settings SEPARATELY.
* Routes every row to the same metric the authors' code uses (routing rules copied from
  `2_Extract.py::split_jsonl_by_filename`, judge routing from `1_prompt_chem.py`).
* Code-graded tasks: extracts the answer ROBUSTLY (finds the {"answer": ...} object even when the
  model adds prose / code fences / <think> blocks) and then applies the repo's metric formulas.
* LLM-judged tasks: uses the repo's judge prompt templates VERBATIM (repo_constants.json, generated
  by extract_from_repo.py via `ast`) and the repo's score aggregation.
* Several runs -> mean +- std (the paper's "times" setting).

Usage
-----
  python chemeval_text_eval.py --inspect --inputs run1.jsonl            # routing/shot sanity check, no cost
  python chemeval_text_eval.py --inputs run1.jsonl --skip-judge --out res   # code-graded tasks only (free)
  python chemeval_text_eval.py --inputs run1.jsonl run2.jsonl --judge-model gpt-4o-2024-08-06 --out res

See `--help` for everything else.
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
log = logging.getLogger("chemeval")

# --------------------------------------------------------------------------------------
# 0. shot detection and task routing
# --------------------------------------------------------------------------------------
_SHOT3 = re.compile(r"(?:3|three)[\s_\-]?shot", re.I)


def detect_shot(filename: str, query: str = "", record_shot: Any = None,
                query_regex: Optional[re.Pattern] = None) -> str:
    """'0shot' or '3shot'. Priority: explicit record field > tag in filename > query regex > 0shot."""
    if record_shot:
        s = str(record_shot).lower()
        if s.startswith("3") or "three" in s:
            return "3shot"
        if s.startswith("0") or "zero" in s:
            return "0shot"
    if _SHOT3.search(filename or ""):
        return "3shot"
    if query_regex is not None and query and query_regex.search(query):
        return "3shot"
    return "0shot"


def _any(f: str, keys) -> bool:
    return any(k in f for k in keys)


def route(fname: str) -> Optional[str]:
    """Category of a task file. Order and keywords copied from the repo's split_jsonl_by_filename()."""
    f = fname
    if _any(f, ["Chemical_Literature_Topic_Classification", "BBBP", "ClinTox", "HIV", "分子性质分类预测极性",
                "Molecular_Property_Classification_Polarity", "产率预测", "Product_Yield_Prediction",
                "Reaction_Type_Recognition_and_Induction", "化学文献主题分类"]):
        return "classification"
    if _any(f, ["ESOL", "HOMO", "Lipo", "LUMO", "分子性质回归预测极性", "Molecular_Property_Regression_Polarity",
                "熔点", "Melting_Point", "沸点", "Boiling_Point", "合成难度评估", "Synthetic_Difficulty_Evaluation",
                "反应温度推荐", "Reaction_Temperature_Recommendation", "反应时间推荐", "Reaction_Time_Recommendation"]):
        return "regression"
    if _any(f, ["SMILES_test", "SMILES_Conversion", "SMILES与SELFIES互译", "SMILES_to_SELFIES"]):
        return "molecule_design_S"
    if _any(f, ["分子式_test", "Formula_test"]):
        return "molecule_design_F"
    if _any(f, ["IUPAC_test", "IUPAC_Conversion", "SMILES转IUPAC"]):
        return "molecule_design_I"
    if _any(f, ["基于文本描述生成分子名称", "Molecular_Name_Generation_from_Text_Description"]):
        return "molecule_design"
    if "SIDER" in f:
        return "sider"
    if _any(f, ["命名实体识别", "Chemical_Named_Entity_Recognition", "产率性能抽取", "Yield_Extraction", "催化类型抽取",
                "Catalysis_Type_Extraction", "化学反应类型识别归纳", "Reaction_Type_Recognition_and_Induction",
                "反应时间抽取", "Reaction_Time_Extraction", "合成反应添加剂抽取", "thetic_Reaction_Additive_Extraction",
                "反应温度抽取", "Reaction_Temperature_Extraction", "反应溶剂抽取", "Synthetic_Reaction_Solvent_Extraction",
                "表征手段抽取", "Characterization_Method_Extraction"]):
        return "entity_extraction"
    if _any(f, ["实体关系分类", "Chemical_Entity_Relationship_Classification"]):
        return "relation_extraction"
    if _any(f, ["合成反应产物抽取", "Reaction_Product_Extraction", "合成反应底物抽取", "Synthetic_Reaction_Substrate_Extraction"]):
        return "entity_recognition"
    if _any(f, ["反应产物预测", "Reaction_Product_Prediction", "反应底物推荐", "Substrate_Recommendation", "配体推荐",
                "Ligand_Recommendation", "溶剂推荐", "Solvent_Recommendation", "试剂推荐", "Reagent_Recommendation",
                "催化剂推荐", "Catalyst_Recommendation"]):
        return "reagent_selection"
    if _any(f, ["反应活化能刻画", "Reaction_Rate_Prediction"]):
        return "reaction_rate"
    if _any(f, ["描述分子的物理化学性质", "Physicochemical_Property_Prediction_from_Molecular_Structure", "填空任务",
                "选择任务", "判断任务", "True_False_Task", "Multiple_Choice", "Fill-in-the-Blank_Task", "简答任务",
                "Short_Answer_Task", "计算任务", "Calculation_Task", "合成路径推荐", "Synthetic_Pathway_Recommendation",
                "反应中间体推导", "Intermediate_Derivation_test", "化学论文摘要生成", "Chemical_Paper_Abstract_Generation",
                "研究内容提纲生成", "Research_Outline_Generation"]):
        return "judge"
    # --- looser fallbacks (not in the repo) for file names that differ slightly ---
    if re.search(r"IUPAC.{0,6}(转|_to_|2).{0,6}SMILES", f, re.I):
        return "molecule_design_S"
    if re.search(r"SMILES.{0,6}(转|_to_|2).{0,6}IUPAC", f, re.I):
        return "molecule_design_I"
    if re.search(r"分子式|Formula", f):
        return "molecule_design_F"
    return None


def judge_kind(f: str) -> Optional[str]:
    """Which judge template. Order copied from 1_prompt_chem.py (+ English names, + generic path rec.)."""
    if _any(f, ["单步合成", "Single_Step"]):
        return "single_step"
    if _any(f, ["多步合成", "Multistep", "Multi_Step", "Multi-step"]):
        return "multi_step"
    if _any(f, ["反应中间体", "Intermediate_Derivation"]):
        return "intermediate"
    if _any(f, ["论文摘要", "Abstract_Generation"]):
        return "abstract"
    if _any(f, ["物理化学性质", "Physicochemical_Property"]):
        return "physchem"
    if _any(f, ["提纲", "Outline_Generation"]):
        return "outline"
    if _any(f, ["FillBlank", "填空", "Fill-in-the-Blank"]):
        return "fillblank"
    if _any(f, ["MultipleChoice", "选择", "选题", "Multiple_Choice"]):
        return "choice"
    if _any(f, ["TrueFalse", "判断", "True_False"]):
        return "truefalse"
    if _any(f, ["ShortAnswer", "简答", "名词解释", "Short_Answer"]):
        return "shortanswer"
    if _any(f, ["CalculationQuestions", "计算", "Calculation_Task"]):
        return "calculation"
    if _any(f, ["合成路径推荐", "Synthetic_Pathway_Recommendation"]):
        return "path_unknown"
    return None


# paper task code, regex on file name (first match wins)
TASKS = [
    ("MCTask", r"选择任务|Multiple_?Choice"), ("FBTask", r"填空任务|Fill-in-the-Blank|FillBlank"),
    ("TFTask", r"判断任务|True_?False"), ("SATask", r"简答任务|Short_?Answer"),
    ("CalcTask", r"计算任务|Calculation"),
    ("CNER", r"命名实体识别|Chemical_Named_Entity"), ("CERC", r"实体关系分类|Entity_Relationship"),
    ("SubE", r"合成反应底物抽取|Substrate_Extraction"), ("ProdE", r"合成反应产物抽取|Reaction_Product_Extraction"),
    ("AddE", r"添加剂抽取|Additive_Extraction"), ("SolvE", r"溶剂抽取|Solvent_Extraction"),
    ("TempE", r"反应温度抽取|Reaction_Temperature_Extraction"), ("TimeE", r"反应时间抽取|Reaction_Time_Extraction"),
    ("CharME", r"表征手段抽取|Characterization_Method"), ("CatTE", r"催化类型抽取|Catalysis_Type"),
    ("YieldE", r"产率性能抽取|Yield_Extraction"),
    ("AbsGen", r"论文摘要生成|Abstract_Generation"), ("OLGen", r"提纲生成|Outline_Generation"),
    ("TopC", r"文献主题分类|Topic_Classification"), ("ReactTR", r"反应类型识别归纳|Reaction_Type_Recognition"),
    ("MolNG", r"基于文本描述生成分子名称|Molecular_Name_Generation"),
    ("IUPAC2MF", r"IUPAC转分子式|IUPAC.{0,8}Formula"), ("SMILES2MF", r"SMILES转分子式|SMILES.{0,8}Formula"),
    ("IUPAC2SMILES", r"IUPAC转SMILES|IUPAC_to_SMILES"), ("SMILES2IUPAC", r"SMILES转IUPAC|SMILES_to_IUPAC"),
    ("S2S", r"SMILES与SELFIES互译|SELFIES"),
    ("MolPC", r"分子性质分类|Molecular_Property_Classification|BBBP|ClinTox|HIV|SIDER"),
    ("MolPR", r"分子性质回归|Molecular_Property_Regression|ESOL|Lipo|HOMO|LUMO|熔点|沸点|Melting|Boiling"),
    ("Mol2PC", r"物理化学性质|Physicochemical_Property"),
    ("SubRec", r"反应底物推荐|Substrate_Recommendation"), ("PathRec", r"合成路径推荐|Pathway_Recommendation"),
    ("SynDE", r"合成难度评估|Synthetic_Difficulty"), ("LRec", r"配体推荐|Ligand_Recommendation"),
    ("RRec", r"试剂推荐|Reagent_Recommendation"), ("SolvRec", r"溶剂推荐|Solvent_Recommendation"),
    ("CatRec", r"催化剂推荐|Catalyst_Recommendation"), ("TempRec", r"反应温度推荐|Temperature_Recommendation"),
    ("TimeRec", r"反应时间推荐|Time_Recommendation"), ("PPred", r"反应产物预测|Reaction_Product_Prediction"),
    ("YPred", r"产物产率预测|Product_Yield_Prediction"), ("RatePred", r"反应速率预测|反应活化能|Reaction_Rate"),
    ("IMDer", r"反应中间体推导|Intermediate_Derivation"),
]
TASK_ORDER = [t for t, _ in TASKS]


def paper_task(fname: str) -> str:
    for code, pat in TASKS:
        if re.search(pat, fname):
            return code
    return Path(fname).stem


# --------------------------------------------------------------------------------------
# 1. raw-answer cleaning and robust answer extraction
# --------------------------------------------------------------------------------------
def repo_preprocess(ans: Any) -> Optional[str]:
    """Exactly what 1_prompt_chem.py does to a model answer before judging (no strip)."""
    if ans is None:
        return None
    if isinstance(ans, (list, tuple)):
        if not ans:
            return None
        ans = ans[0]
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
    t = re.sub(r"```[A-Za-z0-9_+-]*", "", t)  # drop code fences, keep content
    t = t.strip()
    return t or None


def _balanced_objects(text: str, max_starts: int = 200):
    """Yield every balanced {...} substring (quote-aware)."""
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
    t = (s.replace("{{", "{").replace("}}", "}").replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'"))
    t = re.sub(r",\s*([}\]])", r"\1", t)
    cands.append(t)
    for c in cands:
        for fn in (json.loads, ast.literal_eval):
            try:
                return fn(c)
            except Exception:
                pass
    return None


_PLACEHOLDER = re.compile(r"^\s*(the answer you judge|your answer|answer here|valid\s+(smiles|selfies|iupac)\b.*|\.\.\.)\s*$", re.I)


def _is_placeholder(v: Any) -> bool:
    if isinstance(v, str):
        return bool(_PLACEHOLDER.match(v))
    if isinstance(v, list) and v:
        return all(_is_placeholder(x) for x in v)
    return False


def extract_answer_value(text: Optional[str]):
    """Return (value, how). how in {'json','regex','phrase','raw','none'}.
    value is str | list | dict | None. The LAST non-placeholder {"answer": ...} object wins, so a model that
    echoes the format example first and answers afterwards is still scored correctly."""
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
    """Mean of all numbers; 'a-b' / 'a~b' ranges become 'a b' but a leading minus is kept ('-78' stays -78)."""
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
    m = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*(?:~|–|to|-)\s*(\d*\.?\d+)\s*", s)  # range -> midpoint
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return last_number(s)


# --------------------------------------------------------------------------------------
# 2. chemistry helpers (lazy imports so judge-only / generation use does not need RDKit)
# --------------------------------------------------------------------------------------
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
    """Repo's canonicalize_molecule_smiles(): ';'-separated parts, atom maps cleared, kekulised, sorted."""
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
    """Best SMILES/SELFIES string from an extracted value (also salvages answers buried in prose)."""
    t = value_to_text(v, ".")
    if not t:
        return None
    t = re.sub(r"</?SMILES>", "", t, flags=re.I).strip()
    if how in ("raw", "phrase") and re.search(r"\s", t):
        Chem = rd()["Chem"]
        toks = [x.strip(".,;:()\"'`*") for x in re.split(r"\s+", t)]
        ok = [x for x in toks if len(x) > 1 and Chem.MolFromSmiles(x) is not None]
        if ok:
            return ok[-1]
    return t.replace(" ", "").replace("\n", "") or None


def too_long_or_single(s: str, n: int = 4908) -> bool:
    return len(s) >= n or len(set(s)) == 1


# --------------------------------------------------------------------------------------
# 3. data containers
# --------------------------------------------------------------------------------------
@dataclass
class Item:
    idx: int
    query: str
    gold: Any
    answer: Any
    filename: str
    shot: str


@dataclass
class FileScore:
    n: int
    metrics: dict
    item_vals: dict = field(default_factory=dict)   # metric -> per-item values (None = excluded)
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    category: str = ""
    task: str = ""


@dataclass
class Ctx:
    consts: dict
    temp_gold_kelvin: bool = False
    mc_tf_extractor: str = "judge"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _json_rate(hows):
    return sum(h == "json" for h in hows) / len(hows) if hows else 0.0


# --------------------------------------------------------------------------------------
# 4. code-graded scorers (metric formulas follow the repo; extraction is robust)
# --------------------------------------------------------------------------------------
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
    f = items[0].filename
    topic = bool(re.search(r"化学文献主题分类|Chemical_Literature_Topic_Classification", f))
    corr, valid, rows, hows = [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v)
        gold = str(first(it.gold)).replace("正确", "Yes").replace("错误", "No").strip()
        if topic:  # repo: calculate_accuracy2 -> gold is a substring of the prediction
            pred = txt
            ok = bool(pred) and gold.lower() in pred.lower()
        else:      # Yes / No exact match (repo's calculate_accuracy; invalid counts as wrong)
            pred = norm_yes_no(txt, how)
            ok = pred is not None and pred.lower() == gold.lower()
        corr.append(1.0 if ok else 0.0)
        valid.append(1.0 if pred else 0.0)
        rows.append({"idx": it.idx, "pred": pred, "gold": gold, "correct": ok})
    return FileScore(len(items), {"accuracy": _mean(corr), "valid": _mean(valid), "json_rate": _json_rate(hows)},
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


def s_regression(items, ctx):
    f = items[0].filename
    avg_task = bool(re.search(r"反应温度推荐|反应时间推荐|Reaction_(Temperature|Time)_Recommendation", f))
    temp_task = bool(re.search(r"温度|Temperature", f))
    g_ok, p_ok, valid, rows, hows = [], [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v, " ")
        pred = numbers_avg(txt) if avg_task else last_number(txt)
        if pred is not None and abs(pred) > 1000:
            pred = None
        g = gold_number(it.gold)
        if ctx.temp_gold_kelvin and temp_task and g is not None:
            g -= 273.15
        valid.append(0.0 if pred is None else 1.0)
        if pred is not None and g is not None:
            g_ok.append(g)
            p_ok.append(pred)
        rows.append({"idx": it.idx, "pred": pred, "gold": g})
    m = _reg_stats(g_ok, p_ok)
    m.update(valid=_mean(valid), json_rate=_json_rate(hows))
    return FileScore(len(items), m, {"valid": valid}, rows)


def s_smiles(items, ctx):
    """molecule_design_S (IUPAC->SMILES, SMILES<->SELFIES) and molecule_design (text -> SMILES): Tanimoto."""
    f = items[0].filename
    selfies_file = "SELFIES" in f
    sims, valid, noans, rows, hows, notes = [], [], 0, [], [], []
    skipped = 0
    for pos, it in enumerate(items):
        gold = str(first(it.gold)).replace(" ", "").replace(";", ".")
        v, how = get_value(it.answer)
        hows.append(how)
        pred = smiles_candidate(v, how)
        if selfies_file:
            m = re.search(r"convert the given\s+(SMILES|SELFIES)", it.query or "", re.I)
            to_selfies = (m.group(1).upper() == "SMILES") if m else (pos < 25)  # repo's i<25 rule as fallback
        else:
            to_selfies = False
        if to_selfies:
            gold_smi = selfies_to_smiles(gold)
            pred_smi = selfies_to_smiles(pred.replace(";", ".")) if pred else None
        else:
            gold_smi, pred_smi = gold, pred
        gold_c = canon_smiles(gold_smi) if gold_smi else None
        if gold_c is None:
            skipped += 1
            rows.append({"idx": it.idx, "skipped": "gold not parseable"})
            continue
        pred_c = None
        if pred_smi:
            if too_long_or_single(pred_smi):
                pred_c = None
            else:
                pred_c = canon_smiles(pred_smi)
        if not pred:
            noans += 1
        sim = tanimoto(pred_c, gold_c) if pred_c else 0.0
        sims.append(sim)
        valid.append(1.0 if pred_c else 0.0)
        rows.append({"idx": it.idx, "pred": pred_c, "gold": gold_c, "tanimoto": sim})
    if skipped:
        notes.append(f"{skipped} items skipped: gold SMILES/SELFIES not parseable")
    return FileScore(len(sims), {"tanimoto": _mean(sims), "valid": _mean(valid), "no_answer": noans,
                                 "json_rate": _json_rate(hows)},
                     {"tanimoto": sims, "valid": valid}, rows, notes)


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
            l2s.append(0.0)
            l1s.append(0.0)
            coss.append(0.0)
            valid.append(0.0)
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


def _opsin(names):
    try:
        from py2opsin import py2opsin
        out = py2opsin(chemical_name=names, output_format="SMILES")
        return out if isinstance(out, list) else [out]
    except Exception as e:  # Java missing / jar problem
        log.warning("OPSIN unavailable (%s) - SMILES2IUPAC Tanimoto skipped", e)
        return None


def s_iupac(items, ctx):
    """SMILES -> IUPAC: exact match, BLEU, edit distance (+ Tanimoto of OPSIN(pred) vs gold SMILES)."""
    import warnings as _w
    from nltk.tokenize.destructive import NLTKWordTokenizer
    from nltk.translate.bleu_score import sentence_bleu
    import Levenshtein
    tok = NLTKWordTokenizer().tokenize
    ws = lambda s: re.sub(r"\s+", " ", s).strip()
    em, bl, ed, rows, hows, preds, gold_smis = [], [], [], [], [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        pred = value_to_text(v, ";")
        pred = ws(pred) if pred else None
        gold = ws(str(first(it.gold)))
        m = re.search(r"given by SMILES:\s*(.+?)\s*You must output", it.query or "", re.S)
        gold_smis.append(m.group(1).strip() if m else None)
        preds.append(pred)
        if not pred:
            em.append(0.0), bl.append(0.0), ed.append(float(len(gold)))
        else:
            em.append(1.0 if pred.lower() == gold.lower() else 0.0)
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                bl.append(sentence_bleu([tok(gold.lower())], tok(pred.lower()), weights=(0.25,) * 4))
            ed.append(float(Levenshtein.distance(gold, pred)))
        rows.append({"idx": it.idx, "pred": pred, "gold": gold})
    metrics = {"exact_match": _mean(em), "bleu": _mean(bl), "edit_distance": _mean(ed), "json_rate": _json_rate(hows)}
    vals = {"exact_match": em, "bleu": bl, "edit_distance": ed}
    names = [(p or "").replace(";", " ") for p in preds]
    smis = _opsin(names) if any(names) else None
    if smis is not None and len(smis) == len(items):
        sims = []
        for s, g in zip(smis, gold_smis):
            if s and g:
                gc, pc = canon_smiles(g.replace(" ", "")), canon_smiles(s)
                sims.append(tanimoto(pc, gc) if (gc and pc) else 0.0)
            else:
                sims.append(0.0)
        metrics["tanimoto"] = _mean(sims)
        vals["tanimoto"] = sims
    return FileScore(len(items), metrics, vals, rows)


def s_sider(items, ctx):
    labels = ctx.consts["sider_labels"]
    tot_c, tot_n, accs, rows, hows = 0, 0, [], [], []
    for it in items:
        gold = it.gold
        if isinstance(gold, str):
            gold = _parse_obj(gold) or {}
        v, how = get_value(it.answer)
        pred = v if isinstance(v, dict) else None
        if pred is None:  # the dict itself may be the answer, with no "answer" key
            for s in _balanced_objects(clean_model_text(it.answer) or ""):
                o = _parse_obj(s)
                if isinstance(o, dict) and any(k in o for k in labels):
                    pred = o
        hows.append("json" if pred is not None else "none")
        norm = lambda x: str(x).strip().lower()
        c = sum(1 for k, gv in gold.items() if pred is not None and k in pred and norm(pred[k]) == norm(gv))
        n = len(gold) or 1
        tot_c += c
        tot_n += n
        accs.append(c / n)
        rows.append({"idx": it.idx, "correct": c, "of": n})
    return FileScore(len(items), {"accuracy": tot_c / tot_n if tot_n else None, "json_rate": _json_rate(hows)},
                     {"accuracy": accs}, rows)


def _f1_sets(t, p):
    t, p = set(t), set(p)
    tp = len(t & p)
    pr = tp / len(p) if p else 0
    rc = tp / len(t) if t else 0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def _has(name, *keys):
    return any(k in name for k in keys)


def s_entity_extraction(items, ctx):
    R = ctx.consts["replacements"]
    f = items[0].filename

    def apply(d, g, p):
        for k, val in d.items():
            g, p = g.replace(k, val), p.replace(k, val)
        return g, p

    f1s, rows, hows = [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        pre = value_to_text(v, ", ") or ""
        g = first(it.gold)
        gold = ", ".join(map(str, g)) if isinstance(g, (list, tuple)) else str(g)
        if _has(f, "化学命名实体识别", "Chemical_Named_Entity_Recognition"):
            gold, pre = apply(R["chem_entity_dict"], gold, pre)
        if _has(f, "化学实体关系分类", "Chemical_Entity_Relationship_Classification"):
            gold, pre = apply(R["relation_dict"], gold, pre)
        if _has(f, "合成反应添加剂抽取", "thetic_Reaction_Additive_Extraction"):
            gold, pre = apply(R["additive_dict"], gold, pre)
        if _has(f, "合成反应溶剂抽取", "Synthetic_Reaction_Solvent_Extraction"):
            gold, pre = apply(R["solvent_dict"], gold, pre)
        if _has(f, "催化类型抽取", "Catalysis_Type_Extraction", "化学反应类型识别归纳", "Reaction_Type_Recognition_and_Induction"):
            pre = pre.replace("reactions", "").replace("reaction", "").replace(",", ".")
            gold = gold.replace("reactions", "").replace("reaction", "").replace(",", ".")
        if _has(f, "合成反应温度抽取", "Reaction_Temperature_Extraction"):
            pre = pre.replace("℃", "").replace("°C", "").replace("degrees Celsius", "").replace("degrees Celsius.", "")
            gold = gold.replace("℃", "").replace("°C", "")
        if _has(f, "合成反应时间抽取", "Reaction_Time_Extraction"):
            pre = pre.replace("hours", "").replace("hour", "").replace("h", "").replace(".0", "")
            if " min" in pre:
                try:
                    pre = str(float(pre.replace("min", "").strip()) / 60).replace(".0", "")
                except Exception:
                    m = re.search(r"(\d+) min", pre)
                    if m:
                        pre = m.group(1).replace(".0", "")
            gold = gold.replace("h", "")
        if _has(f, "产率性能抽取", "Yield_Extraction"):
            pre = pre.replace("answer:", "").replace("percent", "%").replace("% to ", "-").replace("%-", "-")
        norm = lambda s: set(s.lower().replace(" ", "").split(","))
        s = _f1_sets(norm(gold), norm(pre))
        f1s.append(s)
        rows.append({"idx": it.idx, "pred": pre, "gold": gold, "f1": s})
    return FileScore(len(items), {"f1": _mean(f1s), "json_rate": _json_rate(hows)}, {"f1": f1s}, rows)


def s_relation(items, ctx):
    f1s, rows, hows = [], [], []
    pat = r"\(.*?\)"
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        if isinstance(v, list) and v and all(isinstance(x, (list, tuple)) for x in v):
            pre = ",".join(f"({', '.join(map(str, x))})" for x in v)
        else:
            pre = value_to_text(v, ", ") or ""
        if pre and "(" not in pre:
            parts = pre.replace(";", ",").split(",")
            pre = ",".join(f"({parts[i]}, {parts[i + 1]})" for i in range(0, len(parts), 2) if i + 1 < len(parts))
        gold = str(first(it.gold))
        t = set(re.findall(pat, gold.lower().replace(" ", "")))
        p = set(re.findall(pat, pre.lower().replace(" ", "")))
        s = _f1_sets(t, p)
        f1s.append(s)
        rows.append({"idx": it.idx, "pred": pre, "gold": gold, "f1": s})
    return FileScore(len(items), {"f1": _mean(f1s), "json_rate": _json_rate(hows)}, {"f1": f1s}, rows)


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            r = ast.literal_eval(x)
            return r if isinstance(r, list) else None
        except Exception:
            try:
                r = ast.literal_eval(x.replace("[", '["').replace("]", '"]').replace(", ", '", "'))
                return r if isinstance(r, list) else None
            except Exception:
                return None
    return None


def s_entity_recognition(items, ctx):
    corr = tot = 0
    accs, rows, hows = [], [], []
    for it in items:
        gl = _as_list(first(it.gold)) if not isinstance(it.gold, list) else it.gold
        gl = [str(x).replace(" ", "") for x in (gl or [])]
        v, how = get_value(it.answer)
        hows.append(how)
        al = _as_list(v)
        al = [str(x).replace(" ", "") for x in al] if al is not None else ["X"] * len(gl)
        c = sum(1 for a, b in zip(al, gl) if a == b)
        n = len(gl) or 1
        corr += c
        tot += n
        accs.append(c / n)
        rows.append({"idx": it.idx, "correct": c, "of": n})
    return FileScore(len(items), {"accuracy": corr / tot if tot else None, "json_rate": _json_rate(hows)},
                     {"accuracy": accs}, rows)


def s_reagent(items, ctx):
    Chem = rd()["Chem"]
    f1s, rows, hows, fallback = [], [], [], 0

    def canon_frag(m, selfies):
        if selfies:
            m = selfies_to_smiles(m) or ""
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
        gs = gs.replace(" ", "").replace("and", ".").replace("\n", "")
        flag = False
        dec = selfies_to_smiles(gs) if "[" in gs and "]" in gs else None
        if dec:
            gs, flag = dec, True
        mol = Chem.MolFromSmiles(gs)
        if mol is not None:
            gold_set = set(Chem.MolToSmiles(mol).split("."))
            exact = True
        else:  # repo would `break` here; we fall back to raw-string fragments instead
            gold_set = set(x for x in gs.split(".") if x)
            exact = False
            fallback += 1
        v, how = get_value(it.answer)
        hows.append(how)
        lst = _as_list(v) if isinstance(v, str) and v.strip().startswith("[") else v
        ptxt = ".".join(map(str, lst)) if isinstance(lst, list) else (value_to_text(lst, ".") or "")
        ptxt = ptxt.replace(" ", "").replace("and", ".").replace("\n", "")
        pset = set()
        for frag in ptxt.split("."):
            if not frag:
                continue
            c = canon_frag(frag, flag)
            if c is not None:
                pset.add(c)
            elif not exact:
                pset.add(frag)
        s = _f1_sets(gold_set, pset)
        f1s.append(s)
        rows.append({"idx": it.idx, "pred": sorted(pset), "gold": sorted(gold_set), "f1": s})
    notes = [f"{fallback} gold answers not RDKit-parseable (repo aborts the file here); scored on raw strings"] if fallback else []
    return FileScore(len(items), {"f1": _mean(f1s), "json_rate": _json_rate(hows)}, {"f1": f1s}, rows, notes)


def _parse_range(s):
    nums = re.findall(r"[+-]?[0-9]*\.?[0-9]+", s.replace("-", " "))
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def s_rate(items, ctx):
    ovs, rows, hows = [], [], []
    for it in items:
        v, how = get_value(it.answer)
        hows.append(how)
        txt = value_to_text(v, " ")
        a, g = (_parse_range(txt) if txt else None), _parse_range(str(first(it.gold)))
        if a is None or g is None:
            ov = 0.0
        else:
            inter = max(0.0, min(a[1], g[1]) - max(a[0], g[0]))
            union = max(a[1], g[1]) - min(a[0], g[0])
            ov = inter / union if union else 0.0
        ovs.append(ov)
        rows.append({"idx": it.idx, "pred": a, "gold": g, "overlap": ov})
    return FileScore(len(items), {"overlap": _mean(ovs), "json_rate": _json_rate(hows)}, {"overlap": ovs}, rows)


CODE_SCORERS = {
    "classification": s_classification, "regression": s_regression,
    "molecule_design_S": s_smiles, "molecule_design": s_smiles, "molecule_design_F": s_formula,
    "molecule_design_I": s_iupac, "sider": s_sider, "entity_extraction": s_entity_extraction,
    "relation_extraction": s_relation, "entity_recognition": s_entity_recognition,
    "reagent_selection": s_reagent, "reaction_rate": s_rate,
}

# --------------------------------------------------------------------------------------
# 5. LLM-as-judge: prompts (verbatim templates), client with cache, parsing
# --------------------------------------------------------------------------------------
L1_KINDS = {"choice", "truefalse", "fillblank", "shortanswer", "calculation"}  # 2_L1_task_eval.py semantics
TEMPLATE_OF = {"choice": "template_choice", "truefalse": "template_trueflase", "fillblank": "template_fillblank",
               "shortanswer": "template_score", "calculation": "template_calculate", "abstract": "template_zhaiyao",
               "outline": "template_tigang", "physchem": "template_xingzhi", "single_step": "template_danbulujing",
               "multi_step": "template_duobulujing", "intermediate": "template_fanyingzhongjianti"}


def fillblank_question(query: str) -> str:
    for marker in ("接下来是需要你回答的问题：", "Please answer the following questions:"):
        pos = query.rfind(marker)
        if pos != -1:
            return query[pos:].replace(marker, "")
    return query.replace("请回答下列问题\n", "")  # repo would take just the last character here


def resolve_kind(kind: str, item: Item) -> str:
    if kind != "path_unknown":
        return kind
    g = str(first(item.gold))
    return "multi_step" if len(re.findall(r"(?m)^\s*\d+[.)、]", g)) >= 2 else "single_step"


def build_prompt(kind: str, it: Item, ctx: Ctx) -> str:
    T = ctx.consts["templates"]
    ans = repo_preprocess(it.answer)
    ans = "None" if ans is None else ans
    gold = first(it.gold)
    gold = gold if isinstance(gold, str) else str(gold)
    t = T[TEMPLATE_OF[kind]]
    if kind in ("choice", "truefalse"):
        return t.format(ans)
    if kind == "fillblank":
        return t.format(fillblank_question(it.query or ""), ans, gold)
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
        for a in range(self.retries):
            try:
                r = litellm.completion(model=self.model, messages=[{"role": "user", "content": prompt}],
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


def norm_gold_tf(g: Any) -> str:
    s = str(first(g)).strip().lower()
    if s in {"正确", "对", "correct", "true", "yes", "√"}:
        return "正确"
    if s in {"错误", "错", "incorrect", "false", "no", "×"}:
        return "错误"
    return s


def rule_extract(kind: str, raw: Any) -> str:
    """Deterministic alternative to the judge for choice / true-false (--mc-tf-extractor rule)."""
    v, how = get_value(raw)
    t = value_to_text(v, "") or ""
    if kind == "choice":
        return "".join(sorted(set(re.findall(r"[A-Z]", t)))) if re.fullmatch(r"[A-Z,、\s和and]+", t.strip()) or how == "json" else ""
    t = t.replace("不正确", "错误")
    pos, neg = bool(re.search(r"正确|\b(true|correct|yes)\b", t, re.I)), bool(re.search(r"错误|\b(false|incorrect|no)\b", t, re.I))
    return "正确" if pos and not neg else ("错误" if neg and not pos else "")


def score_judged(kind: str, items: list, outs: list, ctx: Ctx) -> FileScore:
    vals, rows = [], []
    unparsed = 0
    for it, out in zip(items, outs):
        if kind == "choice":
            if ctx.mc_tf_extractor == "rule":
                pred = rule_extract(kind, it.answer)
            else:
                pred = parse_choice(out)
            if pred is None:
                unparsed += 1
            vals.append(1.0 if pred is not None and pred == norm_gold_choice(it.gold) else 0.0)
            rows.append({"idx": it.idx, "judge_extracted": pred, "gold": norm_gold_choice(it.gold), "correct": vals[-1]})
        elif kind == "truefalse":
            pred = rule_extract(kind, it.answer) if ctx.mc_tf_extractor == "rule" else parse_tf(out)
            if pred is None:
                unparsed += 1
            vals.append(1.0 if pred and pred == norm_gold_tf(it.gold) else 0.0)
            rows.append({"idx": it.idx, "judge_extracted": pred, "gold": norm_gold_tf(it.gold), "correct": vals[-1]})
        elif kind == "fillblank":
            s = parse_score(out, 0.0, 1.0)
            if s is None:
                unparsed += 1
            vals.append(s)       # L1 semantics: unparsed items are excluded
            rows.append({"idx": it.idx, "judge_score": s, "judge_output": out})
        else:
            s = parse_score(out, 1.0, 5.0)
            if s is None:
                unparsed += 1
            norm = None if s is None else (s - 1) / 4
            if kind not in L1_KINDS and norm is None:
                norm = 0.0       # 3_other_task_eval.py semantics: missing -> minimum score
            vals.append(norm)
            rows.append({"idx": it.idx, "judge_score_1_5": s, "judge_output": out})
    name = "accuracy" if kind in ("choice", "truefalse") else "score"
    return FileScore(len(items), {name: _mean(vals), "unparsed": unparsed}, {name: vals}, rows,
                     [f"{unparsed} judge outputs could not be parsed"] if unparsed else [])


# --------------------------------------------------------------------------------------
# 6. evaluation of one run
# --------------------------------------------------------------------------------------
def load_run(path: Path, qre, limit=None):
    by_shot: dict = defaultdict(lambda: defaultdict(list))
    n = empty = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fn = r.get("filename") or ""
            shot = detect_shot(fn, r.get("query", ""), r.get("shot"), qre)
            lst = by_shot[shot][fn]
            if limit and len(lst) >= limit:
                continue
            ans = r.get("answer")
            empty += 1 if (ans is None or (isinstance(ans, str) and not ans.strip())) else 0
            lst.append(Item(len(lst), r.get("query", ""), r.get("target", r.get("gold")), ans, fn, shot))
            n += 1
    if empty:
        log.warning("%s: %d/%d rows have an empty answer (API failures?)", path.name, empty, n)
    return by_shot


def evaluate_run(by_shot: dict, ctx: Ctx, judge: Optional[Judge], shots, dump_prompts=None):
    results: dict = {s: {} for s in shots}
    jobs = []  # (shot, fname, kind, items, prompts)
    for shot in shots:
        for fname, items in by_shot.get(shot, {}).items():
            cat = route(fname)
            if cat is None:
                log.warning("[%s] UNROUTED file skipped: %s", shot, fname)
                continue
            if cat == "judge":
                kind = judge_kind(fname)
                if kind is None:
                    log.warning("[%s] no judge template for: %s", shot, fname)
                    continue
                if kind == "path_unknown":
                    groups = defaultdict(list)
                    for it in items:
                        groups[resolve_kind(kind, it)].append(it)
                    for k2, its in groups.items():
                        jobs.append((shot, fname, k2, its))
                else:
                    jobs.append((shot, fname, kind, items))
                continue
            try:
                fs = CODE_SCORERS[cat](items, ctx)
            except Exception as e:
                log.exception("[%s] scoring failed for %s: %s", shot, fname, e)
                continue
            fs.category, fs.task = cat, paper_task(fname)
            results[shot][fname] = fs

    if jobs:
        need_judge = [j for j in jobs if not (ctx.mc_tf_extractor == "rule" and j[2] in ("choice", "truefalse"))]
        prompts = [(j, [build_prompt(j[2], it, ctx) for it in j[3]]) for j in need_judge]
        if dump_prompts is not None:
            for j, ps in prompts:
                for it, p in zip(j[3], ps):
                    dump_prompts.write(json.dumps({"shot": j[0], "filename": j[1], "kind": j[2], "idx": it.idx,
                                                   "query": p, "target": str(first(it.gold))}, ensure_ascii=False) + "\n")
        outs_map = {}
        if judge is not None:
            flat = [p for _, ps in prompts for p in ps]
            outs = judge.run(flat)
            pos = 0
            for j, ps in prompts:
                outs_map[id(j)] = outs[pos:pos + len(ps)]
                pos += len(ps)
        for j in jobs:
            shot, fname, kind, items = j
            rule = ctx.mc_tf_extractor == "rule" and kind in ("choice", "truefalse")
            if judge is None and not rule:
                continue
            outs = [None] * len(items) if rule else outs_map[id(j)]
            fs = score_judged(kind, items, outs, ctx)
            fs.category, fs.task = f"judge:{kind}", paper_task(fname)
            key = fname if fname not in results[shot] else f"{fname}#{kind}"
            results[shot][key] = fs
    return results


# --------------------------------------------------------------------------------------
# 7. aggregation + reporting
# --------------------------------------------------------------------------------------
PCT = {"accuracy", "f1", "tanimoto", "score", "overlap", "exact_match", "bleu", "valid", "json_rate"}


def scale(metric, v):
    return None if v is None else (v * 100 if metric in PCT else v)


def task_level(file_scores: list) -> dict:
    """Pool item-decomposable metrics across the files of one paper task, macro-average the rest."""
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


def aggregate(runs_results: list, shots):
    """-> per_file[shot][file][metric] = [per-run], per_task[shot][task][metric] = [per-run]"""
    per_file = {s: defaultdict(lambda: defaultdict(list)) for s in shots}
    per_task = {s: defaultdict(lambda: defaultdict(list)) for s in shots}
    for res in runs_results:
        for s in shots:
            tasks = defaultdict(list)
            for fn, fs in res[s].items():
                for m, v in fs.metrics.items():
                    per_file[s][fn][m].append(scale(m, v))
                per_file[s][fn]["n"].append(fs.n)
                tasks[fs.task].append(fs)
            for t, fss in tasks.items():
                for m, v in task_level(fss).items():
                    per_task[s][t][m].append(scale(m, v) if m != "n" else v)
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


def report(per_file, per_task, shots, out_dir: Path, nruns: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    for s in shots:
        summary[s] = {}
        tasks = sorted(per_task[s], key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
        print(f"\n{'=' * 22} {s.replace('shot', '-shot').upper()}  ({nruns} run{'s' i nruns > 1 else ''}) {'=' * 22}")
        print(f"{'task':14s}{'n':>6s}  metrics")
        for t in tasks:
            d = per_task[s][t]
            summary[s][t] = {m: {"mean": ms(v)[0], "std": ms(v)[1], "runs": v} for m, v in d.items()}
            shown = "  ".join(f"{m}={fmt(m, v)}" for m, v in d.items() if m != "n")
            print(f"{t:14s}{int(ms(d['n'])[0]):6d}  {shown}")
            for m, v in d.items():
                rows.append([s, t, m, ms(v)[0], ms(v)[1], nruns])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pf = {s: {fn: {m: {"mean": ms(v)[0], "std": ms(v)[1], "runs": v} for m, v in d.items()}
              for fn, d in per_file[s].items()} for s in shots}
    (out_dir / "per_file.json").write_text(json.dumps(pf, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["shot", "task", "metric", "mean", "std", "runs"])
        w.writerows(rows)
    print(f"\nwrote {out_dir}/summary.json, per_file.json, summary.csv")


def inspect(by_shot_runs, qre):
    print(f"{'shot':6s} {'n':>5s} {'json%':>6s}  {'category':20s} {'task':13s} file")
    unrouted = 0
    for shot in sorted(by_shot_runs):
        for fn, items in sorted(by_shot_runs[shot].items()):
            cat = route(fn)
            cat = cat if cat != "judge" else f"judge:{judge_kind(fn)}"
            if route(fn) is None:
                unrouted += 1
            hows = [get_value(it.answer)[1] for it in items]
            print(f"{shot:6s} {len(items):5d} {100 * _json_rate(hows):6.0f}  {str(cat):20s} {paper_task(fn):13s} {fn}")
    n3 = sum(len(v) for v in by_shot_runs.get("3shot", {}).values())
    print(f"\nrows: 0shot={sum(len(v) for v in by_shot_runs.get('0shot', {}).values())}  3shot={n3}  unrouted files={unrouted}")
    if n3 == 0:
        print("NOTE: no 3-shot rows detected. If your file mixes 0-/3-shot rows under identical file names, add a 'shot' "
              "field to each row or pass --three-shot-query-regex.")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="model-output JSONL files; one file = one run")
    ap.add_argument("--out", default="chemeval_results")
    ap.add_argument("--shots", nargs="+", default=["0", "3"], choices=["0", "3"], help="settings to evaluate")
    ap.add_argument("--constants", default=str(HERE / "repo_constants.json"))
    ap.add_argument("--judge-model", default="gpt-4o-2024-08-06")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--judge-max-tokens", type=int, default=2048)
    ap.add_argument("--skip-judge", action="store_true", help="only score code-graded tasks (no API cost)")
    ap.add_argument("--dump-judge-prompts", action="store_true", help="write judge prompts to JSONL, do not call the judge")
    ap.add_argument("--mc-tf-extractor", choices=["judge", "rule"], default="judge",
                    help="multiple-choice / true-false: let the judge extract the option (repo) or use a rule-based extractor")
    ap.add_argument("--three-shot-query-regex", default=None, help="regex that marks a 3-shot query (if file names carry no tag)")
    ap.add_argument("--temp-gold-kelvin", action="store_true", help="subtract 273.15 from temperature golds (repo's dead branch)")
    ap.add_argument("--limit-per-file", type=int, default=None)
    ap.add_argument("--no-item-dump", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="show shot/routing/extraction health per file and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore")
    qre = re.compile(a.three_shot_query_regex) if a.three_shot_query_regex else None
    shots = [f"{s}shot" for s in a.shots]
    paths = [Path(p) for p in a.inputs]
    runs = [load_run(p, qre, a.limit_per_file) for p in paths]

    if a.inspect:
        merged: dict = defaultdict(dict)
        for sh in runs[0]:
            merged[sh].update(runs[0][sh])
        inspect(merged, qre)
        return

    consts = json.loads(Path(a.constants).read_text(encoding="utf-8"))
    ctx = Ctx(consts, a.temp_gold_kelvin, a.mc_tf_extractor)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = None
    if not a.skip_judge and not a.dump_judge_prompts:
        judge = Judge(a.judge_model, out_dir / "judge_cache.jsonl", a.judge_workers, a.judge_max_tokens)
    dump = open(out_dir / "judge_prompts.jsonl", "w", encoding="utf-8") if a.dump_judge_prompts else None

    all_res = []
    for k, (p, by_shot) in enumerate(zip(paths, runs), 1):
        log.info("evaluating run %d/%d: %s", k, len(runs), p.name)
        res = evaluate_run(by_shot, ctx, judge, shots, dump)
        all_res.append(res)
        if not a.no_item_dump:
            for s in shots:
                with open(out_dir / f"items_{s}_run{k}.jsonl", "w", encoding="utf-8") as f:
                    for fn, fs in res[s].items():
                        for r in fs.rows:
                            f.write(json.dumps({"file": fn, **r}, ensure_ascii=False, default=str) + "\n")
    if dump:
        dump.close()
        print(f"wrote {out_dir}/judge_prompts.jsonl")
        return
    per_file, per_task = aggregate(all_res, shots)
    report(per_file, per_task, shots, out_dir, len(runs))
    for s in shots:
        for fn, fs in all_res[0][s].items():
            for note in fs.notes:
                log.warning("[%s] %s: %s", s, fn, note)


if __name__ == "__main__":
    main()
