#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cost_estimate.py - Cost estimation for ChemEval MULTIMODAL benchmark.

Estimates the cost of running multimodal vision models on the 1,240 multimodal
instances of ChemEval across various models (GPT-4o, Claude Sonnet/Haiku, Gemini,
etc.), including both model generation and LLM-as-a-judge evaluation.

Features:
  - Precise vision token calculations (OpenAI tiling model & Claude aspect-ratio scaling)
  - Full routing of all 54 task files to code-graded vs. LLM-judged
  - Optional calibration from real Langfuse billing exports (--calibration-csv)
  - Low / Expected / High scenario projections (accounting for thinking/reasoning tokens)
  - Per-task cost breakdown (--by-task)
  - CSV export for reporting and analysis (--out)

Usage:
    python cost_estimate.py                                       # Default models (Haiku, Luna, GPT-4o)
    python cost_estimate.py --models gpt4o claude-sonnet-5-5      # Specific models
    python cost_estimate.py --by-task                             # Show per-task cost breakdown
    python cost_estimate.py --calibration-csv langfuse_export.csv # Calibrate with real billing
    python cost_estimate.py --list-files                          # Show all 54 files and their routing
    python cost_estimate.py --out cost_report/                    # Export results to CSV
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_EVAL = HERE.parent / "text_eval"
sys.path.insert(0, str(HERE))

try:
    import evaluate as ev
except ImportError:
    sys.exit("multimodal_eval/evaluate.py must be in the same folder as this script.")

# --------------------------------------------------------------------------------------
# Price table - USD per 1M tokens (checked 2026-10-30)
# --------------------------------------------------------------------------------------
PRICES = {
    "claude-haiku-5-5": dict(label="Claude Haiku 5.5", family="claude", **{"in": 0.10, "out": 0.50},
                             long_over=100_000, in_long=0.50, out_long=2.50, batch=0.5),
    "gpt-6-luna": dict(label="GPT-6 Luna", family="openai", **{"in": 0.10, "out": 0.50},
                       long_over=272_000, in_long=0.20, out_long=0.75, batch=0.5),
    "claude-sonnet-5-5": dict(label="Claude Sonnet 5.5", family="claude", **{"in": 2.0, "out": 10.0},
                              long_over=None, in_long=None, out_long=None, batch=0.5),
    "claude-opus-5-5": dict(label="Claude Opus 5.5", family="claude", **{"in": 4.0, "out": 20.0},
                            long_over=None, in_long=None, out_long=None, batch=0.5),
    "gpt-6-sol": dict(label="GPT-6 Sol", family="openai", **{"in": 2.0, "out": 10.0},
                      long_over=272_000, in_long=4.0, out_long=15.0, batch=0.5),
    "gpt4o": dict(label="GPT-4o", family="openai", **{"in": 2.50, "out": 10.0},
                  long_over=None, in_long=None, out_long=None, batch=0.5),
    "gpt-4o-mini": dict(label="GPT-4o Mini", family="openai", **{"in": 0.15, "out": 0.60},
                        long_over=None, in_long=None, out_long=None, batch=0.5),
    "claude-3-5-sonnet": dict(label="Claude 3.5 Sonnet", family="claude", **{"in": 3.0, "out": 15.0},
                              long_over=None, in_long=None, out_long=None, batch=0.5),
    "gemini-1-5-flash": dict(label="Gemini 1.5 Flash", family="gemini", **{"in": 0.075, "out": 0.30},
                             long_over=None, in_long=None, out_long=None, batch=0.5),
    "gemini-1-5-pro": dict(label="Gemini 1.5 Pro", family="gemini", **{"in": 1.25, "out": 5.00},
                           long_over=None, in_long=None, out_long=None, batch=0.5),
}

# Expected output tokens from judge templates
JUDGE_OUT = {
    "judge_choice": 15,
    "judge_fillblank": 250,
    "judge_shortanswer": 250,
    "judge_calculation": 300,
    "judge_single_step": 250,
    "judge_multi_step": 450,
    "judge_intermediate": 200,
    "judge_description": 250,
}

# Output floor for descriptive / complex tasks (when generating answers)
VERBOSE_FLOOR = {
    "judge_shortanswer": 250,
    "judge_calculation": 400,
    "judge_single_step": 100,
    "judge_multi_step": 300,
    "judge_intermediate": 60,
    "judge_description": 250,
    "judge_fillblank": 40,
}

# Scenarios: (visible-answer multiplier, reasoning multiplier, percentile for unmeasured tasks)
SCEN = {
    "low": (0.8, 0.5, 25),
    "expected": (1.0, 1.0, 50),
    "high": (1.3, 1.5, 90),
}


# --------------------------------------------------------------------------------------
# Tokenizer & Vision Token Estimators
# --------------------------------------------------------------------------------------
class Tok:
    def __init__(self):
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("o200k_base")
        except Exception as e:
            print(f"[warn] tiktoken unavailable ({str(e)[:60]}); using character heuristic.", file=sys.stderr)

    @property
    def exact(self):
        return self.enc is not None

    def count(self, s) -> int:
        if not s:
            return 0
        s = str(s)
        if self.enc:
            return len(self.enc.encode(s, disallowed_special=()))
        cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", s))
        return int((len(s) - cjk) / 3.0 + cjk * 1.1) + 1


def claude_image_tokens(w, h, max_side=1568, cap=1600):
    """Anthropic formula: (width * height) / 750 tokens, scaled to fit max_side."""
    if not w or not h:
        return 0
    s = min(1.0, max_side / max(w, h))
    w, h = w * s, h * s
    return int(min(cap, (w * h) / 750) + 0.5)


def openai_image_tokens(w, h, max_side=2048, cap=1536):
    """
    OpenAI GPT-4o vision formula:
    Scale within 2048x2048, scale shortest side to 768, then 170 tokens per 512x512 tile + 85 base.
    """
    if not w or not h:
        return 0
    s1 = min(1.0, max_side / max(w, h))
    w, h = w * s1, h * s1
    s2 = 768.0 / min(w, h) if min(w, h) > 0 else 1.0
    w, h = w * s2, h * s2
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return min(cap, 85 + 170 * tiles)


def gemini_image_tokens(w, h):
    """Gemini charges a flat 258 tokens per image under standard resolution."""
    return 258


def _img_size(x):
    from PIL import Image as PILImage
    try:
        if isinstance(x, dict):
            if x.get("bytes"):
                return PILImage.open(io.BytesIO(x["bytes"])).size
            if x.get("path") and os.path.exists(x["path"]):
                return PILImage.open(x["path"]).size
            return None
        if hasattr(x, "size"):
            return tuple(x.size)
        if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            return tuple(x)
        if isinstance(x, str) and os.path.exists(x):
            return PILImage.open(x).size
    except Exception:
        return None
    return None


def norm_q(s: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", s or "").strip().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Multimodal Data Loading
# --------------------------------------------------------------------------------------
def load_mm_rows(source, jsonl_path=None):
    rows = []
    if source == "jsonl" and jsonl_path:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fp = r.get("file_path") or r.get("filename") or ""
                imgs = r.get("image_size") or r.get("images") or []
                if imgs and isinstance(imgs[0], (int, float)):
                    imgs = [imgs]
                rows.append(dict(
                    query=r.get("query", ""),
                    target=r.get("target", ""),
                    file_path=fp,
                    sizes=[s for s in (_img_size(i) for i in imgs) if s],
                ))
        return rows

    # Default: Hugging Face datasets
    from datasets import load_dataset
    ds = load_dataset("Ooo1/ChemEval", split="multimodal")
    for r in ds:
        fp = r.get("file_path") or r.get("filename") or ""
        im = r.get("image")
        size = _img_size(im)
        sizes = [size] if size else []
        rows.append(dict(
            query=r.get("query", ""),
            target=r.get("target", ""),
            file_path=fp,
            sizes=sizes,
        ))
    return rows


# --------------------------------------------------------------------------------------
# Langfuse Calibration
# --------------------------------------------------------------------------------------
@dataclass
class Call:
    model: str
    qkey: str
    in_tok: int
    out_tok: int
    cost: float
    answer: str
    text: str


@dataclass
class Calib:
    model: str
    calls: list
    by_key: dict = field(default_factory=lambda: defaultdict(list))
    mult: float = 1.0
    task_reasoning: dict = field(default_factory=dict)
    task_visible: dict = field(default_factory=dict)
    task_n: dict = field(default_factory=dict)
    pct: dict = field(default_factory=dict)
    matched: int = 0


def _msg_text(raw: str) -> str:
    try:
        d = json.loads(raw)
    except Exception:
        return ""
    msgs = d["messages"] if isinstance(d, dict) and "messages" in d else (d if isinstance(d, list) else [])
    parts = []
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts += [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
    return "\n".join(parts)


def load_langfuse(path: str) -> Calib:
    csv.field_size_limit(sys.maxsize)
    calls = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                u = json.loads(r["usageDetails"])
                cost = float(r["totalCost"])
            except Exception:
                continue
            try:
                o = json.loads(r["output"])
                ans = (o.get("content") if isinstance(o, dict) else str(o)) or ""
            except Exception:
                ans = ""
            q = _msg_text(r["input"])
            calls.append(Call(
                r.get("providedModelName", ""),
                norm_q(q),
                int(u.get("input", 0)),
                int(u.get("output", 0)),
                cost,
                ans,
                q,
            ))
    if not calls:
        sys.exit(f"no usable Langfuse generations found in {path}")
    model = max({c.model for c in calls}, key=lambda m: sum(c.model == m for c in calls))
    cal = Calib(model=model, calls=[c for c in calls if c.model == model])
    for c in cal.calls:
        cal.by_key[c.qkey].append(c)
    return cal


def calibrate(cal: Calib, rows: list, tok: Tok):
    by_task = defaultdict(list)
    ratio_num = ratio_den = 0
    used = defaultdict(int)
    for r in rows:
        calls = cal.by_key.get(r.qkey)
        if not calls or used[r.qkey] >= len(calls):
            continue
        c = calls[used[r.qkey]]
        used[r.qkey] += 1
        r.actual = c
        cal.matched += 1
        ratio_num += c.in_tok
        ratio_den += r.q_tok
        by_task[r.task].append(c)
    if ratio_den:
        cal.mult = ratio_num / ratio_den
    for t, cs in by_task.items():
        vis = [tok.count(c.answer) * cal.mult for c in cs]
        rea = [max(c.out_tok - v, 0.0) for c, v in zip(cs, vis)]
        cal.task_reasoning[t] = statistics.mean(rea)
        cal.task_visible[t] = statistics.mean(vis)
        cal.task_n[t] = len(cs)
    means = sorted(v for t, v in cal.task_reasoning.items() if cal.task_n[t] >= 10) or [0.0]
    for p in (25, 50, 90):
        cal.pct[p] = means[min(len(means) - 1, int(round((p / 100) * (len(means) - 1))))]


# --------------------------------------------------------------------------------------
# Row Data Class & Row Builder
# --------------------------------------------------------------------------------------
@dataclass
class Row:
    modality: str
    task: str
    file_path: str
    cat: str
    qkey: str
    q_tok: int
    gold_tok: int
    sizes: list
    is_judge: bool
    out_est: int
    judge_in_tok: int = 0
    judge_out_tok: int = 0
    actual: object = None


def build_rows(args, tok, consts):
    ctx = ev.Ctx(consts)
    raw_rows = load_mm_rows(args.source, args.jsonl)
    print(f"[load] multimodal ({len(raw_rows)} rows) ...", file=sys.stderr)
    out = []

    for r in raw_rows:
        fp = r["file_path"]
        cat = ev.route(fp)
        if cat is None:
            continue
        task = ev.paper_task(fp)
        is_judge = cat.startswith("judge_")

        gold = str(ev.first(r["target"]))
        gold_tok = tok.count(gold)
        q_tok = tok.count(r["query"])

        if is_judge and cat in VERBOSE_FLOOR:
            out_est = int(max(1.5 * gold_tok, VERBOSE_FLOOR[cat])) + 10
        else:
            out_est = int(1.3 * gold_tok) + 10

        row = Row(
            modality="multimodal",
            task=task,
            file_path=fp,
            cat=cat,
            qkey=norm_q(r["query"]),
            q_tok=q_tok,
            gold_tok=gold_tok,
            sizes=r["sizes"],
            is_judge=is_judge,
            out_est=out_est,
        )

        if is_judge:
            # Estimate judge input tokens using the actual judge template
            proxy = '{"answer": "' + gold + '"}'
            item = ev.Item(0, r["query"], gold, proxy, fp)
            try:
                judge_prompt = ev.build_judge_prompt(cat, item, ctx)
                row.judge_in_tok = tok.count(judge_prompt) - tok.count(proxy)
            except Exception:
                row.judge_in_tok = q_tok + gold_tok + 500
            row.judge_out_tok = JUDGE_OUT.get(cat, 250)

        out.append(row)

    return out


# --------------------------------------------------------------------------------------
# Cost Model
# --------------------------------------------------------------------------------------
def req_cost(model, in_tok, out_tok, batch):
    p = PRICES[model]
    long = p["long_over"] is not None and in_tok > p["long_over"]
    ri, ro = (p["in_long"], p["out_long"]) if long else (p["in"], p["out"])
    c = (in_tok * ri + out_tok * ro) / 1e6
    return c * (p["batch"] if batch else 1.0)


def fam_mult(model, mult):
    return mult if PRICES[model]["family"] == "claude" else 1.0


def input_tokens(model, row, args, mult):
    text = row.q_tok * fam_mult(model, mult)
    img = 0
    family = PRICES[model]["family"]
    for (w, h) in row.sizes:
        if args.image_tokens is not None:
            img += args.image_tokens
        elif family == "claude":
            img += claude_image_tokens(w, h, args.max_side)
        elif family == "gemini":
            img += gemini_image_tokens(w, h)
        else:
            img += openai_image_tokens(w, h, args.max_side)
    return int(text) + img + args.overhead


def output_tokens(model, row, scen, args, cal, mult):
    """Visible answer + hidden reasoning, in model's own tokens."""
    vm, rm, pct = SCEN[scen]
    fm = fam_mult(model, mult)
    if cal is not None and row.task in cal.task_reasoning and cal.task_n[row.task] >= 3:
        vis = cal.task_visible[row.task] / cal.mult * fm
        rea = cal.task_reasoning[row.task] * rm
        measured = True
    else:
        vis = row.out_est * fm
        rea = cal.pct[pct] if cal is not None else {
            "low": 0,
            "expected": args.reasoning_expected,
            "high": args.reasoning_high,
        }[scen]
        measured = False
    if cal is not None and model != cal.model:
        rea *= args.other_reasoning_factor
    return int(vis * vm + rea), measured


def estimate(rows, args, mult, cal):
    res = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for model in args.models:
            for scen in SCEN:
                in_tok = input_tokens(model, row, args, mult)
                k = (row.task, model, scen)
                res[k]["rows"] += 1
                if cal is not None and model == cal.model and row.actual is not None:
                    c = row.actual
                    res[k]["in_tok"] += c.in_tok
                    res[k]["out_tok"] += c.out_tok
                    res[k]["eval_usd"] += c.cost * (PRICES[model]["batch"] if args.batch else 1.0)
                    res[k]["actual_usd"] += c.cost
                    res[k]["actual_rows"] += 1
                else:
                    out_tok, measured = output_tokens(model, row, scen, args, cal, mult)
                    res[k]["in_tok"] += in_tok
                    res[k]["out_tok"] += out_tok
                    res[k]["eval_usd"] += req_cost(model, in_tok, out_tok, args.batch)
                    res[k]["est_rows"] += 1
                    res[k]["unmeasured_rows"] += 0 if measured else 1

                if args.judge and row.is_judge:
                    jm = args.judge_model
                    jreason = args.judge_reasoning * {"low": 0.0, "expected": 1.0, "high": 3.0}[scen]
                    ans_tok = int(row.out_est * fam_mult(jm, mult) * SCEN[scen][0])
                    ji = int(row.judge_in_tok * fam_mult(jm, mult) + ans_tok) + 8
                    jo = int(row.judge_out_tok + jreason)
                    res[k]["judge_calls"] += 1
                    res[k]["judge_usd"] += req_cost(jm, ji, jo, args.batch)
    return res


def roll(res, keep, group=lambda k: "x"):
    out = defaultdict(lambda: defaultdict(float))
    for k, v in res.items():
        if keep(k):
            for m, x in v.items():
                out[group(k)][m] += x
    return out


def usd(x):
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def report(rows, res, args, tok, mult, cal):
    runs = args.runs
    print("\nPrices (USD / 1M tokens):")
    all_models = list(dict.fromkeys(args.models + ([args.judge_model] if args.judge else [])))
    for m in all_models:
        p = PRICES[m]
        tier = f"; >{p['long_over']:,} prompt tokens: {p['in_long']}/{p['out_long']}" if p["long_over"] else ""
        print(f"  {p['label']:18s} in {p['in']:<5} out {p['out']:<5}{tier}")
    print(f"  tokenizer: {'tiktoken o200k_base' if tok.exact else 'heuristic'}; Claude multiplier x{mult:.3f}"
          f"{' (calibrated from billing)' if cal else ' (default)'}; runs={runs}; "
          f"{'BATCH' if args.batch else 'standard'} pricing; judge={PRICES[args.judge_model]['label'] if args.judge else 'off'}")

    if cal is not None:
        done = [r for r in rows if r.actual is not None]
        tot_in = sum(c.in_tok for c in cal.calls)
        tot_out = sum(c.out_tok for c in cal.calls)
        billed = sum(c.cost for c in cal.calls)
        p = PRICES.get(cal.model)
        recomputed = sum(req_cost(cal.model, c.in_tok, c.out_tok, False) for c in cal.calls) if p else float("nan")
        vis = sum(tok.count(c.answer) * cal.mult for c in cal.calls)
        print(f"\nCALIBRATION ({cal.model}, {len(cal.calls):,} billed calls)")
        print(f"  billed {usd(billed)}   price table reproduces it to {usd(abs(billed - recomputed))}")
        print(f"  tokens: input {tot_in:,}   output {tot_out:,} -> visible answer ~{100 * vis / max(tot_out, 1):.0f}% / "
              f"hidden reasoning ~{100 * (1 - vis / max(tot_out, 1)):.0f}% of billed output")
        print(f"  matched to dataset rows: {cal.matched:,}/{len(cal.calls):,} calls ({len(done):,} rows)")

    n_rows = len(rows)
    n_judge = sum(1 for r in rows if r.is_judge)
    print(f"\n=== MULTIMODAL BENCHMARK | {n_rows:,} rows ({n_judge:,} judged) x {runs} run(s) ===")
    print(f"{'model':18s}{'input tok':>12s}{'output tok':>12s} |{'eval exp':>10s}{'(actual part)':>14s} |{'judge':>9s}"
          f"{'TOTAL exp':>11s}{'  range low..high':>22s}")

    grand = defaultdict(lambda: defaultdict(float))
    csv_rows = []

    for model in args.models:
        by = {s: roll(res, lambda k, s=s: k[1] == model and k[2] == s)["x"] for s in SCEN}
        e = by["expected"]
        tot = {s: (by[s]["eval_usd"] + by[s]["judge_usd"]) * runs for s in by}
        act = e["actual_usd"]
        print(f"{PRICES[model]['label']:18s}{e['in_tok'] * runs:12,.0f}{e['out_tok'] * runs:12,.0f} |"
              f"{usd(e['eval_usd'] * runs):>10s}{('(' + usd(act) + ')') if act else '':>14s} |{usd(e['judge_usd'] * runs):>9s}"
              f"{usd(tot['expected']):>11s}{'  ' + usd(tot['low']) + ' .. ' + usd(tot['high']):>22s}")

        for s in by:
            for m in ("rows", "in_tok", "out_tok", "eval_usd", "judge_usd", "actual_usd", "actual_rows", "est_rows"):
                grand[(model, s)][m] += by[s][m] * (1 if m in ("rows", "actual_usd", "actual_rows", "est_rows") else runs)
            csv_rows.append(["multimodal", model, s, int(by[s]["rows"]), by[s]["in_tok"] * runs, by[s]["out_tok"] * runs,
                             by[s]["eval_usd"] * runs, by[s]["judge_usd"] * runs, tot[s], by[s]["actual_usd"]])

    print("\n=== GRAND TOTAL ===")
    print(f"{'model':18s}{'low':>11s}{'expected':>11s}{'high':>11s}   {'already spent':>14s}{'remaining (exp)':>17s}")
    for model in args.models:
        t = {s: grand[(model, s)]["eval_usd"] + grand[(model, s)]["judge_usd"] for s in SCEN}
        spent = grand[(model, "expected")]["actual_usd"]
        print(f"{PRICES[model]['label']:18s}{usd(t['low']):>11s}{usd(t['expected']):>11s}{usd(t['high']):>11s}   "
              f"{usd(spent):>14s}{usd(t['expected'] - spent):>17s}")

    if args.by_task:
        print("\n=== PER-TASK expected cost (eval + judge), runs applied ===")
        tasks = sorted({k[0] for k in res}, key=lambda t: ev.TASK_ORDER.index(t) if t in ev.TASK_ORDER else 999)
        hdr = f"{'task':16s}{'rows':>6s}{'judged':>8s}" + "".join(f"{PRICES[m]['label'][:16]:>18s}" for m in args.models)
        print(hdr)
        for t in tasks:
            r0 = roll(res, lambda k, t=t: k[0] == t and k[1] == args.models[0] and k[2] == "expected")["x"]
            is_j = "yes" if r0["judge_calls"] > 0 else "no"
            line = f"{t:16s}{int(r0['rows']):6d}{is_j:>8s}"
            for m in args.models:
                r = roll(res, lambda k, t=t, m=m: k[0] == t and k[1] == m and k[2] == "expected")["x"]
                line += f"{usd((r['eval_usd'] + r['judge_usd']) * runs):>18s}"
            print(line)

    if args.out:
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "multimodal_cost_estimate.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["modality", "model", "scenario", "rows", "input_tokens", "output_tokens",
                        "eval_usd", "judge_usd", "total_usd", "of_which_actual_usd"])
            w.writerows(csv_rows)
        print(f"\nwrote {od}/multimodal_cost_estimate.csv")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-csv", help="Langfuse events export of a previous run")
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5", "gpt-6-luna", "gpt4o"],
                    choices=list(PRICES))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-model", default="gpt4o", choices=list(PRICES))
    ap.add_argument("--judge-reasoning", type=int, default=150,
                    help="hidden reasoning tokens per judge call")
    ap.add_argument("--other-reasoning-factor", type=float, default=1.0)
    ap.add_argument("--batch", action="store_true", help="use batch API pricing (50%% discount)")
    ap.add_argument("--source", choices=["hf", "jsonl"], default="hf")
    ap.add_argument("--jsonl", help="path to local multimodal JSONL file if source=jsonl")
    ap.add_argument("--constants", default=str(TEXT_EVAL / "repo_constants.json"))
    ap.add_argument("--claude-multiplier", type=float, default=1.35)
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--image-tokens", type=int, default=None,
                    help="override image tokens with a flat count per image")
    ap.add_argument("--overhead", type=int, default=8)
    ap.add_argument("--reasoning-expected", type=int, default=0,
                    help="flat hidden tokens/call when not calibrating")
    ap.add_argument("--reasoning-high", type=int, default=2000)
    ap.add_argument("--by-task", action="store_true", help="show per-task cost breakdown")
    ap.add_argument("--list-files", action="store_true", help="list all files and exit")
    ap.add_argument("--out", default=None, help="directory to export CSV summary")
    args = ap.parse_args()

    consts = json.loads(Path(args.constants).read_text(encoding="utf-8"))
    tok = Tok()
    rows = build_rows(args, tok, consts)

    if not rows:
        sys.exit("No rows loaded - check dataset or jsonl path.")

    if args.list_files:
        agg = defaultdict(int)
        for r in rows:
            agg[(r.file_path, r.cat, r.task, r.is_judge)] += 1
        print(f"{'rows':>5s}  {'category':25s} {'task':16s} {'judged':>7s}  file_path")
        for (fp, cat, task, is_j), n in sorted(agg.items()):
            print(f"{n:5d}  {cat:25s} {task:16s} {('yes' if is_j else 'no'):>7s}  {fp}")
        print(f"\nTotal: {len(rows)} rows across {len(agg)} files")
        return

    cal, mult = None, args.claude_multiplier
    if args.calibration_csv:
        cal = load_langfuse(args.calibration_csv)
        calibrate(cal, rows, tok)
        if cal.matched:
            mult = cal.mult
        else:
            print("[warn] no billed call matched a dataset row; falling back to defaults.", file=sys.stderr)
            cal = None

    res = estimate(rows, args, mult, cal)
    report(rows, res, args, tok, mult, cal)


if __name__ == "__main__":
    main()
