#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chemeval_cost_estimate.py (v2) - cost of running ChemEval on Claude Haiku 5.5 / GPT-6 Luna (or any model in the
price table), calibrated against YOUR real Langfuse billing export.

Why v2: a real run of 953 text rows on Haiku 5.5 cost $0.41, and ~95% of the billed OUTPUT tokens were hidden
reasoning (adaptive thinking is on by default at effort=medium), not the visible {"answer": ...}. The reasoning
volume varies hugely by task (SMILES<->SELFIES ~2,800 tokens/call vs. Yes/No yield ~20). v2 therefore:
  * reads the Langfuse CSV, verifies it against the price table, and measures reasoning tokens PER TASK
  * joins every billed call to its dataset row (exact query match) -> per-task actual cost
  * predicts the rest of the benchmark (other rows, 3-shot, other models) from those measurements
  * reports  ALREADY SPENT (actual)  +  REMAINING (estimate)  =  FULL BENCH, with a low/expected/high range

Needs chemeval_text_eval.py + repo_constants.json next to it (re-uses its routing / shot detection / judge prompts).

  pip install datasets tiktoken pillow
  python chemeval_cost_estimate.py --calibration-csv lf-events-export.csv --by-task
  python chemeval_cost_estimate.py --calibration-csv lf.csv --runs 5 --batch --out cost/
  python chemeval_cost_estimate.py --list-files                 # routing / which files count as judged
Text only by default; add  --modalities text multimodal  for images.
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
sys.path.insert(0, str(HERE))
try:
    import evaluate as ev
except ImportError:
    sys.exit("chemeval_text_eval.py must be in the same folder as this script.")

# --------------------------------------------------------------------------------------
# Price table - USD per 1M tokens, checked 2026-10-30.
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
    
    "gpt4o": dict(
    label="GPT-4o",
    family="openai",
    **{"in": 2.50, "out": 10.0},
    long_over=None,
    in_long=None,
    out_long=None,
    batch=0.5,
),
}
JUDGE_OUT = {"choice": 15, "truefalse": 15, "fillblank": 250, "shortanswer": 250, "calculation": 300,
             "abstract": 300, "outline": 300, "physchem": 250, "single_step": 250, "multi_step": 450,
             "intermediate": 200, "path_unknown": 300, "mm_qa": 250}
VERBOSE_FLOOR = {"shortanswer": 250, "calculation": 400, "abstract": 250, "outline": 150, "physchem": 250,
                 "single_step": 100, "multi_step": 300, "intermediate": 60, "path_unknown": 200,
                 "fillblank": 40, "mm_qa": 200}
MM_JUDGED_DEFAULT = (r"QA|问答|谱|路径|中间体|描述|物理化学|简答|计算|摘要|提纲|Pathway|Intermediate|"
                     r"Short|Calc|Spectrum|IRS|RS2|UV|DP2|MS2|NMR|识别")
# (visible-answer multiplier, reasoning multiplier, percentile used for tasks never measured)
SCEN = {"low": (0.8, 0.5, 25), "expected": (1.0, 1.0, 50), "high": (1.3, 1.5, 90)}


# --------------------------------------------------------------------------------------
# tokens / images
# --------------------------------------------------------------------------------------
class Tok:
    def __init__(self):
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("o200k_base")
        except Exception as e:
            print(f"[warn] tiktoken unavailable ({str(e)[:60]}); using a character heuristic. The Claude multiplier "
                  f"is calibrated against it, so Claude estimates stay consistent.", file=sys.stderr)

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


def claude_image_tokens(w, h, max_side, cap=1600):
    s = min(1.0, max_side / max(w, h))
    w, h = w * s, h * s
    s2 = min(1.0, 1568 / max(w, h))
    return int(min(cap, (w * s2) * (h * s2) / 750) + 0.5)


def openai_image_tokens(w, h, max_side, cap=1536):
    s = min(1.0, max_side / max(w, h))
    return min(cap, math.ceil(w * s / 32) * math.ceil(h * s / 32))


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


def load_rows(source, split, jsonl_path):
    rows = []
    if source == "jsonl":
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                imgs = r.get("image_size") or r.get("images") or []
                if imgs and isinstance(imgs[0], (int, float)):
                    imgs = [imgs]
                rows.append(dict(query=r.get("query", ""), target=r.get("target", ""), filename=r.get("filename", ""),
                                 sizes=[s for s in (_img_size(i) for i in imgs) if s], shot=r.get("shot")))
        return rows
    from datasets import Image as HFImage, load_dataset
    ds = load_dataset("Ooo1/ChemEval", split=split)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage(decode=False))
    for r in ds:
        im = r.get("image")
        ims = im if isinstance(im, list) else ([im] if im else [])
        rows.append(dict(query=r["query"], target=r["target"], filename=r["filename"],
                         sizes=[s for s in (_img_size(i) for i in ims) if s], shot=None))
    return rows


# --------------------------------------------------------------------------------------
# Langfuse calibration
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
    task_reasoning: dict = field(default_factory=dict)   # task -> mean hidden tokens (Claude tokens)
    task_visible: dict = field(default_factory=dict)     # task -> mean visible tokens (Claude tokens)
    task_n: dict = field(default_factory=dict)
    pct: dict = field(default_factory=dict)              # 25/50/90 percentile of per-task mean reasoning
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
            calls.append(Call(r.get("providedModelName", ""), norm_q(q), int(u.get("input", 0)), int(u.get("output", 0)),
                              cost, ans, q))
    if not calls:
        sys.exit(f"no usable Langfuse generations found in {path}")
    model = max({c.model for c in calls}, key=lambda m: sum(c.model == m for c in calls))
    cal = Calib(model=model, calls=[c for c in calls if c.model == model])
    for c in cal.calls:
        cal.by_key[c.qkey].append(c)
    return cal


def calibrate(cal: Calib, rows: list, tok: Tok):
    """Join calls to dataset rows (exact query match) and measure token multiplier + per-task reasoning."""
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
# rows
# --------------------------------------------------------------------------------------
@dataclass
class Row:
    modality: str
    shot: str
    task: str
    filename: str
    qkey: str
    q_tok: int
    gold_tok: int
    sizes: list
    kind: str | None
    out_est: int
    judge_in_tok: int = 0
    judge_out_tok: int = 0
    actual: object = None


def build_rows(args, tok, consts):
    ctx = ev.Ctx(consts)
    mm_re = re.compile(args.mm_judged_regex)
    qre = re.compile(args.three_shot_query_regex) if args.three_shot_query_regex else None
    out = []
    for modality in args.modalities:
        split = "text" if modality == "text" else "multimodal"
        jsonl = args.text_jsonl if modality == "text" else args.mm_jsonl
        if args.source == "jsonl" and not jsonl:
            continue
        print(f"[load] {modality} ...", file=sys.stderr)
        for r in load_rows(args.source, split, jsonl):
            fn = r["filename"]
            shot = ev.detect_shot(fn, r["query"], r.get("shot"), qre) if modality == "text" else "0shot"
            if shot[0] not in args.shots:
                continue
            kind = None
            if modality == "text":
                if ev.route(fn) == "judge":
                    kind = ev.judge_kind(fn)
            else:
                kind = ev.judge_kind(fn) if ev.route(fn) == "judge" else ("mm_qa" if mm_re.search(fn) else None)
            gold = str(ev.first(r["target"]))
            gold_tok = tok.count(gold)
            out_est = int(max(1.5 * gold_tok, VERBOSE_FLOOR[kind])) + 10 if kind in VERBOSE_FLOOR else int(1.3 * gold_tok) + 10
            row = Row(modality, shot, ev.paper_task(fn), fn, norm_q(r["query"]), tok.count(r["query"]), gold_tok,
                      r["sizes"], kind, out_est)
            if kind and kind != "mm_qa" and modality == "text":
                proxy = '{"answer": "' + gold + '"}'
                item = ev.Item(0, r["query"], gold, proxy, fn, shot)
                k2 = ev.resolve_kind(kind, item)
                row.kind = k2
                row.judge_in_tok = tok.count(ev.build_prompt(k2, item, ctx)) - tok.count(proxy)
                row.judge_out_tok = JUDGE_OUT.get(k2, 250)
                if k2 in VERBOSE_FLOOR:
                    row.out_est = int(max(1.5 * gold_tok, VERBOSE_FLOOR[k2])) + 10
            elif kind == "mm_qa":
                row.judge_in_tok = args.mm_judge_overhead + row.q_tok + gold_tok
                row.judge_out_tok = JUDGE_OUT["mm_qa"]
            out.append(row)
    return out


# --------------------------------------------------------------------------------------
# cost model
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
    for (w, h) in row.sizes:
        if args.image_tokens is not None:
            img += args.image_tokens
        elif PRICES[model]["family"] == "claude":
            img += claude_image_tokens(w, h, args.max_side)
        else:
            img += openai_image_tokens(w, h, args.max_side)
    return int(text) + img + args.overhead


def output_tokens(model, row, scen, args, cal, mult):
    """visible answer + hidden reasoning, in `model`'s own tokens."""
    vm, rm, pct = SCEN[scen]
    fm = fam_mult(model, mult)
    if cal is not None and row.task in cal.task_reasoning and cal.task_n[row.task] >= 3:
        vis = cal.task_visible[row.task] / cal.mult * fm          # measured, converted to this model's tokens
        rea = cal.task_reasoning[row.task] * rm
        measured = True
    else:
        vis = row.out_est * fm
        rea = cal.pct[pct] if cal is not None else {"low": 0, "expected": args.reasoning_expected, "high": args.reasoning_high}[scen]
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
                k = (row.modality, row.shot, row.task, model, scen)
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
                if args.judge and row.kind:
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
def report(rows, res, args, tok, mult, cal):
    runs = args.runs
    print("\nPrices (USD / 1M tokens):")
    for m in args.models + ([args.judge_model] if args.judge and args.judge_model not in args.models else []):
        p = PRICES[m]
        tier = f"; >{p['long_over']:,} prompt tokens: {p['in_long']}/{p['out_long']}" if p["long_over"] else ""
        print(f"  {p['label']:18s} in {p['in']:<5} out {p['out']:<5}{tier}")
    print(f"  tokenizer: {'tiktoken o200k_base' if tok.exact else 'heuristic'}; Claude token multiplier x{mult:.3f}"
          f"{' (measured from your billing)' if cal else ' (default; pass --calibration-csv)'}; runs={runs}; "
          f"{'BATCH' if args.batch else 'standard'} pricing; judge={PRICES[args.judge_model]['label'] if args.judge else 'off'}")

    if cal is not None:
        done = [r for r in rows if r.actual is not None]
        tot_in = sum(c.in_tok for c in cal.calls)
        tot_out = sum(c.out_tok for c in cal.calls)
        billed = sum(c.cost for c in cal.calls)
        p = PRICES.get(cal.model)
        recomputed = sum(req_cost(cal.model, c.in_tok, c.out_tok, False) for c in cal.calls) if p else float("nan")
        vis = sum(tok.count(c.answer) * cal.mult for c in cal.calls)
        print(f"\nCALIBRATION  ({cal.model}, {len(cal.calls):,} billed calls)")
        print(f"  billed {usd(billed)}   price table reproduces it to {usd(abs(billed - recomputed))}")
        print(f"  tokens: input {tot_in:,}   output {tot_out:,}   ->  visible answer ~{100 * vis / max(tot_out, 1):.0f}% / "
              f"hidden reasoning ~{100 * (1 - vis / max(tot_out, 1)):.0f}% of billed output; output = "
              f"{100 * tot_out * p['out'] / max(tot_in * p['in'] + tot_out * p['out'], 1):.0f}% of the bill" if p else "")
        print(f"  matched to dataset rows: {cal.matched:,}/{len(cal.calls):,} calls ({len(done):,} rows)"
              + ("   <-- LOW: check --source/--shots, or the CSV is from a different prompt version" if cal.matched < 0.9 * len(cal.calls) else ""))
        print(f"  per-task measured reasoning (mean billed output tokens beyond the visible answer):")
        print(f"    {'task':14s}{'n':>5s}{'reasoning':>11s}{'visible':>9s}{'$/call':>10s}")
        by = defaultdict(list)
        for r in done:
            by[r.task].append(r.actual)
        for t in sorted(by, key=lambda t: -cal.task_reasoning[t]):
            cs = by[t]
            print(f"    {t:14s}{len(cs):5d}{cal.task_reasoning[t]:11.0f}{cal.task_visible[t]:9.0f}"
                  f"{statistics.mean(c.cost for c in cs):10.5f}")
        print(f"  tasks never measured fall back to the per-task reasoning percentiles: "
              f"low {cal.pct[25]:.0f} / expected {cal.pct[50]:.0f} / high {cal.pct[90]:.0f} tokens per call")

    groups = sorted({(k[0], k[1]) for k in res})
    grand = defaultdict(lambda: defaultdict(float))
    csv_rows = []
    for (mod, shot) in groups:
        sel = lambda k, mod=mod, shot=shot: k[0] == mod and k[1] == shot
        n = int(roll(res, lambda k: sel(k) and k[3] == args.models[0] and k[4] == "expected")["x"]["rows"])
        nj = int(roll(res, lambda k: sel(k) and k[3] == args.models[0] and k[4] == "expected")["x"]["judge_calls"])
        print(f"\n=== {mod.upper()} | {shot.replace('shot', '-shot')} | {n:,} rows ({nj:,} judged) x {runs} run(s) ===")
        print(f"{'model':18s}{'input tok':>12s}{'output tok':>12s} |{'eval exp':>10s}{'(actual part)':>14s} |{'judge':>9s}"
              f"{'TOTAL exp':>11s}{'  range low..high':>22s}")
        for model in args.models:
            by = {s: roll(res, lambda k, s=s: sel(k) and k[3] == model and k[4] == s)["x"] for s in SCEN}
            e = by["expected"]
            tot = {s: (by[s]["eval_usd"] + by[s]["judge_usd"]) * runs for s in by}
            act = e["actual_usd"]
            print(f"{PRICES[model]['label']:18s}{e['in_tok'] * runs:12,.0f}{e['out_tok'] * runs:12,.0f} |"
                  f"{usd(e['eval_usd'] * runs):>10s}{('(' + usd(act) + ')') if act else '':>14s} |{usd(e['judge_usd'] * runs):>9s}"
                  f"{usd(tot['expected']):>11s}{'  ' + usd(tot['low']) + ' .. ' + usd(tot['high']):>22s}")
            for s in by:
                for m in ("rows", "in_tok", "out_tok", "eval_usd", "judge_usd", "actual_usd", "actual_rows", "est_rows"):
                    grand[(model, s)][m] += by[s][m] * (1 if m in ("rows", "actual_usd", "actual_rows", "est_rows") else runs)
                csv_rows.append([mod, shot, model, s, int(by[s]["rows"]), by[s]["in_tok"] * runs, by[s]["out_tok"] * runs,
                                 by[s]["eval_usd"] * runs, by[s]["judge_usd"] * runs, tot[s], by[s]["actual_usd"]])

    print("\n=== GRAND TOTAL ===")
    print(f"{'model':18s}{'low':>11s}{'expected':>11s}{'high':>11s}   {'already spent':>14s}{'remaining (exp)':>17s}")
    for model in args.models:
        t = {s: grand[(model, s)]["eval_usd"] + grand[(model, s)]["judge_usd"] for s in SCEN}
        spent = grand[(model, "expected")]["actual_usd"]
        print(f"{PRICES[model]['label']:18s}{usd(t['low']):>11s}{usd(t['expected']):>11s}{usd(t['high']):>11s}   "
              f"{usd(spent):>14s}{usd(t['expected'] - spent):>17s}")

    if args.by_task:
        print("\n=== PER-TASK expected cost (eval + judge, all shots), runs applied ===")
        tasks = sorted({k[2] for k in res}, key=lambda t: ev.TASK_ORDER.index(t) if t in ev.TASK_ORDER else 999)
        hdr = f"{'task':14s}{'rows':>6s}{'measured':>9s}" + "".join(f"{PRICES[m]['label'][:16]:>18s}" for m in args.models)
        print(hdr)
        for t in tasks:
            r0 = roll(res, lambda k, t=t: k[2] == t and k[3] == args.models[0] and k[4] == "expected")["x"]
            line = f"{t:14s}{int(r0['rows']):6d}{int(r0['actual_rows']):9d}"
            for m in args.models:
                r = roll(res, lambda k, t=t, m=m: k[2] == t and k[3] == m and k[4] == "expected")["x"]
                line += f"{usd((r['eval_usd'] + r['judge_usd']) * runs):>18s}"
            print(line)

    if args.out:
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "cost_estimate.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["modality", "shot", "model", "scenario", "rows", "input_tokens", "output_tokens",
                        "eval_usd", "judge_usd", "total_usd", "of_which_actual_usd"])
            w.writerows(csv_rows)
        print(f"\nwrote {od}/cost_estimate.csv")

    unmeasured = sum(v["unmeasured_rows"] for k, v in res.items() if k[3] == args.models[0] and k[4] == "expected")
    print("\nHow to read this:")
    print(" * 'actual' = real billed numbers from your CSV (only for the model in the CSV); everything else is an estimate.")
    if cal is not None:
        print(f" * {int(unmeasured):,} estimated rows belong to tasks absent from your CSV; they use the percentile fallback, "
              f"so the high/low band is wide there. Run a few rows of each missing task and re-export to tighten it.")
    print(" * Reasoning is the cost driver. Lowering effort (output_config.effort=low) or disabling thinking cuts it, but it also "
          "changes accuracy - decide that before the full run, not after. Retries and failed calls are not included.")
    if cal is not None and any(c.out_tok == 0 for c in cal.calls):
        print(f" * {sum(c.out_tok == 0 for c in cal.calls)} calls in your CSV returned 0 output tokens (empty answers): "
              f"re-run those rows; the generation script in this kit retries empty/truncated outputs.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-csv", help="Langfuse events export of a previous run (any model in the price table)")
    ap.add_argument("--models", nargs="+", default=["claude-haiku-5-5", "gpt-6-luna"], choices=list(PRICES))
    ap.add_argument("--modalities", nargs="+", default=["text"], choices=["text", "multimodal"])
    ap.add_argument("--shots", nargs="+", default=["0", "3"], choices=["0", "3"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-model", default="gpt4o", choices=list(PRICES))
    ap.add_argument("--judge-reasoning", type=int, default=150, help="hidden reasoning tokens per judge call (assumption)")
    ap.add_argument("--other-reasoning-factor", type=float, default=1.0,
                    help="models other than the calibrated one think this many times as much (Luna at effort=none -> 0)")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--source", choices=["hf", "jsonl"], default="hf")
    ap.add_argument("--text-jsonl")
    ap.add_argument("--mm-jsonl")
    ap.add_argument("--three-shot-query-regex")
    ap.add_argument("--claude-multiplier", type=float, default=1.35, help="used only without --calibration-csv")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--image-tokens", type=int, default=None)
    ap.add_argument("--overhead", type=int, default=8)
    ap.add_argument("--reasoning-expected", type=int, default=0, help="flat hidden tokens/call when NOT calibrating")
    ap.add_argument("--reasoning-high", type=int, default=2000)
    ap.add_argument("--mm-judged-regex", default=MM_JUDGED_DEFAULT)
    ap.add_argument("--mm-judge-overhead", type=int, default=1500)
    ap.add_argument("--by-task", action="store_true")
    ap.add_argument("--list-files", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    consts = json.loads((HERE / "repo_constants.json").read_text(encoding="utf-8"))
    tok = Tok()
    rows = build_rows(args, tok, consts)
    if not rows:
        sys.exit("no rows loaded - check --source / --*-jsonl / --shots")

    if args.list_files:
        agg = defaultdict(int)
        for r in rows:
            agg[(r.modality, r.shot, r.filename, ev.route(r.filename), r.kind)] += 1
        print(f"{'mod':10s}{'shot':6s}{'rows':>5s}  {'route':20s}{'judge kind':14s}file")
        for (m, s, fn, rt, kd), n in sorted(agg.items(), key=lambda x: str(x)):
            print(f"{m:10s}{s:6s}{n:5d}  {str(rt):20s}{str(kd):14s}{fn}")
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