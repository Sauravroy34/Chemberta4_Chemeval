"""
create.py - Run a multimodal vision LLM on the ChemEval multimodal split.

Produces a JSONL output file where each line has:
    {"query": ..., "target": ..., "file_path": ..., "img_name": ..., "answer": <model output>}

Supports resume: if the output file already exists, completed rows are skipped.

Usage:
    python create.py --model gpt-4o --name gpt4o --limit 5          # litellm smoke test
    python create.py --model gpt-4o --name gpt4o                     # litellm full run
    python create.py --backend hf --model Qwen/Qwen2-VL-7B-Instruct --name qwen2vl
    python create.py --backend anthropic-batch --model claude-sonnet-4-5 --name claude   # submit
    python create.py --backend anthropic-batch --name claude --batch-action retrieve --wait
"""
import argparse
import json
import os
import time

from datasets import load_dataset
from model import (MultimodalModel, HuggingFaceMultimodalModel,
                   AnthropicMultimodalBatchModel, OpenAIMultimodalBatchModel)

BATCH_BACKENDS = {"anthropic-batch": AnthropicMultimodalBatchModel,
                  "openai-batch": OpenAIMultimodalBatchModel}


def load_done(path):
    """Return count of valid JSONL lines already written (for resume)."""
    good = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    good.append(json.loads(line))
                except Exception:
                    break  # truncated last line from a crash
        # Re-write only good lines (fix partial writes)
        with open(path, "w", encoding="utf-8") as f:
            for rec in good:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(good)


def load_ds(limit):
    ds = load_dataset("Ooo1/ChemEval", split="multimodal")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def row_id(i):
    return f"row-{i}"          # valid custom_id for both Anthropic and OpenAI


def make_record(row, answer):
    return {"query": row["query"], "target": row["target"],
            "file_path": row["file_path"], "img_name": row["img_name"], "answer": answer}


# ------------------------------------------------------------------ batch: submit
def batch_submit(args, ds, state_path):
    if os.path.exists(state_path):
        raise SystemExit(f"{state_path} already exists (a batch was already submitted). "
                         "Use --batch-action retrieve, or delete that file to submit again.")
    model = BATCH_BACKENDS[args.backend](args.model, max_new_tokens=args.max_new_tokens)
    items = ((row_id(i), ds[i]["query"], ds[i]["image"]) for i in range(len(ds)))
    batch_id = model.submit(args.sys_prompt, items)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "model": args.model, "batch_id": batch_id,
                   "total": len(ds), "limit": args.limit}, f, indent=2)
    print(f"Saved batch info to {state_path}")
    print("Later run the same command with:  --batch-action retrieve  (add --wait to poll)")


# ------------------------------------------------------------------ batch: retrieve
def batch_retrieve(args, out_path, state_path):
    if not os.path.exists(state_path):
        raise SystemExit(f"{state_path} not found - submit a batch first.")
    with open(state_path, encoding="utf-8") as f:
        st = json.load(f)
    model = BATCH_BACKENDS[st["backend"]](st["model"])

    while True:
        state, info = model.status(st["batch_id"])
        print(f"[{st['batch_id']}] {info}")
        if state != "running" or not args.wait:
            break
        time.sleep(args.poll_interval)

    if state == "running":
        print("Batch still running. Re-run with --batch-action retrieve later (or add --wait).")
        return
    if state == "failed":
        raise SystemExit("Batch failed; nothing to retrieve.")
    if args.batch_action == "status":
        return

    answers, errors = model.results(st["batch_id"])

    # Only text columns -> the (large) images are never decoded here
    ds = load_ds(st["limit"]).select_columns(["query", "target", "file_path", "img_name"])
    assert len(ds) == st["total"], "dataset size differs from when the batch was submitted"

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:     # same record format as the sync path
        for i, row in enumerate(ds):
            f.write(json.dumps(make_record(row, answers.get(row_id(i), "")),
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    missing = [row_id(i) for i in range(len(ds)) if row_id(i) not in answers]
    print(f"Wrote {out_path}: {len(ds) - len(missing)}/{len(ds)} answered")
    if missing:
        print(f"WARNING: {len(missing)} rows failed and were written with an empty answer, "
              f"e.g. {missing[:5]}")
        for cid in missing[:5]:
            print("  ", cid, "->", errors.get(cid, "no result returned"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["litellm", "hf", *BATCH_BACKENDS], default="litellm")
    ap.add_argument("--model", default=None,
                    help="litellm model string (e.g. gpt-4o), HF repo/path, or API model id")
    ap.add_argument("--name", required=True,
                    help="short run id (no underscores)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default 2048 for hf / anthropic-batch; unset for openai-batch")
    ap.add_argument("--max-model-len", type=int, default=None, help="vLLM context length")
    ap.add_argument("--batch-size", type=int, default=16, help="rows per vLLM call (hf backend)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows only (for smoke testing)")
    ap.add_argument("--sys-prompt", default="",
                    help="optional system prompt")
    ap.add_argument("--batch-action", choices=["submit", "retrieve", "status"], default="submit",
                    help="for *-batch backends: submit a job, or retrieve its results into the jsonl file")
    ap.add_argument("--wait", action="store_true", help="with retrieve/status: poll until the batch ends")
    ap.add_argument("--poll-interval", type=int, default=60)
    args = ap.parse_args()

    is_batch = args.backend in BATCH_BACKENDS
    if args.max_new_tokens is None and args.backend in ("hf", "anthropic-batch"):
        args.max_new_tokens = 2048
    if not (is_batch and args.batch_action != "submit") and not args.model:
        raise SystemExit("--model is required")

    # Output directory
    out_dir = f"results/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.name}_multimodal.jsonl"
    state_path = f"{out_dir}/{args.name}_multimodal_batch.json"

    if is_batch:
        if args.batch_action == "submit":
            batch_submit(args, load_ds(args.limit), state_path)
        else:
            batch_retrieve(args, out_path, state_path)
        return

    # Load multimodal split
    ds = load_ds(args.limit)
    total = len(ds)

    # Resume from where we left off
    start = load_done(out_path)
    print(f"{start}/{total} rows already done, resuming from row {start}")

    if args.backend == "litellm":
        model = MultimodalModel(args.model, temperature=0.0)
        bs = 1
        gen = lambda rows: [model.generate(args.sys_prompt, r["query"], r["image"]) for r in rows]
    else:
        model = HuggingFaceMultimodalModel(temperature=0.0, model_name=args.model,
                                           max_new_tokens=args.max_new_tokens,
                                           max_model_len=args.max_model_len)
        bs = args.batch_size
        gen = lambda rows: model.generate_batch(args.sys_prompt,
                                                [r["query"] for r in rows],
                                                [r["image"] for r in rows])

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(start, total, bs):
            rows = [ds[j] for j in range(i, min(i + bs, total))]
            answers = gen(rows)
            for row, a in zip(rows, answers):
                f.write(json.dumps(make_record(row, a), ensure_ascii=False) + "\n")
            f.flush()

            done = min(i + bs, total)
            if bs > 1 or done % 10 == 0 or done == total:
                print(f"{done}/{total}")

    if args.backend == "litellm":
        print(f"Approx. cost: ${model.cost:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()