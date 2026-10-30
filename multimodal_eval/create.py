"""
create.py - Run a multimodal vision LLM on the ChemEval multimodal split.

Produces a JSONL output file where each line has:
    {"query": ..., "target": ..., "file_path": ..., "img_name": ..., "answer": <model output>}

Supports resume: if the output file already exists, completed rows are skipped.

Usage:
    python create.py --model gpt-4o --name gpt4o --limit 5   # smoke test
    python create.py --model gpt-4o --name gpt4o              # full run
"""
import argparse
import json
import os

from datasets import load_dataset
from model import MultimodalModel


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="litellm model string (e.g. gpt-4o, claude-sonnet-4-20250514)")
    ap.add_argument("--name", required=True,
                    help="short run id (no underscores)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows only (for smoke testing)")
    ap.add_argument("--sys-prompt", default="",
                    help="optional system prompt")
    args = ap.parse_args()

    # Output directory
    out_dir = f"results/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.name}_multimodal.jsonl"

    # Load multimodal split
    ds = load_dataset("Ooo1/ChemEval", split="multimodal")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    total = len(ds)

    # Resume from where we left off
    start = load_done(out_path)
    print(f"{start}/{total} rows already done, resuming from row {start}")

    model = MultimodalModel(args.model, temperature=0.0)

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(start, total):
            row = ds[i]
            query = row["query"]
            image = row["image"]
            target = row["target"]
            file_path = row["file_path"]
            img_name = row["img_name"]

            answer = model.generate(args.sys_prompt, query, image)

            rec = {
                "query": query,
                "target": target,
                "file_path": file_path,
                "img_name": img_name,
                "answer": answer,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"{i + 1}/{total}")

    print(f"Approx. cost: ${model.cost:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
