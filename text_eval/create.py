import argparse, json, os
from datasets import load_dataset
from model import LiteLLMModel, HuggingFaceModel


def load_done(path):
    """Keep only the valid, in-order prefix of an existing output file (for resume)."""
    good = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    good.append(json.loads(line))
                except Exception:
                    break          # truncated last line from a crash
        with open(path, "w", encoding="utf-8") as f:
            for rec in good:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["litellm", "hf"], required=True)
    ap.add_argument("--model", required=True, help="litellm model string or HF repo/path")
    ap.add_argument("--name", required=True, help="short id: no underscores, must not end in 't'")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=None, help="vLLM context length")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--sys-prompt", default="")
    args = ap.parse_args()

    if "_" in args.name or args.name.endswith("t"):
        raise SystemExit("--name must have no underscores and must not end with 't'")

    root = f"dataset_1times/{args.name}/0shot"
    os.makedirs(root, exist_ok=True)
    out_path = f"{root}/{args.name}_1t.json"

    ds = load_dataset("Ooo1/ChemEval", split="text").select_columns(["query", "target", "filename"])
    if args.limit:
        ds = ds.select(range(args.limit))          # original order is preserved
    total = len(ds)

    start = load_done(out_path)
    print(f"{start}/{total} rows already done, resuming from row {start}")

    if args.backend == "litellm":
        model = LiteLLMModel(args.model, temperature=0.0,
                             max_new_tokens=args.max_new_tokens)
        gen = lambda qs: [model.generate(args.sys_prompt, q)[0] for q in qs]
    else:
        model = HuggingFaceModel(temperature=0.0, model_name=args.model,
                                 max_new_tokens=args.max_new_tokens,
                                 max_model_len=args.max_model_len)
        gen = lambda qs: model.generate_batch(args.sys_prompt, qs)

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(start, total, args.batch_size):
            rows = ds[i:i + args.batch_size]       # dict of lists
            answers = gen(rows["query"])
            for q, t, fn, a in zip(rows["query"], rows["target"], rows["filename"], answers):
                f.write(json.dumps({"query": q, "target": t, "filename": fn, "answer": a},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + args.batch_size, total)}/{total}")

    if args.backend == "litellm":
        print(f"Approx. cost: ${model.cost:.4f}")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()