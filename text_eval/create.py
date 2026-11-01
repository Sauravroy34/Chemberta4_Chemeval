import argparse, json, os, time
from datasets import load_dataset
from model import LiteLLMModel, HuggingFaceModel, AnthropicBatchModel, OpenAIBatchModel
from warnings import warn
BATCH_BACKENDS = {"anthropic-batch": AnthropicBatchModel, "openai-batch": OpenAIBatchModel}


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


def load_ds(limit):
    ds = load_dataset("Ooo1/ChemEval", split="text").select_columns(["query", "target", "filename"])
    if limit:
        ds = ds.select(range(limit))               # original order is preserved
    return ds


def row_id(i):
    return f"row-{i}"                              # valid for both Anthropic and OpenAI custom_id


# ------------------------------------------------------------------ batch: submit
def batch_submit(args, ds, state_path):
    if os.path.exists(state_path):
        raise SystemExit(f"{state_path} already exists (a batch was already submitted). "
                         "Use --batch-action retrieve, or delete that file to submit again.")
    model = BATCH_BACKENDS[args.backend](args.model, max_new_tokens=args.max_new_tokens)
    queries = ds["query"]
    batch_id = model.submit(args.sys_prompt, queries, [row_id(i) for i in range(len(queries))])
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "model": args.model, "batch_id": batch_id,
                   "total": len(queries), "limit": args.limit}, f, indent=2)
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

    ds = load_ds(st["limit"])
    assert len(ds) == st["total"], "dataset size differs from when the batch was submitted"
    answers, errors = model.results(st["batch_id"])

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:    # same record format as the sync path
        for i, row in enumerate(ds):
            rec = {"query": row["query"], "target": row["target"],
                   "filename": row["filename"], "answer": answers.get(row_id(i), "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
    ap.add_argument("--backend", choices=["litellm", "hf", *BATCH_BACKENDS], required=True)
    ap.add_argument("--model", default=None, help="litellm model string, HF repo/path, or API model id")
    ap.add_argument("--name", required=True, help="short id: no underscores, must not end in 't'")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default 2048 (unset for openai-batch)")
    ap.add_argument("--max-model-len", type=int, default=None, help="vLLM context length")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--sys-prompt", default="")
    ap.add_argument("--batch-action", choices=["submit", "retrieve", "status"], default="submit",
                    help="for *-batch backends: submit a job, or retrieve its results into the json file")
    ap.add_argument("--wait", action="store_true", help="with retrieve/status: poll until the batch ends")
    ap.add_argument("--poll-interval", type=int, default=60)
    args = ap.parse_args()

    if "_" in args.name or args.name.endswith("t"):
        raise SystemExit("--name must have no underscores and must not end with 't'")

    is_batch = args.backend in BATCH_BACKENDS
    if args.max_new_tokens is None and args.backend != "openai-batch":
        warn("WARNING: --max-new-tokens not set; using 32000")
        args.max_new_tokens = 32000
    if not (is_batch and args.batch_action != "submit") and not args.model:
        raise SystemExit("--model is required")

    root = f"dataset_1times/{args.name}/0shot"
    os.makedirs(root, exist_ok=True)
    out_path = f"{root}/{args.name}_1t.json"
    state_path = f"{root}/{args.name}_batch.json"

    if is_batch:
        if args.batch_action == "submit":
            batch_submit(args, load_ds(args.limit), state_path)
        else:
            batch_retrieve(args, out_path, state_path)
        return

    ds = load_ds(args.limit)
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