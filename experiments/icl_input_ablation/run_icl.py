"""Balanced few-shot ICL on the paper's input-ablation task (inference only).

Builds k-shot chat prompts for Qwen3-8B from the paper's zero-shot template
(config/input_ablation/zs_qwen_qwen_hint.yaml), greedily generates explanations
for the test split of Transluce/input_ablation_qwen3_8b_mmlu_hint, and writes
one predictions file per (k, seed) condition plus results/run_metadata.json.

Demonstrations come from the TRAIN split only; evaluation rows from TEST only.

`--inspect` loads only the dataset and tokenizer (no torch, no GPU, no model
weights), prints the schema and fully rendered prompts, and exits. Run it on
the HPC login node before spending any GPU time.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

DATASET_ID = "Transluce/input_ablation_qwen3_8b_mmlu_hint"
DEMO_SPLIT = "train"  # demonstrations only ever come from here
EVAL_SPLIT = "test"   # evaluation only ever runs on this

REQUIRED_FIELDS = {
    "system_prompt",
    "hint_user_prompt",
    "random_hint_prediction",
    "zeroshot_prediction",
    "changed_pred_bc_hint",
}

# Copied verbatim from config/input_ablation/zs_qwen_qwen_hint.yaml: the yaml
# user message is "{user_prompt}\n" followed by this block, including the
# trailing newline. verify_against_yaml() re-checks the copy is exact.
INSTRUCTION_BLOCK = """If the hint were removed how would the assistant answer change?

Respond with exactly one of the two options below, and nothing else:

The output would remain unchanged from <<<Answer: X>>>.
The output would change to <<<Answer: X>>>.

Replace X with the answer letter (A, B, C, or D).
Do not include any explanations, reasoning, or extra text.

Example outputs:
The output would remain unchanged from <<<Answer: C>>>.
The output would change to <<<Answer: B>>>.
"""

# Gold strings use the format the yaml prompt instructs the model to emit.
# The upstream dataloader's changed-case training target instead reads
# "The most likely output would change to <<<...>>>." (dataloaders/causal.py);
# the parser below accepts both phrasings, so scoring is unaffected.
GOLD_UNCHANGED_TEMPLATE = "The output would remain unchanged from <<<Answer: {letter}>>>."
GOLD_CHANGED_TEMPLATE = "The output would change to <<<Answer: {letter}>>>."

LETTER_RE = re.compile(r"[ABCD]")
CHANGED_PHRASE_RE = re.compile(r"remain\s+unchanged|change\s+to", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<<<\s*Answer:\s*([ABCD])\s*>>>", re.IGNORECASE)


def extract_letter(s):
    """Last A-D letter in a prediction string ('B', ' B', 'Answer: B'), else None."""
    if not isinstance(s, str):
        return None
    found = LETTER_RE.findall(s)
    return found[-1] if found else None


def parse_output(raw):
    """Parse a model output into (pred_changed, pred_letter).

    pred_changed: False for 'remain unchanged', True for 'change to' (leftmost
    phrase wins), None if neither phrase occurs. pred_letter: the letter of the
    first <<<Answer: X>>> tag, None if absent. Never raises on garbage.
    """
    if not isinstance(raw, str):
        return None, None
    m = CHANGED_PHRASE_RE.search(raw)
    pred_changed = None if m is None else not m.group(0).lower().startswith("remain")
    tags = ANSWER_TAG_RE.findall(raw)
    pred_letter = tags[0].upper() if tags else None
    return pred_changed, pred_letter


def gold_fields(row):
    """(gold_changed, gold_letter, gold_string) for a dataset row.

    gold_changed mirrors the upstream dataloader (dataloaders/hint_attribution.py):
    use changed_pred_bc_hint when present, else derive it from whether the
    with-hint and no-hint predictions differ. The field is null for ~24% of rows
    in both splits; the derivation agrees with every non-null label.
    """
    derived = row["random_hint_prediction"] != row["zeroshot_prediction"]
    if row["changed_pred_bc_hint"] is None:
        changed = derived
    else:
        changed = bool(row["changed_pred_bc_hint"])
        assert changed == derived, (
            f"changed_pred_bc_hint={changed} contradicts the prediction pair "
            f"({row['random_hint_prediction']!r} vs {row['zeroshot_prediction']!r}); "
            "the dataset no longer matches the verified schema - investigate before trusting results"
        )
    letter = extract_letter(row["zeroshot_prediction"])  # the NO-hint answer
    if letter is None:
        print(f"WARNING: no A-D letter in zeroshot_prediction={row['zeroshot_prediction']!r}")
    template = GOLD_CHANGED_TEMPLATE if changed else GOLD_UNCHANGED_TEMPLATE
    return changed, letter, template.format(letter=letter)


def sample_demo_indices(changed_pool, unchanged_pool, k, seed):
    """Balanced, seed-fixed demo bank: k/2 changed + k/2 unchanged train rows.

    The bank is sampled once per (k, seed) and prepended to every test item.
    """
    assert k > 0 and k % 2 == 0, f"k must be a positive even number, got {k}"
    rng = random.Random(seed)
    picks = rng.sample(changed_pool, k // 2) + rng.sample(unchanged_pool, k // 2)
    rng.shuffle(picks)
    return picks


def build_messages(test_row, demos):
    """Chat messages: system + k solved demos + the test question.

    Each demo is a complete solved instance of the k=0 format: the demo user
    turn repeats the full yaml instruction block, and the demo assistant turn
    is the demo's true gold string. demos is a list of (train_row, gold_string).
    """
    messages = [{"role": "system", "content": test_row["system_prompt"]}]
    for demo_row, demo_gold in demos:
        messages.append({"role": "user", "content": demo_row["hint_user_prompt"] + "\n" + INSTRUCTION_BLOCK})
        messages.append({"role": "assistant", "content": demo_gold})
    messages.append({"role": "user", "content": test_row["hint_user_prompt"] + "\n" + INSTRUCTION_BLOCK})
    return messages


def render_prompt(tokenizer, messages):
    """The exact string fed to the model.

    enable_thinking=False always, identical across all k, so that thinking mode
    is held constant and k is the only variable between conditions.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def verify_against_yaml():
    """Assert the embedded template matches the repo yaml verbatim, if reachable."""
    yaml_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "config", "input_ablation", "zs_qwen_qwen_hint.yaml",
        )
    )
    if not os.path.exists(yaml_path):
        return f"yaml not found at {yaml_path}; verbatim check skipped"
    import yaml as pyyaml  # transitive dependency of transformers/datasets

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = pyyaml.safe_load(f)
    messages = cfg["train"]["tasks"]["hint_attribution"]["question_types"][
        "generative_explanation"]["prompts"][0]["messages"]
    assert messages[0]["role"] == "system" and messages[0]["content"] == "{system_prompt}", (
        "yaml system message differs from the expected '{system_prompt}' template"
    )
    expected_user = "{user_prompt}\n" + INSTRUCTION_BLOCK
    assert messages[1]["role"] == "user" and messages[1]["content"] == expected_user, (
        "embedded INSTRUCTION_BLOCK does not match the repo yaml verbatim"
    )
    return f"embedded template verified verbatim against {yaml_path}"


def load_splits():
    from datasets import load_dataset

    assert DEMO_SPLIT == "train" and EVAL_SPLIT == "test" and DEMO_SPLIT != EVAL_SPLIT
    train_ds = load_dataset(DATASET_ID, split=DEMO_SPLIT)
    test_ds = load_dataset(DATASET_ID, split=EVAL_SPLIT)
    for ds, name in [(train_ds, DEMO_SPLIT), (test_ds, EVAL_SPLIT)]:
        loaded_split = getattr(ds, "split", None)
        assert loaded_split is None or str(loaded_split) == name, (
            f"loaded split {loaded_split!r} != requested {name!r}"
        )
        missing = REQUIRED_FIELDS - set(ds.column_names)
        assert not missing, f"{name} split is missing expected fields: {missing}"
    return train_ds.to_list(), test_ds.to_list()


def build_pools(train_rows):
    """Train row indices by gold changed/unchanged (for balanced demo sampling)."""
    train_gold = [gold_fields(r) for r in train_rows]
    changed_pool = [i for i, g in enumerate(train_gold) if g[0]]
    unchanged_pool = [i for i, g in enumerate(train_gold) if not g[0]]
    return train_gold, changed_pool, unchanged_pool


def condition_list(ks, seeds):
    """k=0 runs once (seed-independent, stored as seed 0); each k>0 runs per seed."""
    conditions = []
    for k in ks:
        if k == 0:
            conditions.append((0, 0))
        else:
            conditions.extend((k, s) for s in seeds)
    return conditions


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip() != ""]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", default="0,4,8", help="comma-separated demo counts; 0 = zero-shot, k>0 must be even")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated demo-sampling seeds (ignored for k=0)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test rows (default: full split)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--inspect", action="store_true", help="dataset/prompt dry run: no GPU, no model weights")
    ap.add_argument("--overwrite", action="store_true", help="re-run conditions whose predictions file already exists")
    return ap.parse_args()


def sweep_labels(rows, name):
    """Count nulls and re-assert label consistency over an entire split."""
    n_null = sum(1 for r in rows if r["changed_pred_bc_hint"] is None)
    n_changed = sum(1 for r in rows if gold_fields(r)[0])  # asserts on every non-null row
    print(
        f"{name}: {len(rows)} rows | changed_pred_bc_hint null on {n_null} "
        f"| after derivation: changed={n_changed}, unchanged={len(rows) - n_changed}"
    )


def run_inspect(args):
    """Schema + prompt dry run (step 0 on the HPC login node)."""
    print(verify_against_yaml())
    train_rows, test_rows = load_splits()
    print(f"\nsplits: {DEMO_SPLIT}={len(train_rows)} rows, {EVAL_SPLIT}={len(test_rows)} rows")
    print(f"fields: {sorted(train_rows[0].keys())}\n")

    for name, rows in [(DEMO_SPLIT, train_rows), (EVAL_SPLIT, test_rows)]:
        for j in range(3):
            print(f"----- {name} row {j} -----")
            print(json.dumps(rows[j], indent=2, default=str))
        print()

    sweep_labels(train_rows, DEMO_SPLIT)
    sweep_labels(test_rows, EVAL_SPLIT)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_gold, changed_pool, unchanged_pool = build_pools(train_rows)

    test_row = test_rows[0]
    k0_prompt = render_prompt(tokenizer, build_messages(test_row, []))
    demo_idx = sample_demo_indices(changed_pool, unchanged_pool, 4, 0)
    demos = [(train_rows[i], train_gold[i][2]) for i in demo_idx]
    k4_prompt = render_prompt(tokenizer, build_messages(test_row, demos))

    print("\n===== RENDERED k=0 PROMPT (test row 0), exact model input =====")
    print(k0_prompt)
    print("===== END k=0 PROMPT =====")
    print(f"\n===== RENDERED k=4 PROMPT (seed 0, demo train indices {demo_idx}), exact model input =====")
    print(k4_prompt)
    print("===== END k=4 PROMPT =====")

    print("\ngold target strings for the first 3 test rows:")
    for j in range(3):
        gold_changed, gold_letter, gold_string = gold_fields(test_rows[j])
        print(f"  test row {j}: gold_changed={gold_changed}, gold_letter={gold_letter}, gold={gold_string!r}")

    n_tokens = [len(tokenizer(p).input_ids) for p in (k0_prompt, k4_prompt)]
    print(f"\nprompt lengths (tokens): k=0 -> {n_tokens[0]}, k=4 -> {n_tokens[1]}")
    print("inspect OK")


def run_generate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    yaml_status = verify_against_yaml()
    print(yaml_status)
    train_rows, test_rows_full = load_splits()
    test_rows = test_rows_full[: args.limit] if args.limit else test_rows_full
    train_gold, changed_pool, unchanged_pool = build_pools(train_rows)
    test_gold = [gold_fields(r) for r in test_rows]

    ks = parse_int_list(args.ks)
    seeds = parse_int_list(args.seeds)
    conditions = condition_list(ks, seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    def out_path(k, s):
        return os.path.join(args.out_dir, f"preds_k{k}_seed{s}.jsonl")

    def existing_covers_run(path):
        """Crash-resume guard: an existing predictions file only counts as done
        if it covers this invocation's item count. Without this, the 300-item
        pilot files would silently satisfy the full-split run and ship as
        full results. (If the never-expected truncation guard skipped items,
        the shorter file is re-run too - that costs time, never correctness.)
        """
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            n_records = sum(1 for line in f if line.strip())
        if n_records != len(test_rows):
            print(
                f"{path} exists but has {n_records} records, expected {len(test_rows)} "
                f"(stale run with a different --limit?); re-running this condition"
            )
            return False
        return True

    pending = [c for c in conditions if args.overwrite or not existing_covers_run(out_path(*c))]
    for k, s in conditions:
        if (k, s) not in pending:
            print(
                f"skipping k={k} seed={s}: {out_path(k, s)} already covers all "
                f"{len(test_rows)} items (use --overwrite to redo)"
            )
    if not pending:
        print("all requested conditions already have predictions; nothing to do")
        return

    meta_path = os.path.join(args.out_dir, "run_metadata.json")
    meta = {}
    if os.path.exists(meta_path) and not args.overwrite:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(
        {
            "model": args.model,
            "dataset": DATASET_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "dtype": "bfloat16",
                "padding_side": "left",
            },
            "enable_thinking": False,
            "split_sizes": {DEMO_SPLIT: len(train_rows), EVAL_SPLIT: len(test_rows_full)},
            "limit": args.limit,
            "git_commit": git_commit_hash(),
            "yaml_check": yaml_status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    meta.setdefault("conditions", {})

    def write_meta():
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)

    print(f"loading tokenizer and model {args.model} (bf16, single GPU) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # critical for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers v5 renamed from_pretrained's torch_dtype kwarg to dtype
    major = int(transformers.__version__.split(".")[0])
    dtype_kwargs = {"dtype": torch.bfloat16} if major >= 5 else {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwargs)
    model.to("cuda")
    model.eval()
    meta["gpu"] = torch.cuda.get_device_name(0)
    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    max_prompt_tokens = max_ctx - args.max_new_tokens

    for k, s in pending:
        if k == 0:
            demo_idx = []
        else:
            demo_idx = sample_demo_indices(changed_pool, unchanged_pool, k, s)
        demos = [(train_rows[i], train_gold[i][2]) for i in demo_idx]
        n_changed_demos = sum(train_gold[i][0] for i in demo_idx)
        assert len(demo_idx) == k and n_changed_demos == k // 2, "demo bank is not balanced"

        prompts, kept_idx, skipped_idx = [], [], []
        for i, row in enumerate(test_rows):
            prompt = render_prompt(tokenizer, build_messages(row, demos))
            n_tok = len(tokenizer(prompt).input_ids)
            if n_tok > max_prompt_tokens:
                print(f"WARNING: skipping test idx {i}: {n_tok} prompt tokens > {max_prompt_tokens}")
                skipped_idx.append(i)
                continue
            prompts.append(prompt)
            kept_idx.append(i)

        print(f"k={k} seed={s}: generating for {len(prompts)} test items "
              f"(demo train indices: {demo_idx})")
        records = []
        for b in range(0, len(prompts), args.batch_size):
            batch = prompts[b : b + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j, text in enumerate(texts):
                i = kept_idx[b + j]
                gold_changed, gold_letter, _ = test_gold[i]
                pred_changed, pred_letter = parse_output(text)
                records.append(
                    {
                        "idx": i,
                        "k": k,
                        "seed": s,
                        "raw_output": text,
                        "pred_changed": pred_changed,
                        "pred_letter": pred_letter,
                        "gold_changed": gold_changed,
                        "gold_letter": gold_letter,
                    }
                )
            done = min(b + args.batch_size, len(prompts))
            if done % (args.batch_size * 10) < args.batch_size or done == len(prompts):
                print(f"  k={k} seed={s}: {done}/{len(prompts)}", flush=True)

        path = out_path(k, s)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)

        unparseable = sum(
            1 for r in records if r["pred_changed"] is None or r["pred_letter"] is None
        )
        meta["conditions"][f"k{k}_seed{s}"] = {
            "demo_train_indices": demo_idx,
            "n_evaluated": len(records),
            "skipped_too_long": skipped_idx,
            "unparseable": unparseable,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_meta()
        print(f"k={k} seed={s}: wrote {len(records)} predictions to {path} "
              f"({unparseable} unparseable)")

    print("all conditions done")


def main():
    args = parse_args()
    for k in parse_int_list(args.ks):
        assert k == 0 or k % 2 == 0, f"k must be 0 or even for balanced demos, got {k}"
    if args.inspect:
        run_inspect(args)
    else:
        run_generate(args)


if __name__ == "__main__":
    main()
