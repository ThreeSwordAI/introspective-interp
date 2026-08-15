# Few-shot ICL on the input-ablation task (no weight updates)

## What & why

The paper *Training Language Models to Explain Their Own Computations* (Li et al., arXiv:2511.08579) fine-tunes Qwen3-8B to report how its own answer to an MMLU question would change if an injected hint were removed, reaching **83.4 Exact Match / 87.0 Has-Changed F1 / 90.6 Content Match** (Table 2), while *untrained* Qwen3-8B scores only **8.9 / 44.4 / 35.3**. The untrained baseline (Appendix H.3) is zero-shot with format examples only — the prompt shows the two allowed output shapes but contains no solved demonstrations. This experiment asks whether balanced few-shot in-context learning (k = 0, 4, 8 solved demonstrations, 3 demo seeds, no weight updates) closes any of that gap under the same prompt template and metrics. If it does, part of the capability is elicitable by task induction alone; if not, the paper's fine-tuning result stands even stronger.

## Method

- **Prompt structure.** Chat messages rendered with `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`. The system message is the row's `system_prompt`. Each of the k demonstrations (drawn from the **train** split only) is a complete solved instance of the zero-shot format: a user turn holding the demo's `hint_user_prompt` plus the full instruction block copied verbatim from `config/input_ablation/zs_qwen_qwen_hint.yaml` (repeated in every demo turn), then an assistant turn holding the demo's true gold string. The final user turn is the test row's `hint_user_prompt` plus the same instruction block. `run_icl.py --inspect` re-verifies the embedded block against the yaml at runtime.
- **`enable_thinking=False` always, identical across all k.** This is a design decision: thinking mode is held constant so that k is the only variable between conditions.
- **Demonstrations.** Per (k, seed), `random.Random(seed)` draws k/2 train rows whose answer changed because of the hint and k/2 whose answer did not; the bank is fixed for the whole condition and the chosen train indices are logged in `results/run_metadata.json`. k = 0 is seed-independent and runs once.
- **Gold construction.** `gold_changed` mirrors the upstream dataloader (`dataloaders/hint_attribution.py`): `changed_pred_bc_hint` when present, else derived as `random_hint_prediction != zeroshot_prediction` (the field is null for ~24% of rows in both splits; the derivation agrees with every non-null label). `gold_letter` is the letter of `zeroshot_prediction` (the no-hint answer). Gold strings follow the format the yaml prompt instructs the model to emit ("The output would change to <<<Answer: X>>>."); note the upstream training target for changed items instead reads "The most likely output would change to <<<...>>>." — the parser accepts both phrasings.
- **Decoding.** bf16, single GPU, greedy (`do_sample=False`), `max_new_tokens=40`, left padding, only newly generated tokens decoded.
- **Metrics** (all in percent, over all evaluated items, unparseable counted as incorrect; parse rate reported per condition): Exact Match = parsed changed/unchanged verdict **and** parsed letter both correct; Has-Changed macro F1 over parseable items only (an item is parseable iff both the verdict and the letter parse); Content Match = parsed letter correct. Because Exact Match is computed from the parsed output rather than by strict string equality against the upstream "most likely" target strings, the k = 0 row may land somewhat above the paper's 8.9 (an untrained model that follows the prompt's own format can never strict-string-match a changed-item target). Paired bootstrap (10,000 resamples over test items) gives 95% CIs on ΔEM of each (k > 0, seed) vs k = 0.

## How to run on HPC

```bash
# ---------- 0. Login (login node has internet; compute nodes run offline) ----------
ssh iwi5359h@alex.nhr.fau.de

# ---------- 1. One-time: conda env on vault ----------
source /etc/profile
module load python/3.12-conda
conda create -p /home/vault/iwi5/iwi5359h/envs/iclhint python=3.12 -y
source activate /home/vault/iwi5/iwi5359h/envs/iclhint
pip install torch --index-url https://download.pytorch.org/whl/cu128

# ---------- 2. One-time: clone the fork (code lives on /home/hpc) ----------
mkdir -p /home/hpc/iwi5/iwi5359h/my_repos && cd /home/hpc/iwi5/iwi5359h/my_repos
git clone https://github.com/ThreeSwordAI/introspective-interp.git
cd introspective-interp/experiments/icl_input_ablation
pip install -r requirements.txt
mkdir -p /home/vault/iwi5/iwi5359h/icl_input_ablation/logs

# ---------- 3. One-time: pre-download model + dataset on the LOGIN node ----------
export HF_HOME=/home/vault/iwi5/iwi5359h/hf_cache
python3 - <<'EOF'
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("Qwen/Qwen3-8B")
load_dataset("Transluce/input_ablation_qwen3_8b_mmlu_hint")
print("download ok")
EOF

# ---------- 4. Schema check (CPU, login node, seconds) ----------
python3 run_icl.py --inspect        # read the printed prompts/fields; must look right

# ---------- 5. Pilot on GPU (edit sbatch: uncomment PILOT line, comment FULL) ----------
sbatch job_icl.sbatch
squeue -u iwi5359h                  # watch; then check the log:
tail -f /home/vault/iwi5/iwi5359h/icl_input_ablation/logs/job_<JOBID>.log
python3 score.py                    # pilot gate: k=0 should land near Table 2 (8.9 / 44.4 / 35.3)

# ---------- 6. Full run (restore FULL line in sbatch) ----------
sbatch job_icl.sbatch

# ---------- 7. Ship results back ----------
cd /home/hpc/iwi5/iwi5359h/my_repos/introspective-interp
git add experiments/icl_input_ablation/results
git commit -m "results: ICL input-ablation, full test split, k=0/4/8 x 3 seeds"
git push
```

Interactive debugging alternative to step 5: `salloc --gres=gpu:a40:1 --partition=a40 --time=01:00:00`, then the same `source /etc/profile` / `module load` / `source activate` lines and run the pilot command directly.

**Pilot gate:** if the k=0 pilot is wildly off Table 2 (e.g., EM > 25 or F1 < 30), something is wrong (template, thinking mode, parsing) — report the numbers and debug before the full run. A few points of deviation is fine (300-item subset, possible decoding differences, and the parse-based Exact Match noted under Method).

Moving from pilot to full run needs no cleanup: a predictions file only satisfies crash-resume if it covers the current run's item count, so the full run automatically re-runs the 300-item pilot conditions rather than silently reusing them; `score.py` additionally refuses to score conditions with unequal item counts together.

## Results

<!-- FILLED IN PHASE 2 -->

## Interpretation

<!-- FILLED IN PHASE 2 -->

## Reproducibility

<!-- FILLED IN PHASE 2 -->

## Limitations

- Single model (Qwen3-8B); no evidence the effect size transfers to other scales or families.
- Demonstrations are randomly sampled (balanced by changed/unchanged), not retrieved by similarity.
- Nominal-format ICL only: demos reuse the zero-shot format with its instruction block; other demo formats (e.g., without repeated instructions, or with rationales) are untested.
