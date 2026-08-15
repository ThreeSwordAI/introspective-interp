"""Scoring for the ICL input-ablation experiment.

Reads results/preds_k{k}_seed{s}.jsonl, computes Exact Match, Has-Changed
macro F1 and Content Match per condition (all in percent, like the paper's
Table 2), paired-bootstrap deltas of each k>0 condition vs k=0, and writes
results/metrics.csv and results/plot.png.

Unparseable outputs count as incorrect for Exact Match and Content Match;
Has-Changed macro F1 is computed over parseable items only, with the parse
rate reported per condition.

`--selftest` unit-tests the shared output parser and exits (no results needed).
"""

import argparse
import glob
import json
import os
import re
import sys

from run_icl import parse_output  # single source of truth for parsing

PAPER_UNTRAINED = {"em": 8.9, "f1_macro": 44.4, "content": 35.3}
PAPER_FINETUNED = {"em": 83.4, "f1_macro": 87.0, "content": 90.6}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_BLOCK = 1_000  # resamples per vectorized block, to bound memory

# Categorical palette (colorblind-validated), fixed order: EM, F1, Content.
SERIES = [
    ("em", "Exact Match", "#2a78d6"),
    ("f1_macro", "Has-Changed F1 (macro)", "#eb6834"),
    ("content", "Content Match", "#1baf7a"),
]

CSV_COLUMNS = ["k", "seed", "n", "parse_rate", "em", "f1_macro", "content",
               "d_em_vs_k0", "ci_low", "ci_high"]


def selftest():
    """Parser unit-asserts: canonical strings, variants, garbage."""
    cases = [
        # the two canonical strings from the prompt's instruction block
        ("The output would remain unchanged from <<<Answer: C>>>.", (False, "C")),
        ("The output would change to <<<Answer: B>>>.", (True, "B")),
        # the upstream fine-tuning target phrasing must parse identically
        ("The most likely output would change to <<<Answer: B>>>.", (True, "B")),
        # whitespace / case variants
        ("  the output would REMAIN   UNCHANGED from <<<answer:  c >>>.  ", (False, "C")),
        ("The output would change to <<<Answer:B>>>", (True, "B")),
        ("The output would\nchange  to <<< Answer: D >>>.", (True, "D")),
        # partial or garbage output: nulls, never a crash
        ("", (None, None)),
        ("Answer: B", (None, None)),
        ("The output would change to Answer: B.", (True, None)),
        ("total garbage >>> <<< >>>", (None, None)),
        ("<<<Answer: E>>>", (None, None)),
        (None, (None, None)),
        # leftmost phrase and first tag win if the model repeats itself
        ("The output would change to <<<Answer: A>>>. The output would change to <<<Answer: B>>>.", (True, "A")),
        ("The output would remain unchanged from <<<Answer: D>>>. It would not change to anything.", (False, "D")),
    ]
    for raw, want in cases:
        got = parse_output(raw)
        assert got == want, f"parse_output({raw!r}) = {got}, want {want}"
    print(f"selftest OK: {len(cases)} parser cases passed")


def load_conditions(results_dir):
    """{(k, seed): [record, ...]} from every preds_k*_seed*.jsonl present."""
    conditions = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "preds_k*_seed*.jsonl"))):
        m = re.fullmatch(r"preds_k(\d+)_seed(\d+)\.jsonl", os.path.basename(path))
        if not m:
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        conditions[(int(m.group(1)), int(m.group(2)))] = rows
    return conditions


def item_scores(rows):
    """Per-item correctness. An item is parseable iff both the changed/unchanged
    verdict and the answer letter parse; None never counts as a match."""
    scored = []
    for r in rows:
        pred_changed, pred_letter = parse_output(r["raw_output"])
        parseable = pred_changed is not None and pred_letter is not None
        em = (
            parseable
            and pred_changed == r["gold_changed"]
            and pred_letter == r["gold_letter"]
        )
        content = pred_letter is not None and pred_letter == r["gold_letter"]
        scored.append(
            {
                "idx": r["idx"],
                "parseable": parseable,
                "em": bool(em),
                "content": bool(content),
                "pred_changed": pred_changed,
                "gold_changed": bool(r["gold_changed"]),
            }
        )
    return scored


def condition_metrics(scored):
    from sklearn.metrics import f1_score

    n = len(scored)
    parse_rate = sum(s["parseable"] for s in scored) / n
    em = 100.0 * sum(s["em"] for s in scored) / n
    content = 100.0 * sum(s["content"] for s in scored) / n
    parseable = [s for s in scored if s["parseable"]]
    if parseable:
        f1 = 100.0 * f1_score(
            [s["gold_changed"] for s in parseable],
            [s["pred_changed"] for s in parseable],
            average="macro",
            zero_division=0,
        )
    else:
        f1 = 0.0
    return {"n": n, "parse_rate": round(parse_rate, 4), "em": round(em, 2),
            "f1_macro": round(f1, 2), "content": round(content, 2)}


def paired_bootstrap_delta(scored_k, scored_0, rng):
    """95% CI for (EM_k - EM_0) over test-item indices, paired on idx."""
    import numpy as np

    em_k = {s["idx"]: s["em"] for s in scored_k}
    em_0 = {s["idx"]: s["em"] for s in scored_0}
    common = sorted(set(em_k) & set(em_0))
    if len(common) != len(em_k) or len(common) != len(em_0):
        print(
            f"WARNING: condition and k=0 cover different items "
            f"({len(em_k)} vs {len(em_0)}); bootstrapping over the {len(common)} common items"
        )
    a = np.array([em_k[i] for i in common], dtype=float)
    b = np.array([em_0[i] for i in common], dtype=float)
    point = 100.0 * float(a.mean() - b.mean())
    deltas = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining > 0:
        block = min(BOOTSTRAP_BLOCK, remaining)
        pick = rng.integers(0, len(common), size=(block, len(common)))
        deltas.append(100.0 * (a[pick].mean(axis=1) - b[pick].mean(axis=1)))
        remaining -= block
    deltas = np.concatenate(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return round(point, 2), round(float(lo), 2), round(float(hi), 2)


def make_plot(df, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ks = sorted(df["k"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for col, label, color in SERIES:
        means, lows, highs = [], [], []
        for k in ks:
            vals = df.loc[df["k"] == k, col]
            means.append(vals.mean())
            lows.append(vals.mean() - vals.min())
            highs.append(vals.max() - vals.mean())
        ax.errorbar(
            ks, means, yerr=[lows, highs], color=color, marker="o",
            markersize=5, linewidth=2, capsize=3, label=label, zorder=3,
        )
        ax.axhline(PAPER_UNTRAINED[col], color=color, linestyle="--", linewidth=1, alpha=0.55, zorder=1)
        ax.axhline(PAPER_FINETUNED[col], color=color, linestyle=":", linewidth=1, alpha=0.55, zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="#898781", linestyle="--", linewidth=1),
        Line2D([0], [0], color="#898781", linestyle=":", linewidth=1),
    ]
    labels += ["paper: untrained (8.9 / 44.4 / 35.3)", "paper: fine-tuned (83.4 / 87.0 / 90.6)"]
    ax.legend(handles, labels, loc="center right", frameon=False, fontsize=8)

    ax.set_xlabel("k (in-context demonstrations)")
    ax.set_ylabel("score (%)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    ax.set_title("Few-shot ICL on input ablation - Qwen3-8B, no weight updates", fontsize=11)
    ax.text(
        0.0, 1.02,
        "points = mean over demo seeds, error bars = min-max; reference lines colored per metric",
        transform=ax.transAxes, fontsize=7.5, color="#52514e",
    )
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--selftest", action="store_true", help="unit-test the output parser and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import numpy as np
    import pandas as pd

    conditions = load_conditions(args.results_dir)
    if not conditions:
        print(f"no preds_k*_seed*.jsonl files in {args.results_dir!r}; run run_icl.py first")
        sys.exit(1)

    # Refuse to mix conditions evaluated on different item counts (e.g. stale
    # 300-item pilot files next to full-split files): averaged metrics and the
    # k=0 baseline would silently describe different samples.
    n_by_condition = {cond: len(rows) for cond, rows in conditions.items()}
    if len(set(n_by_condition.values())) > 1:
        print("ERROR: conditions cover different item counts; refusing to score them together:")
        for (k, s), n in sorted(n_by_condition.items()):
            print(f"  k={k} seed={s}: {n} items")
        print("delete the stale preds_*.jsonl (or re-run run_icl.py, which re-runs "
              "conditions whose files do not cover the current item count), then re-score")
        sys.exit(1)

    scored = {cond: item_scores(rows) for cond, rows in conditions.items()}
    baseline = scored.get((0, 0))
    if baseline is None:
        print("WARNING: no k=0 predictions found; d_em_vs_k0 and CIs will be empty")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out_rows = []
    for k, s in sorted(scored):
        metrics = condition_metrics(scored[(k, s)])
        if k > 0 and baseline is not None:
            d_em, ci_low, ci_high = paired_bootstrap_delta(scored[(k, s)], baseline, rng)
        else:
            d_em = ci_low = ci_high = float("nan")
        out_rows.append({"k": k, "seed": s, **metrics,
                         "d_em_vs_k0": d_em, "ci_low": ci_low, "ci_high": ci_high})

    df = pd.DataFrame(out_rows, columns=CSV_COLUMNS)
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"wrote {csv_path}")

    make_plot(df, os.path.join(args.results_dir, "plot.png"))


if __name__ == "__main__":
    main()
