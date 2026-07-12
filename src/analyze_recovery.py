"""Compute no-CoT recovery vs prompt-repeat count, test significance, and plot.

Usage: python src/analyze_recovery.py [LOG_DIR]   (default: logs/gemma-repeats)

Per model: baseline = no-CoT @ 1 repeat, ceiling = CoT. Recovery fraction =
(acc_r - baseline) / (ceiling - baseline). Significance vs baseline via McNemar
(paired on sample id). Error bars = paired SE of the accuracy delta, rescaled by
the CoT gap. Writes recovery.png into LOG_DIR.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.analysis import evals_df, samples_df
from statsmodels.stats.contingency_tables import mcnemar

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/gemma-repeats"

# Categorical slots 1-3 (validated dataviz palette, light mode).
COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def short(model: str) -> str:
    return model.split("/")[-1].removeprefix("gemma-3-")


def main() -> None:
    meta = evals_df(LOG_DIR)[
        ["eval_id", "model", "task_arg_prompt_repeats", "task_arg_use_cot"]
    ]
    s = samples_df(LOG_DIR)[["eval_id", "id", "score_match"]].copy()
    s["correct"] = s["score_match"] == "C"
    df = s.merge(meta, on="eval_id")

    series = {}  # model -> (repeats, recovery, se)
    for model in sorted(df["model"].unique()):
        m = df[df["model"] == model]
        ceiling = m[m["task_arg_use_cot"]]["correct"].mean()
        nocot = m[~m["task_arg_use_cot"]]
        base = nocot[nocot["task_arg_prompt_repeats"] == 1]
        base_acc = base["correct"].mean()
        base_by_id = base.set_index("id")["correct"]
        gap = ceiling - base_acc

        print(f"\n{model}  (CoT={ceiling:.3f}, noCoT@1={base_acc:.3f})")
        print(f"{'repeats':>7} {'acc':>6} {'recovery':>9} {'p(McNemar)':>11}")
        rs, recovs, ses = [], [], []
        for r in sorted(nocot["task_arg_prompt_repeats"].unique()):
            arm = nocot[nocot["task_arg_prompt_repeats"] == r]
            acc = arm["correct"].mean()
            recov = (acc - base_acc) / gap if gap else float("nan")
            if r == 1:
                rs, recovs, ses = [r], [0.0], [0.0]
                print(f"{r:>7} {acc:>6.3f} {'baseline':>9} {'-':>11}")
                continue
            # paired 2x2 vs baseline on shared sample ids
            cur = arm.set_index("id")["correct"]
            ids = base_by_id.index.intersection(cur.index)
            b, c = base_by_id.loc[ids], cur.loc[ids]
            n01 = int((~b & c).sum())  # baseline wrong, repeat right
            n10 = int((b & ~c).sum())  # baseline right, repeat wrong
            n = len(ids)
            p = mcnemar([[0, n01], [n10, 0]], exact=True).pvalue
            # paired SE of the accuracy delta, rescaled to recovery units
            se_delta = np.sqrt(n01 + n10 - (n01 - n10) ** 2 / n) / n
            se = se_delta / abs(gap) if gap else float("nan")
            rs.append(r); recovs.append(recov); ses.append(se)
            print(f"{r:>7} {acc:>6.3f} {recov:>9.2f} {p:>11.4f}")
        series[model] = (rs, recovs, ses)

    plot(series)


def plot(series: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_r = sorted({r for rs, _, _ in series.values() for r in rs})
    ax.set_xscale("log", base=2)
    ax.set_xlim(all_r[0] * 0.9, all_r[-1] * 2.0)  # right pad for direct labels
    ax.set_ylim(-0.06, 1.06)
    xleft = ax.get_xlim()[0]

    # reference lines: 0 = no-CoT baseline, 1 = CoT ceiling (labels at left)
    for y, lab, va in [(0.0, "no-CoT baseline", "top"), (1.0, "CoT ceiling", "bottom")]:
        ax.axhline(y, color=MUTED, lw=1, ls="--", zorder=0)
        ax.text(xleft, y, lab, color=MUTED, va=va, ha="left", fontsize=8)

    for i, (model, (rs, recovs, ses)) in enumerate(series.items()):
        color = COLORS[i % len(COLORS)]
        ax.errorbar(rs, recovs, yerr=ses, color=color, lw=2, marker="o",
                    ms=7, capsize=3, zorder=3, label=short(model))
        ax.annotate(short(model), (rs[-1], recovs[-1]), color=color,
                    fontsize=9, fontweight="bold", xytext=(8, 0),
                    textcoords="offset points", va="center")

    ax.set_xticks(all_r)
    ax.set_xticklabels(all_r)
    ax.set_xlabel("Prompt repeats", color=INK)
    ax.set_ylabel("Recovery fraction  (0 = no-CoT, 1 = CoT)", color=INK)
    ax.set_title("No-CoT recovery vs prompt repeats", color=INK, loc="left")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="upper right")

    out = f"{LOG_DIR}/recovery.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
