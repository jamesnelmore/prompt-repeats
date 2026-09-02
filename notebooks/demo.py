# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "inspect-ai==0.3.261",
#     "marimo>=0.24.0",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
#     "statsmodels==0.15.0",
# ]
# ///
import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import hashlib
    import math
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    from inspect_ai.log import read_eval_log
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint

    def pretty_model(name: str) -> str:
        """gemma-3-12b-it / google/gemma-3-12b-it → 'Gemma 3 12b'."""
        slug = name.split("/")[-1]
        if slug.endswith("-it"):
            slug = slug[: -len("-it")]
        parts = []
        for i, p in enumerate(slug.split("-")):
            if i == 0:
                parts.append(p.capitalize())
            elif p[:-1].replace(".", "").isdigit() and p.endswith("b"):
                parts.append(p)
            elif p.replace(".", "").isdigit():
                parts.append(p)
            else:
                parts.append(p.upper())
        return " ".join(parts)

    return (
        Path,
        alt,
        hashlib,
        math,
        mcnemar,
        mo,
        np,
        pd,
        pretty_model,
        proportion_confint,
        read_eval_log,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt duplication improves nonreasoning performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation
    [Greenblatt found](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens) that some LLMs can use filler tokens and prompt repeats to improve nonreasoning performance.
    [Later work](https://arxiv.org/abs/2606.07157) by Redwood found that one question duplication modestly improves performance but filler tokens and multiple repeats do not.

    Understanding this phenomenon is important to measuring models' opaque reasoning capabilities.

    ## Contributions
    1. Replicate phenomenon with frontier open weight models
    2. Identify mechanism causing performance uplift
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open Weight Replication
    - Used OpenRouter to test open weight models on GSM8k with 1/2/4/8 copies of the question
    - Tested Gemma 3 family
    - Compared to performance with reasoning and measured performance recovery

    ```text
    User:
    	[Instructions]

    	[Question copy 1]
    	...
    	[Question copy n]

    Assistant:
    	ANSWER:
    ```
    """)
    return


@app.cell(hide_code=True)
def _(
    Path,
    alt,
    hashlib,
    np,
    pd,
    pretty_model,
    proportion_confint,
    read_eval_log,
):
    ROOT = Path(__file__).parent.parent
    LOG_DIR = ROOT / "logs"
    CACHE_DIR = ROOT / "data" / "cache"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    MUTED = "#898781"
    N_BOOT = 2000

    def load_samples() -> pd.DataFrame:
        paths = sorted(LOG_DIR.glob("*.eval"))
        key = hashlib.sha1(
            repr([(p.name, p.stat().st_size) for p in paths]).encode()
        ).hexdigest()[:12]
        cache = CACHE_DIR / f"samples-{key}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            if "repeats" in df.columns and "copies" not in df.columns:
                df = df.rename(columns={"repeats": "copies"})
            return df

        rows = []
        for path in paths:
            try:
                log = read_eval_log(str(path))
            except Exception:
                continue
            if log.status != "success":
                continue
            args = log.eval.task_args or {}
            n_copies = int(args.get("prompt_copies", args.get("prompt_repeats", 1)))
            for s in log.samples:
                score = next(iter(s.scores.values()))
                rows.append(
                    {
                        "model": log.eval.model.split("/")[-1],
                        "use_cot": bool(args.get("use_cot", False)),
                        "copies": n_copies,
                        "sample_id": s.id,
                        "correct": score.value == "C",
                    }
                )
        df = pd.DataFrame(rows)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    def summary_frame(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for model, sub in df.groupby("model"):
            wide = (
                sub.pivot_table(
                    index="sample_id", columns=["use_cot", "copies"], values="correct"
                )
                .dropna()
                .astype(float)
            )
            if (True, 1) not in wide.columns or (False, 1) not in wide.columns:
                continue
            cot = wide[(True, 1)].to_numpy()
            base = wide[(False, 1)].to_numpy()
            n = len(wide)
            gap = cot.mean() - base.mean()
            idx = rng.integers(0, n, size=(n_boot, n))
            cot_b, base_b = cot[idx].mean(axis=1), base[idx].mean(axis=1)
            gap_b = cot_b - base_b

            for c in sorted(c for use_cot, c in wide.columns if not use_cot):
                arm = wide[(False, c)].to_numpy()
                hits = int(arm.sum())
                lo, hi = proportion_confint(hits, n, method="wilson")
                arm_b = arm[idx].mean(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot = np.where(gap_b > 0, (arm_b - base_b) / gap_b, np.nan)
                r_lo, r_hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append(
                    {
                        "model": pretty_model(model),
                        "copies": c,
                        "n": n,
                        "accuracy": hits / n,
                        "acc_lo": lo,
                        "acc_hi": hi,
                        "cot_acc": cot.mean(),
                        "recovery": (arm.mean() - base.mean()) / gap if gap > 0 else np.nan,
                        "rec_lo": r_lo,
                        "rec_hi": r_hi,
                    }
                )
        return pd.DataFrame(rows)

    samples = load_samples()
    gemma = samples[samples.model.str.contains(r"^gemma-[34]-", regex=True)]
    summary = summary_frame(gemma)
    models = sorted(summary.model.unique(), key=lambda m: int("".join(c for c in m if c.isdigit()) or "0"))
    visible = summary[summary.model.isin(models)]

    color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=models, range=SERIES[: len(models)]),
        legend=alt.Legend(title="Model", orient="bottom", columns=2),
    )
    x = alt.X(
        "copies:Q",
        scale=alt.Scale(type="log", base=2, nice=False),
        axis=alt.Axis(values=sorted(visible.copies.unique()), title="Question copies"),
    )
    return MUTED, SERIES, color, gemma, visible, x


@app.cell(hide_code=True)
def _(alt, color, visible, x):
    _tip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("copies:Q", title="Copies"),
        alt.Tooltip("accuracy:Q", format=".1%", title="Accuracy"),
        alt.Tooltip("acc_lo:Q", format=".1%", title="CI low"),
        alt.Tooltip("acc_hi:Q", format=".1%", title="CI high"),
        alt.Tooltip("cot_acc:Q", format=".1%", title="CoT ceiling"),
    ]
    _base = alt.Chart(visible)
    acc_chart = (
        _base.mark_rule(strokeDash=[4, 4], strokeWidth=1, opacity=0.7).encode(
            y="cot_acc:Q", color=color
        )
        + _base.mark_rule(strokeWidth=1.5, opacity=0.65).encode(
            x=x, y="acc_lo:Q", y2="acc_hi:Q", color=color, tooltip=_tip
        )
        + _base.mark_line(point=True).encode(
            x=x,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=color,
            tooltip=_tip,
        )
    ).properties(
        width=520, height=280, title="No-CoT accuracy vs copies (dashed = CoT ceiling)"
    )
    acc_chart
    return


@app.cell(hide_code=True)
def _(MUTED, alt, color, pd, visible, x):
    _tip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("copies:Q", title="Copies"),
        alt.Tooltip("recovery:Q", format=".1%", title="Recovery"),
        alt.Tooltip("rec_lo:Q", format=".1%", title="CI low"),
        alt.Tooltip("rec_hi:Q", format=".1%", title="CI high"),
        alt.Tooltip("accuracy:Q", format=".1%", title="Accuracy"),
    ]
    _y_scale = alt.Scale(
        domain=[
            min(-0.12, float(visible.rec_lo.min()) - 0.07),
            max(1.05, float(visible.rec_hi.max()) + 0.05),
        ]
    )
    _anchors = pd.DataFrame({"y": [0.0, 1.0], "label": ["no-CoT, 1 copy", "CoT ceiling"]})
    _base = alt.Chart(visible)
    rec_chart = (
        alt.Chart(_anchors)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=MUTED)
        .encode(y=alt.Y("y:Q", scale=_y_scale))
        + alt.Chart(_anchors)
        .mark_text(align="left", dx=4, dy=-6, fontSize=11, color=MUTED)
        .encode(x=alt.value(4), y=alt.Y("y:Q", scale=_y_scale), text="label:N")
        + _base.mark_rule(strokeWidth=1.5, opacity=0.65).encode(
            x=x,
            y=alt.Y("rec_lo:Q", scale=_y_scale, title=""),
            y2="rec_hi:Q",
            color=color,
            tooltip=_tip,
        )
        + _base.mark_line(point=True).encode(
            x=x,
            y=alt.Y(
                "recovery:Q",
                scale=_y_scale,
                axis=alt.Axis(format="%", title="Recovery of CoT gap", grid=True),
            ),
            color=color,
            tooltip=_tip,
        )
    ).properties(
        width=520,
        height=280,
        title="Share of each model's CoT gap recovered by repeating the question",
    )
    rec_chart
    return


@app.cell(hide_code=True)
def _(gemma, mcnemar, mo, pd, pretty_model):
    def _mcnemar_p(base: pd.Series, arm: pd.Series) -> float:
        fixed = int((~base & arm).sum())
        broken = int((base & ~arm).sum())
        table = [
            [int((base & arm).sum()), broken],
            [fixed, int((~base & ~arm).sum())],
        ]
        return float(
            mcnemar(table, exact=(fixed + broken) < 25, correction=True).pvalue
        )

    _rows = []
    _vs1_cols: list[int] = []
    _vs2_rows = []
    for _model, _sub in gemma[~gemma.use_cot].groupby("model"):
        _wide = (
            _sub.pivot_table(index="sample_id", columns="copies", values="correct")
            .dropna()
            .astype(bool)
        )
        _wide.columns = _wide.columns.astype(int)
        if 1 not in _wide.columns:
            continue
        _name = pretty_model(_model)
        _row: dict[str, object] = {"model": _name}
        for _c in sorted(int(c) for c in _wide.columns if int(c) != 1):
            _row[f"{_c} copies"] = _mcnemar_p(_wide[1], _wide[_c])
            if _c not in _vs1_cols:
                _vs1_cols.append(_c)
        _rows.append(_row)
        if 2 in _wide.columns:
            _vs2: dict[str, object] = {"model": _name}
            for _c in (4, 8):
                if _c in _wide.columns:
                    _vs2[f"{_c} copies"] = _mcnemar_p(_wide[2], _wide[_c])
            if len(_vs2) > 1:
                _vs2_rows.append(_vs2)

    _vs1_cols = sorted(_vs1_cols)
    _headers = [f"{c} copies" for c in _vs1_cols]
    tests = (
        pd.DataFrame(_rows)
        .sort_values(
            "model",
            key=lambda s: s.map(lambda m: int("".join(ch for ch in m if ch.isdigit()) or "0")),
        )
        .reset_index(drop=True)[["model", *_headers]]
    )
    _vs2_headers = [c for c in ("4 copies", "8 copies") if c in pd.DataFrame(_vs2_rows).columns]
    tests_vs2 = (
        pd.DataFrame(_vs2_rows)
        .sort_values(
            "model",
            key=lambda s: s.map(lambda m: int("".join(ch for ch in m if ch.isdigit()) or "0")),
        )
        .reset_index(drop=True)[["model", *_vs2_headers]]
        if _vs2_rows
        else pd.DataFrame()
    )

    def _fmt_p(v: float) -> str:
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    mo.vstack(
        [
            mo.md(
                r"McNemar \(p\)-values vs 1 copy (same GSM8K items, no CoT). "
                r"Small \(p\) means that copy count differs from one copy."
            ),
            mo.ui.table(
                tests,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={col: _fmt_p for col in _headers},
                label="McNemar vs 1 copy",
            ),
            mo.md(
                r"Same test vs 2 copies. Large \(p\) means 4 and 8 copies are not "
                r"distinguishable from two."
            ),
            mo.ui.table(
                tests_vs2,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={col: _fmt_p for col in _vs2_headers},
                label="McNemar vs 2 copies",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Causal Interventions
    Activations from each copy flow through 3 pathways.
    We ablate each pathway in order to narrow down the causal mechanism.

    Using $p< .01$ as the cutoff for significence
    """)
    return


@app.cell
def _(mo):
    mo.mermaid('''
    flowchart LR
      copy1["Copy 1"]
      copy2["Copy 2"]
      output["Output"]
      copy1 -->|"(A)"| copy2
      copy2 -->|"(B)"| output
      copy1 -->|"(C)"| output
      ''')
    return


@app.cell(hide_code=True)
def _(Path, pd, pretty_model):
    ABLATION_DIR = Path(__file__).parent.parent / "ablation_results"
    BASELINE, SINGLE = "2copies", "1copy"
    PATHWAY_DISPLAY = {
        "1copy": "1 copy",
        "2copies": "2 copies",
        "block_copy2_from_copy1": "block (A)",
        "block_answer_from_copy1": "block (C)",
        "copy2_past_or_aligned_copy1": "block looking forward",
        "copy2_future_or_aligned_copy1": "block looking back",
    }

    def by_size(tag: str) -> float:
        digits = "".join(c for c in tag if c.isdigit())
        return float(digits) if digits else float("inf")

    _frames = [
        pd.read_json(path, lines=True)
        for path in sorted(ABLATION_DIR.glob("attention_*/samples.jsonl"))
        if not path.parent.name.endswith("_smoke")
    ]
    if not _frames:
        raise FileNotFoundError(f"no full-run samples.jsonl under {ABLATION_DIR}")
    _ablation = pd.concat(_frames, ignore_index=True)
    _short = {m: pretty_model(m) for m in _ablation.model.unique()}
    panels: dict[str, pd.DataFrame] = {
        _short[model]: (
            group.droplevel("model").dropna(axis=1, how="all").dropna().astype(bool)
        )
        for model, group in _ablation.pivot(
            index=["model", "gsm8k_test_index"], columns="arm", values="correct"
        ).groupby(level="model")
    }
    return BASELINE, PATHWAY_DISPLAY, SINGLE, by_size, panels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Block (A): copy 2 cannot read copy 1
    """)
    return


@app.cell
def _(mo):
    mo.mermaid('''
    flowchart LR
      copy1["Copy 1"]
      copy2["Copy 2"]
      output["Output"]
      copy1 x--x |"(A) Blocked"| copy2
      copy2 -->|"(B)"| output
      copy1 -->|"(C)"| output
      ''')
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    PATHWAY_DISPLAY,
    SERIES,
    SINGLE,
    alt,
    by_size,
    panels: "dict[str, pd.DataFrame]",
    pd,
    proportion_confint,
):
    _arms = [SINGLE, BASELINE, "block_copy2_from_copy1"]
    _rows = []
    for _model, _sub in panels.items():
        for _arm in _arms:
            if _arm not in _sub.columns:
                continue
            _hits, _n = int(_sub[_arm].sum()), len(_sub)
            _lo, _hi = proportion_confint(_hits, _n, method="wilson")
            _rows.append(
                {
                    "model": _model,
                    "arm": PATHWAY_DISPLAY[_arm],
                    "n": _n,
                    "accuracy": _hits / _n,
                    "ci_low": _lo,
                    "ci_high": _hi,
                }
            )
    block_a_acc = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    _models = sorted(block_a_acc.model.unique(), key=by_size)
    _order = [PATHWAY_DISPLAY[a] for a in _arms]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_models, range=SERIES[: len(_models)]),
        legend=alt.Legend(title="Model"),
    )
    _x = alt.X("arm:N", sort=_order, title=None)
    _tip = [
        alt.Tooltip("model:N"),
        alt.Tooltip("arm:N"),
        alt.Tooltip("accuracy:Q", format=".1%"),
        alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
        alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
    ]
    block_a_chart = (
        alt.Chart(block_a_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(block_a_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=_x,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=_tip,
        )
    ).properties(width=480, height=260, title="Accuracy after blocking (A) (95% Wilson CIs)")
    block_a_chart
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    SINGLE,
    by_size,
    math,
    mcnemar,
    mo,
    panels: "dict[str, pd.DataFrame]",
    pd,
):
    def mcnemar_pair(base: pd.Series, arm: pd.Series) -> dict[str, float]:
        n = len(base)
        broken = int((base & ~arm).sum())
        fixed = int((~base & arm).sum())
        table = [
            [int((base & arm).sum()), broken],
            [fixed, int((~base & ~arm).sum())],
        ]
        res = mcnemar(table, exact=(broken + fixed) < 25, correction=True)
        delta = (fixed - broken) / n
        var = max((broken + fixed) - (fixed - broken) ** 2 / n, 0.0)
        se = math.sqrt(var) / n
        return {
            "delta": delta,
            "delta_low": delta - 1.96 * se,
            "delta_high": delta + 1.96 * se,
            "p": float(res.pvalue),
        }

    def fmt_p(v: float) -> str:
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    def fmt_pp(v: float) -> str:
        return f"{v * 100:+.2f}"

    _rows = []
    for _model, _sub in panels.items():
        if "block_copy2_from_copy1" not in _sub.columns:
            continue
        _vs2 = mcnemar_pair(_sub[BASELINE], _sub["block_copy2_from_copy1"])
        _vs1 = mcnemar_pair(_sub[SINGLE], _sub["block_copy2_from_copy1"])
        _rows.append(
            {
                "model": _model,
                "Δ vs 2 copies (pp)": _vs2["delta"],
                "p vs 2 copies": _vs2["p"],
                "Δ vs 1 copy (pp)": _vs1["delta"],
                "p vs 1 copy": _vs1["p"],
            }
        )
    block_a_tests = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    mo.ui.table(
        block_a_tests,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "Δ vs 2 copies (pp)": fmt_pp,
            "Δ vs 1 copy (pp)": fmt_pp,
            "p vs 2 copies": fmt_p,
            "p vs 1 copy": fmt_p,
        },
        label="McNemar: block (A) vs 2 copies and vs 1 copy",
    )
    return fmt_p, fmt_pp, mcnemar_pair


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway:** Pathway A is necessary for performance improvement. With A blocked performance is not significently different from 1 copy at any model size
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Block (C): output cannot read copy 1
    """)
    return


@app.cell
def _(mo):
    mo.mermaid('''
    flowchart LR
      copy1["Copy 1"]
      copy2["Copy 2"]
      output["Output"]
      copy1 -->|"(A)"| copy2
      copy2 -->|"(B)"| output
      copy1 x--x|"(C) Attention Masked"| output
      ''')
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    PATHWAY_DISPLAY,
    SERIES,
    SINGLE,
    alt,
    by_size,
    panels: "dict[str, pd.DataFrame]",
    pd,
    proportion_confint,
):
    _arms = [SINGLE, BASELINE, "block_answer_from_copy1"]
    _rows = []
    for _model, _sub in panels.items():
        for _arm in _arms:
            if _arm not in _sub.columns:
                continue
            _hits, _n = int(_sub[_arm].sum()), len(_sub)
            _lo, _hi = proportion_confint(_hits, _n, method="wilson")
            _rows.append(
                {
                    "model": _model,
                    "arm": PATHWAY_DISPLAY[_arm],
                    "n": _n,
                    "accuracy": _hits / _n,
                    "ci_low": _lo,
                    "ci_high": _hi,
                }
            )
    block_c_acc = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    _models = sorted(block_c_acc.model.unique(), key=by_size)
    _order = [PATHWAY_DISPLAY[a] for a in _arms]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_models, range=SERIES[: len(_models)]),
        legend=alt.Legend(title="Model"),
    )
    _x = alt.X("arm:N", sort=_order, title=None)
    _tip = [
        alt.Tooltip("model:N"),
        alt.Tooltip("arm:N"),
        alt.Tooltip("accuracy:Q", format=".1%"),
        alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
        alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
    ]
    block_c_chart = (
        alt.Chart(block_c_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(block_c_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=_x,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=_tip,
        )
    ).properties(width=480, height=260, title="Accuracy after blocking (C) (95% Wilson CIs)")
    block_c_chart
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    SINGLE,
    by_size,
    fmt_p,
    fmt_pp,
    mcnemar_pair,
    mo,
    panels: "dict[str, pd.DataFrame]",
    pd,
):
    _rows = []
    for _model, _sub in panels.items():
        if "block_answer_from_copy1" not in _sub.columns:
            continue
        _vs2 = mcnemar_pair(_sub[BASELINE], _sub["block_answer_from_copy1"])
        _vs1 = mcnemar_pair(_sub[SINGLE], _sub["block_answer_from_copy1"])
        _rows.append(
            {
                "model": _model,
                "Δ vs 2 copies (pp)": _vs2["delta"],
                "p vs 2 copies": _vs2["p"],
                "Δ vs 1 copy (pp)": _vs1["delta"],
                "p vs 1 copy": _vs1["p"],
            }
        )
    block_c_tests = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    mo.ui.table(
        block_c_tests,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "Δ vs 2 copies (pp)": fmt_pp,
            "Δ vs 1 copy (pp)": fmt_pp,
            "p vs 2 copies": fmt_p,
            "p vs 1 copy": fmt_p,
        },
        label="McNemar: block (C) vs 2 copies and vs 1 copy",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway:** Test is underpowered, we cannot reject either null. There is no evidence that (C) is or isn't necessary for improved performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Restricting (A): looking forward and back

    Blocking all of (A) kills the gain. But why?

    > **Hypothesis:** Performance gain comes from the model being able to "look forward in time" by letting the early tokens in copy 2 attend to the late tokens in copy 1.

    (Without copying, this kind of attention is blocked by the causal attention mask.)

    A similar method called Echo Embedding is introduced by [Springer et al. (2025)](https://arxiv.org/abs/2402.15449) to adapt autoregressive LMs into text embedding models.


    Let $t_{k,i}$ be the $i$th token in the $k$th copy of the prompt ($k \in \{1,2\}$).
    When $t_{2,i}$ attends to $t_{1,j}$ we say $t_{2,i}$ is "looking back" if $j<i$, and "looking forward" if $j>i$.
    Our hypothesis is that the benefit of copying comes from tokens in copy 2 "looking forward" by attending to tokens in copy 1.

    Example, sitting on **50** in copy 2. Arrows are attention from that token into copy 1:

    ```text
    copy 1: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
                                                      Looking back /\             /\ Looking forward
                                                                    --------  -----
                                                                            \/
    copy 2: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    ```

    Without a second copy, **50** cannot see "minutes of ... she earn?".
    With a copy, those tokens already occurred in copy 1, so looking forward is allowed.

    #### 2 More Experiments
    - **Block looking forward:** expect performance similar to no duplication
    - **Block looking back:** expect performance to stay about the same

    We run these on the same dataset with Gemma 3 12b and 27b.
    We do not prevent $t_{2,i}$ from attending to $t_{1,i}$, in order to help the model align itself.
    """)
    return


@app.cell(hide_code=True)
def _(MUTED, alt, mo, pd):
    _question = (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?"
    )
    _words = ["Weng", "earns", "$12", "hour", "50", "minutes", "earn?"]
    _labels = [f"1:{w}" for w in _words] + [f"2:{w}" for w in _words]
    _n = len(_words)
    _rows = []
    for _qi, _q in enumerate(_labels):
        for _kj, _k in enumerate(_labels):
            _i, _j = _qi - _n, _kj
            if _kj > _qi:
                _region = "blocked by causal mask"
            elif _qi >= _n and _kj < _n:
                if _j < _i:
                    _region = "before — block looking back"
                elif _j > _i:
                    _region = "after — block looking forward"
                else:
                    _region = "aligned token (left on)"
            else:
                _region = "allowed by causal mask"
            _rows.append({"token_i": _q, "can_see": _k, "region": _region})
    _mask = pd.DataFrame(_rows)
    _domain = [
        "before — block looking back",
        "after — block looking forward",
        "aligned token (left on)",
        "allowed by causal mask",
        "blocked by causal mask",
    ]
    _range = ["#2a78d6", "#eb6834", MUTED, "#d7e3d4", "#f3f1ec"]
    _heatmap = (
        alt.Chart(_mask)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "token_i:N",
                sort=_labels,
                title="The sequence: token i (copy 1, then copy 2)",
                axis=alt.Axis(labelAngle=-40, labelLimit=90),
            ),
            y=alt.Y(
                "can_see:N",
                sort=list(reversed(_labels)),
                title="What token i can attend to",
            ),
            color=alt.Color(
                "region:N",
                scale=alt.Scale(domain=_domain, range=_range),
                legend=alt.Legend(title=None, orient="bottom", columns=1, labelLimit=320),
            ),
            tooltip=[
                alt.Tooltip("token_i:N", title="Token i"),
                alt.Tooltip("can_see:N", title="Can see"),
                alt.Tooltip("region:N", title="Region"),
            ],
        )
        .properties(
            width=440,
            height=440,
            title="Pick a column (a token). Read up the column for what it can see.",
        )
    )
    mask_figure = mo.vstack(
        [
            mo.md(f"**Copy 1 then copy 2:** {_question}"),
            _heatmap,
        ]
    )
    mask_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Block looking forward (orange)

    Copy 2 cannot read later copy-1 tokens. It can still look back (and at the aligned token).
    """)
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    PATHWAY_DISPLAY,
    SERIES,
    SINGLE,
    alt,
    by_size,
    panels: "dict[str, pd.DataFrame]",
    pd,
    proportion_confint,
):
    _arms = [
        SINGLE,
        BASELINE,
        "copy2_past_or_aligned_copy1",
    ]
    _rows = []
    for _model, _sub in panels.items():
        if _model.endswith("4b"):
            continue
        for _arm in _arms:
            if _arm not in _sub.columns:
                continue
            _hits, _n = int(_sub[_arm].sum()), len(_sub)
            _lo, _hi = proportion_confint(_hits, _n, method="wilson")
            _rows.append(
                {
                    "model": _model,
                    "arm": PATHWAY_DISPLAY[_arm],
                    "n": _n,
                    "accuracy": _hits / _n,
                    "ci_low": _lo,
                    "ci_high": _hi,
                }
            )
    lookback_acc = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    _models = sorted(lookback_acc.model.unique(), key=by_size)
    _order = [PATHWAY_DISPLAY[a] for a in _arms]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_models, range=SERIES[: len(_models)]),
        legend=alt.Legend(title="Model"),
    )
    _x = alt.X("arm:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-20))
    _tip = [
        alt.Tooltip("model:N"),
        alt.Tooltip("arm:N"),
        alt.Tooltip("accuracy:Q", format=".1%"),
        alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
        alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
    ]
    lookback_chart = (
        alt.Chart(lookback_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(lookback_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=_x,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=_tip,
        )
    ).properties(
        width=480, height=260, title="Accuracy after blocking looking forward (95% Wilson CIs)"
    )
    lookback_chart
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    SINGLE,
    by_size,
    fmt_p,
    fmt_pp,
    mcnemar_pair,
    mo,
    panels: "dict[str, pd.DataFrame]",
    pd,
):
    _masked = ["copy2_past_or_aligned_copy1"]
    _rows = []
    for _model, _sub in panels.items():
        if _model.endswith("4b"):
            continue
        for _arm in _masked:
            if _arm not in _sub.columns:
                continue
            _vs2 = mcnemar_pair(_sub[BASELINE], _sub[_arm])
            _vs1 = mcnemar_pair(_sub[SINGLE], _sub[_arm])
            _rows.append(
                {
                    "model": _model,
                    "Δ vs 2 copies (pp)": _vs2["delta"],
                    "p vs 2 copies": _vs2["p"],
                    "Δ vs 1 copy (pp)": _vs1["delta"],
                    "p vs 1 copy": _vs1["p"],
                }
            )
    lookback_tests = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    mo.ui.table(
        lookback_tests,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "Δ vs 2 copies (pp)": fmt_pp,
            "Δ vs 1 copy (pp)": fmt_pp,
            "p vs 2 copies": fmt_p,
            "p vs 1 copy": fmt_p,
        },
        label="McNemar: block looking forward vs 2 copies and vs 1 copy",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Block looking back (blue)

    Copy 2 cannot read earlier copy-1 tokens. It can still look forward (and at the aligned token).
    """)
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    PATHWAY_DISPLAY,
    SERIES,
    SINGLE,
    alt,
    by_size,
    panels: "dict[str, pd.DataFrame]",
    pd,
    proportion_confint,
):
    _arms = [
        SINGLE,
        BASELINE,
        "copy2_future_or_aligned_copy1",
    ]
    _rows = []
    for _model, _sub in panels.items():
        if _model.endswith("4b"):
            continue
        for _arm in _arms:
            if _arm not in _sub.columns:
                continue
            _hits, _n = int(_sub[_arm].sum()), len(_sub)
            _lo, _hi = proportion_confint(_hits, _n, method="wilson")
            _rows.append(
                {
                    "model": _model,
                    "arm": PATHWAY_DISPLAY[_arm],
                    "n": _n,
                    "accuracy": _hits / _n,
                    "ci_low": _lo,
                    "ci_high": _hi,
                }
            )
    lookfwd_acc = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    _models = sorted(lookfwd_acc.model.unique(), key=by_size)
    _order = [PATHWAY_DISPLAY[a] for a in _arms]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_models, range=SERIES[: len(_models)]),
        legend=alt.Legend(title="Model"),
    )
    _x = alt.X("arm:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-20))
    _tip = [
        alt.Tooltip("model:N"),
        alt.Tooltip("arm:N"),
        alt.Tooltip("accuracy:Q", format=".1%"),
        alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
        alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
    ]
    lookfwd_chart = (
        alt.Chart(lookfwd_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(lookfwd_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=_x,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=_tip,
        )
    ).properties(
        width=480,
        height=260,
        title="Accuracy after blocking looking back (95% Wilson CIs)",
    )
    lookfwd_chart
    return


@app.cell(hide_code=True)
def _(
    BASELINE,
    SINGLE,
    by_size,
    fmt_p,
    fmt_pp,
    mcnemar_pair,
    mo,
    panels: "dict[str, pd.DataFrame]",
    pd,
):
    _masked = ["copy2_future_or_aligned_copy1"]
    _rows = []
    for _model, _sub in panels.items():
        if _model.endswith("4b"):
            continue
        for _arm in _masked:
            if _arm not in _sub.columns:
                continue
            _vs2 = mcnemar_pair(_sub[BASELINE], _sub[_arm])
            _vs1 = mcnemar_pair(_sub[SINGLE], _sub[_arm])
            _rows.append(
                {
                    "model": _model,
                    "Δ vs 2 copies (pp)": _vs2["delta"],
                    "p vs 2 copies": _vs2["p"],
                    "Δ vs 1 copy (pp)": _vs1["delta"],
                    "p vs 1 copy": _vs1["p"],
                }
            )
    lookfwd_tests = (
        pd.DataFrame(_rows)
        .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
        .reset_index(drop=True)
    )
    mo.ui.table(
        lookfwd_tests,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "Δ vs 2 copies (pp)": fmt_pp,
            "Δ vs 1 copy (pp)": fmt_pp,
            "p vs 2 copies": fmt_p,
            "p vs 1 copy": fmt_p,
        },
        label="McNemar: block looking back vs 2 copies and vs 1 copy",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway:** The two-copy gain lives in looking *forward* along (A).
    Blocking looking forward returns performance to one copy; blocking looking
    back does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limitations
    - Only 1 question dataset. May not replicate on non-mathematical problems
    - Length matched control: Redwood said filler tokens don't replicate, causal intervention and lack of improvement on more than 2 copies makes it unlikely, but should still try
    - Had to use JSON schema decoding to reliably prevent reasoning. Possible this is weird OOD behavior
    """)
    return


if __name__ == "__main__":
    app.run()
