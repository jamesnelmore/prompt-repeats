# /// script
# requires-python = ">=3.12"
# ///
"""Portfolio overview of the prompt-repeat GSM8K evals.

Run with:  uv run marimo edit notebooks/portfolio.py
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis
    Status: Tenative results from early investigation
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
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
        multipletests,
        np,
        pd,
        pretty_model,
        proportion_confint,
        read_eval_log,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Past work](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens) by Ryan Greenblatt found that non-reasoning performance for frontier models is improved by copying a prompt multiple times. Normally, an LLM input might look something like this.


    ```
    User:
    	[Instructions]

    	[Question]

    Assistant:
    	ANSWER:
    ```

    He found that this

    ```
    User:
    	[Instructions]

    	[Question copy 1]
    	[Question copy 2]
    	[Question copy 3]
    	...

    Assistant:
    	ANSWER:
    ```

    improves performance in certain frontier LLMs.


    In this project I test other other models for the same behavior and do a shallow investigation into why it occurs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gemma 3 Benefits from prompy copying
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I tested various models' nonreasoning performance on GSM8k, using a structured response schema to prevent models from reasoning
    """)
    return


@app.cell
def _(Path, hashlib, pd, read_eval_log):
    # Pull in inspect evals data

    ROOT = Path(__file__).parent.parent
    LOG_DIR = ROOT / "logs"
    CACHE_DIR = ROOT / "data" / "cache"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    MUTED = "#898781"

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

    samples = load_samples()
    return MUTED, SERIES, samples


@app.cell
def _(np, pd, pretty_model, proportion_confint, samples):
    N_BOOT = 2000

    def summary_frame(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
        """Accuracy + recovery of the CoT gap, per (model, copy count)."""
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
                        "base_acc": base.mean(),
                        "recovery": (arm.mean() - base.mean()) / gap if gap > 0 else np.nan,
                        "rec_lo": r_lo,
                        "rec_hi": r_hi,
                    }
                )
        return pd.DataFrame(rows)

    summary = summary_frame(samples)
    models = sorted(summary.model.unique())
    return models, summary


@app.cell
def _(mo, models):
    model_picker = mo.ui.multiselect(
        options=models,
        value=models,
        label="Models",
    )
    model_picker
    return (model_picker,)


@app.cell
def _(SERIES, alt, mo, model_picker, models, summary):
    # Chart 1: x axis is number of repeats, y is accuracy with 95% CIs and h-lines showing per model CoT ceiling
    visible = summary[summary.model.isin(model_picker.value)]
    color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=models, range=SERIES[: len(models)]),
        legend=alt.Legend(title="Model", orient="bottom", columns=2),
    )
    x = alt.X(
        "copies:Q",
        scale=alt.Scale(type="log", base=2, nice=False),
        axis=alt.Axis(values=sorted(summary.copies.unique()), title="Question copies"),
    )
    _tip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("copies:Q", title="Copies"),
        alt.Tooltip("accuracy:Q", format=".1%", title="Accuracy"),
        alt.Tooltip("acc_lo:Q", format=".1%", title="CI low"),
        alt.Tooltip("acc_hi:Q", format=".1%", title="CI high"),
        alt.Tooltip("cot_acc:Q", format=".1%", title="CoT ceiling"),
    ]
    _base = alt.Chart(visible)
    acc_chart = mo.ui.altair_chart(
        (
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
            width=520, height=320, title="No-CoT accuracy vs copies (dashed = CoT ceiling)"
        ),
        legend_selection=["model"],
    )
    acc_chart
    return acc_chart, color, visible, x


@app.cell
def _(MUTED, alt, color, mo, pd, visible, x):
    # Chart 2: x axis is number of repeats, y is recovery from non Cot to Cot baseline with 95% CIs
    _tip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("copies:Q", title="Copies"),
        alt.Tooltip("recovery:Q", format=".1%", title="Recovery"),
        alt.Tooltip("rec_lo:Q", format=".1%", title="CI low"),
        alt.Tooltip("rec_hi:Q", format=".1%", title="CI high"),
        alt.Tooltip("accuracy:Q", format=".1%", title="Accuracy"),
    ]
    _y_scale = alt.Scale(
        domain=[min(-0.12, float(visible.rec_lo.min()) - 0.07), max(1.05, float(visible.rec_hi.max()) + 0.05)]
    )
    _y = alt.Y(
        "recovery:Q",
        scale=_y_scale,
        axis=alt.Axis(format="%", title="Recovery of CoT gap", grid=True),
    )
    _anchors = pd.DataFrame({"y": [0.0, 1.0], "label": ["no-CoT, 1 copy", "CoT ceiling"]})
    _base = alt.Chart(visible)
    rec_chart = mo.ui.altair_chart(
        (
            alt.Chart(_anchors)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=MUTED)
            .encode(y=alt.Y("y:Q", scale=_y_scale))
            + alt.Chart(_anchors)
            .mark_text(align="left", dx=4, dy=-6, fontSize=11, color=MUTED)
            .encode(x=alt.value(4), y=alt.Y("y:Q", scale=_y_scale), text="label:N")
            + _base.mark_rule(strokeWidth=1.5, opacity=0.65).encode(
                x=x, y=alt.Y("rec_lo:Q", scale=_y_scale, title=""), y2="rec_hi:Q", color=color, tooltip=_tip
            )
            + _base.mark_line(point=True).encode(x=x, y=_y, color=color, tooltip=_tip)
        ).properties(
            width=520,
            height=320,
            title="Share of each model's CoT gap recovered by repeating the question",
        ),
        legend_selection=["model"],
    )
    rec_chart
    return (rec_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is 2 copies better than 1?

    Same problems on every arm, so McNemar's test on the discordant items
    (fixed vs broken). One row per model, **1 copy vs 2**. `sig` stars
    raw \(p < 0.001\); `p_holm` is Holm-corrected across models.
    """)
    return


@app.cell
def _(mcnemar, multipletests, pd, pretty_model, samples):
    P_STAR = 1e-3

    def intervention_tests(df: pd.DataFrame) -> pd.DataFrame:
        """McNemar 1 copy vs 2 copies, per model."""
        rows = []
        for model, sub in df[~df.use_cot].groupby("model"):
            wide = (
                sub.pivot_table(index="sample_id", columns="copies", values="correct")
                .dropna()
                .astype(bool)
            )
            if not {1, 2} <= set(wide.columns):
                continue
            base, arm = wide[1], wide[2]
            fixed = int((~base & arm).sum())
            broken = int((base & ~arm).sum())
            table = [
                [int((base & arm).sum()), broken],
                [fixed, int((~base & ~arm).sum())],
            ]
            res = mcnemar(table, exact=(fixed + broken) < 25, correction=True)
            rows.append(
                {
                    "model": pretty_model(model),
                    "fixed": fixed,
                    "broken": broken,
                    "p": float(res.pvalue),
                    "n": len(wide),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["p_holm"] = multipletests(out.p, method="holm")[1]
        out["sig"] = ["*" if p < P_STAR else "" for p in out.p]
        return out[["model", "fixed", "broken", "p", "p_holm", "sig", "n"]]

    tests = intervention_tests(samples)
    return (tests,)


@app.cell
def _(mo, tests):
    def _fmt_p(v: float) -> str:
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    _starred = int((tests.sig == "*").sum()) if not tests.empty else 0
    mo.vstack(
        [
            mo.ui.table(
                tests,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={"p": _fmt_p, "p_holm": _fmt_p, "n": "{:,}"},
                label="McNemar: 1 copy vs 2 copies",
            ),
            mo.md(
                f"**{_starred} of {len(tests)}** models starred at raw p < 0.001. "
                "A star only says two copies beat one — effect size is in the charts above."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that effect appears to scale *roughly* with model size.
    """)
    return


@app.cell
def _(acc_chart, mo, rec_chart, visible):
    # Table: one row per model, one column per copy count (mean ± CI half-width)
    _chart = rec_chart if rec_chart.selections and not acc_chart.selections else acc_chart
    _selected = _chart.apply_selection(visible)

    def _cell(mean: float, lo: float, hi: float) -> str:
        return f"{mean:.1%} ± {(hi - lo) / 2:.1%}"

    _long = _selected.assign(
        cell=[
            _cell(m, lo, hi)
            for m, lo, hi in zip(_selected.accuracy, _selected.acc_lo, _selected.acc_hi)
        ],
        copies=_selected.copies.map(lambda c: f"{int(c)} copies"),
    )
    _table = (
        _long.pivot(index="model", columns="copies", values="cell")
        .reindex(columns=[f"{int(c)} copies" for c in sorted(_selected.copies.unique())])
        .reset_index()
    )
    # CoT ceiling is constant per model; attach once for reference.
    _cot = (
        _selected[["model", "cot_acc"]]
        .drop_duplicates("model")
        .assign(**{"CoT Acc.": lambda d: d.cot_acc.map("{:.1%}".format)})
        [["model", "CoT Acc."]]
    )
    _table = _cot.merge(_table, on="model")[
        ["model", "CoT Acc.", *[c for c in _table.columns if c != "model"]]
    ]
    mo.ui.table(_table, selection=None, pagination=False, show_column_summaries=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Causal Interventions on the Gemma 3 family

    The previous section showed that adding a second copy of a question improves model performance.
    This one will precent the second copy and the answer from attending to the first copy and measure the result.

    - Intervention 1: Prevent copy 2 from attending to copy 1
    - Intervention 2: Prevent answer tokens from attending to copy 1
    """)
    return


@app.cell
def _(Path, pd, pretty_model):
    # load data for this section
    ABLATION_DIR = Path(__file__).parent.parent / "ablation_results"
    BASELINE, SINGLE = "2copies", "1copy"
    DISPLAY = {
        "1copy": "1 copy",
        "2copies": "2 copies",
        "block_copy2_from_copy1": "block copy 2",
        "block_answer_from_copy1": "block answer",
        "copy2_strictly_past_copy1": "Cross-copy causal (strict)",
        "copy2_past_or_aligned_copy1": "Cross-copy causal (inclusive)",
    }
    PATHWAY_ARMS = ["block_copy2_from_copy1", "block_answer_from_copy1"]
    CAUSAL_ARMS = ["copy2_strictly_past_copy1", "copy2_past_or_aligned_copy1"]

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
    ablation = pd.concat(_frames, ignore_index=True)
    wide = (
        ablation.pivot(
            index=["model", "gsm8k_test_index"], columns="arm", values="correct"
        )
        .dropna()
        .astype(bool)
    )
    short = {m: pretty_model(m) for m in ablation.model.unique()}
    return (
        BASELINE,
        CAUSAL_ARMS,
        DISPLAY,
        PATHWAY_ARMS,
        SINGLE,
        by_size,
        short,
        wide,
    )


@app.cell
def _(
    BASELINE,
    DISPLAY,
    PATHWAY_ARMS,
    SERIES,
    SINGLE,
    alt,
    by_size,
    mo,
    pd,
    proportion_confint,
    short,
    wide,
):
    # Data and results
    pathway_arms = [SINGLE, BASELINE, *PATHWAY_ARMS]

    def arm_accuracy(arms: list[str]) -> pd.DataFrame:
        rows = []
        for model, sub in wide.groupby(level="model"):
            for arm in arms:
                if arm not in sub.columns:
                    continue
                hits, n = int(sub[arm].sum()), len(sub)
                lo, hi = proportion_confint(hits, n, method="wilson")
                rows.append(
                    {
                        "model": short[model],
                        "arm": DISPLAY.get(arm, arm),
                        "arm_key": arm,
                        "n": n,
                        "accuracy": hits / n,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values("model", key=lambda c: c.map(by_size), kind="stable")
            .reset_index(drop=True)
        )

    pathway_acc = arm_accuracy(pathway_arms)
    ablation_models = sorted(pathway_acc.model.unique(), key=by_size)
    _arm_order = [DISPLAY[a] for a in pathway_arms if a in DISPLAY]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=ablation_models, range=SERIES[: len(ablation_models)]),
        legend=alt.Legend(title="Model"),
    )
    pathway_chart = (
        alt.Chart(pathway_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X("arm:N", sort=_arm_order, title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("arm:N"),
                alt.Tooltip("accuracy:Q", format=".1%"),
                alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
                alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
            ],
        )
    )
    pathway_err = (
        alt.Chart(pathway_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=alt.X("arm:N", sort=_arm_order), y="ci_low:Q", y2="ci_high:Q", color=_color)
    )
    mo.vstack(
        [
            mo.md("Accuracy under the two pathway blocks (95% Wilson CIs)"),
            mo.ui.altair_chart((pathway_err + pathway_chart).properties(width=520, height=280)),
            mo.ui.table(
                pathway_acc[["model", "arm", "accuracy", "ci_low", "ci_high", "n"]],
                selection=None,
                pagination=False,
                show_column_summaries=False,
                format_mapping={
                    "accuracy": "{:.1%}".format,
                    "ci_low": "{:.1%}".format,
                    "ci_high": "{:.1%}".format,
                    "n": "{:,}".format,
                },
            ),
        ]
    )
    return ablation_models, arm_accuracy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Model doesn't really benefit from final token seeing 2 copies of the prompt, it benefits from the second copy attending to the first
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which parts?


    - New intervention: Prevent each token in copy2 from attending to copy1 tokens that come after it.

    The two causal interventions differ only at the diagonal — whether
    copy 2 token *i* can see the *aligned* copy 1 token at the same
    ordinal position:

    ```
      Copy 1 token position:   t1  t2  t3  t4  t5
                                ────────────────────
      Copy 2 token t1 sees:   [.]  x   x   x   x    <- inclusive: aligned; strict: nothing
      Copy 2 token t2 sees:    v  [.]  x   x   x    <- inclusive: t1+t2; strict: t1 only
      Copy 2 token t3 sees:    v   v  [.]  x   x
      Copy 2 token t4 sees:    v   v   v  [.]  x
      Copy 2 token t5 sees:    v   v   v   v  [.]

      v = allowed   x = blocked   [.] = aligned position (allowed in inclusive, blocked in strict)
    ```
    """)
    return


@app.cell
def _(
    BASELINE,
    CAUSAL_ARMS,
    DISPLAY,
    SERIES,
    SINGLE,
    ablation_models,
    alt,
    arm_accuracy,
    mo,
):
    # Data showing this seemed to do it
    causal_arms = [BASELINE, SINGLE, *CAUSAL_ARMS]
    causal_acc = arm_accuracy(causal_arms)
    _arm_order = [DISPLAY[a] for a in causal_arms if a in DISPLAY]
    _color = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=_arm_order, range=SERIES[: len(_arm_order)]),
        legend=alt.Legend(title="Intervention"),
    )
    _x = alt.X("model:N", sort=ablation_models, title=None)
    _xoff = alt.XOffset("arm:N", sort=_arm_order)
    causal_chart = (
        alt.Chart(causal_acc)
        .mark_point(filled=True, size=80)
        .encode(
            x=_x,
            xOffset=_xoff,
            y=alt.Y("accuracy:Q", title="Accuracy", axis=alt.Axis(format="%")),
            color=_color,
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("arm:N"),
                alt.Tooltip("accuracy:Q", format=".1%"),
                alt.Tooltip("ci_low:Q", format=".1%", title="CI low"),
                alt.Tooltip("ci_high:Q", format=".1%", title="CI high"),
            ],
        )
    )
    causal_err = (
        alt.Chart(causal_acc)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(x=_x, xOffset=_xoff, y="ci_low:Q", y2="ci_high:Q", color=_color)
    )
    mo.vstack(
        [
            mo.md(
                "Restricting copy2→copy1 attention to past (or past+aligned) tokens "
                "removes most of the two-copy gain."
            ),
            mo.ui.altair_chart((causal_err + causal_chart).properties(width=520, height=280)),
            mo.ui.table(
                causal_acc[["model", "arm", "accuracy", "ci_low", "ci_high", "n"]],
                selection=None,
                pagination=False,
                show_column_summaries=False,
                format_mapping={
                    "accuracy": "{:.1%}".format,
                    "ci_low": "{:.1%}".format,
                    "ci_high": "{:.1%}".format,
                    "n": "{:,}".format,
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paired tests (McNemar + Holm)

    Same GSM8K items on every arm. Per model: `repeat_effect` = 2copies vs 1copy;
    `vs_2copies` / `vs_1copy` = each masked arm vs those baselines. Holm correction
    is within-model across that model's comparisons. `delta` is paired accuracy
    change of `arm` relative to `reference` (percentage points).
    """)
    return


@app.cell
def _(
    BASELINE,
    DISPLAY,
    SINGLE,
    by_size,
    math,
    mcnemar,
    multipletests,
    pd,
    short,
    wide,
):
    ALPHA = 0.05

    def paired_test(base: pd.Series, arm: pd.Series) -> dict[str, float]:
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
            "n": n,
            "broken": broken,
            "fixed": fixed,
            "delta": delta,
            "delta_low": delta - 1.96 * se,
            "delta_high": delta + 1.96 * se,
            "p": float(res.pvalue),
        }

    _masked = [a for a in wide.columns if a not in (BASELINE, SINGLE)]
    _plan = [("repeat_effect", SINGLE, BASELINE)]
    _plan += [("vs_2copies", BASELINE, arm) for arm in _masked]
    _plan += [("vs_1copy", SINGLE, arm) for arm in _masked]

    _rows = []
    for model, sub in wide.groupby(level="model"):
        for family, reference, arm in _plan:
            _rows.append(
                {
                    "model": short[model],
                    "family": family,
                    "reference": DISPLAY.get(reference, reference),
                    "arm": DISPLAY.get(arm, arm),
                    **paired_test(sub[reference], sub[arm]),
                }
            )
    ablation_tests = pd.DataFrame(_rows)
    ablation_tests["p_holm"] = ablation_tests.groupby("model").p.transform(
        lambda p: multipletests(p, method="holm")[1]
    )
    ablation_tests["sig"] = ablation_tests.p_holm < ALPHA
    ablation_tests = ablation_tests.sort_values(
        "model", key=lambda c: c.map(by_size), kind="stable"
    ).reset_index(drop=True)
    return ALPHA, ablation_tests


@app.cell
def _(ablation_tests, by_size, mo, pd):
    def _fmt_p(v: float) -> str:
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    def _fmt_pp(v: float) -> str:
        return f"{v * 100:+.2f}"

    _view = ablation_tests.assign(sig=ablation_tests.sig.map({True: "*", False: ""}))[
        [
            "model",
            "family",
            "reference",
            "arm",
            "broken",
            "fixed",
            "delta",
            "delta_low",
            "delta_high",
            "p",
            "p_holm",
            "sig",
        ]
    ]

    _base_label, _single_label = "2 copies", "1 copy"
    _vs_base = ablation_tests[ablation_tests.family == "vs_2copies"].set_index(
        ["model", "arm"]
    )
    _vs_one = ablation_tests[ablation_tests.family == "vs_1copy"].set_index(
        ["model", "arm"]
    )
    _has_gain = (
        ablation_tests[ablation_tests.family == "repeat_effect"]
        .set_index("model")
        .apply(lambda r: bool(r.sig and r.delta > 0), axis=1)
        .to_dict()
    )
    _verdicts = []
    for key, base_row in _vs_base.iterrows():
        one_row = _vs_one.loc[key]
        hurts = bool(base_row.sig and base_row.delta < 0)
        above_single = bool(one_row.sig and one_row.delta > 0)
        if not _has_gain[key[0]]:
            verdict = "n/a — no established repeat gain"
        elif not hurts:
            verdict = f"no detectable role (≈ {_base_label})"
        elif above_single:
            verdict = "carries part of the gain"
        else:
            verdict = f"carries the whole gain (≈ {_single_label})"
        _verdicts.append(
            {
                "model": key[0],
                "blocked pathway": key[1],
                f"vs {_base_label} (pp)": base_row.delta * 100,
                f"vs {_single_label} (pp)": one_row.delta * 100,
                "verdict": verdict,
            }
        )
    verdicts = (
        pd.DataFrame(_verdicts)
        .sort_values(
            ["model", f"vs {_base_label} (pp)"],
            key=lambda c: c.map(by_size) if c.name == "model" else c,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    mo.vstack(
        [
            mo.ui.table(
                _view,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={
                    "delta": _fmt_pp,
                    "delta_low": _fmt_pp,
                    "delta_high": _fmt_pp,
                    "p": _fmt_p,
                    "p_holm": _fmt_p,
                },
                label="McNemar ablation contrasts (Holm within model)",
            ),
            mo.md("### Verdict per blocked pathway"),
            mo.ui.table(
                verdicts,
                selection=None,
                pagination=False,
                show_column_summaries=False,
                show_data_types=False,
                format_mapping={
                    f"vs {_base_label} (pp)": "{:+.2f}".format,
                    f"vs {_single_label} (pp)": "{:+.2f}".format,
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Next Experiment:** Do the reverse, i.e. block each copy2 token from attending to copy1 tokens that come before it, and hopefully see that it does very little.
    """)
    return


if __name__ == "__main__":
    app.run()
