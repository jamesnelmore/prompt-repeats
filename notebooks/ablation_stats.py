# /// script
# requires-python = ">=3.12"
# ///
"""Paired significance tests for the cross-copy attention ablation.

Run with:  uv run marimo edit notebooks/ablation_stats.py

Every arm is evaluated on the same pinned GSM8K rows, so each pair of arms is
matched item-by-item and McNemar's test is the right instrument: it looks only
at the items where the two arms disagree.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Which cross-copy pathways actually carry the repeat gain?

    Each arm answers the same 1319 GSM8K questions, so the arms are **paired**
    per item. McNemar's test uses only the discordant items --- the ones one arm
    gets right and the other gets wrong --- which is what gives it far more power
    here than comparing two independent accuracy numbers.

    Three questions, asked separately:

    | family | comparison | what a significant result means |
    |---|---|---|
    | `repeat_effect` | `2copies` vs `1copy` | repeating the prompt changes accuracy at all |
    | `vs_2copies` | masked arm vs `2copies` | the blocked pathway carries some of the gain |
    | `vs_1copy` | masked arm vs `1copy` | the masked arm still beats a single copy |

    The last two together identify **how much** of the gain a pathway carries: an
    arm that drops below `2copies` *and* is indistinguishable from `1copy` has had
    the whole effect removed.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import math
    from pathlib import Path

    import altair as alt
    import pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint

    ROOT = Path(__file__).parent.parent
    RESULTS_DIR = ROOT / "ablation_results"

    BASELINE = "2copies"
    SINGLE = "1copy"
    ALPHA = 0.05

    # Validated categorical slots, in order (see the data-viz palette: the first
    # three clear the all-pairs CVD and normal-vision floors in both modes).
    # Slot 3 sits under 3:1 on the light surface, so the table view above the
    # chart is the required relief -- keep it if you add models.
    SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

    def by_size(tag: str) -> float:
        """Sort '4b' < '12b' < '27b'; plain sorted() would give 12b, 27b, 4b."""
        digits = "".join(c for c in tag if c.isdigit())
        return float(digits) if digits else float("inf")

    return (
        ALPHA,
        BASELINE,
        RESULTS_DIR,
        SERIES_COLORS,
        SINGLE,
        alt,
        by_size,
        math,
        mcnemar,
        multipletests,
        pd,
        proportion_confint,
    )


@app.cell
def _(RESULTS_DIR, pd):
    def load_samples() -> pd.DataFrame:
        """One row per (model, question, arm) across every full ablation run.

        Smoke directories are skipped -- they are 10-question sanity checks, not
        results, and pooling them would silently contaminate the counts.
        """
        frames = [
            pd.read_json(path, lines=True)
            for path in sorted(RESULTS_DIR.glob("attention_*/samples.jsonl"))
            if not path.parent.name.endswith("_smoke")
        ]
        if not frames:
            raise FileNotFoundError(f"no full-run samples.jsonl under {RESULTS_DIR}")
        return pd.concat(frames, ignore_index=True)

    samples = load_samples()

    # One arm result per (model, question), or the pairing below is not a pairing.
    _key = ["model", "gsm8k_test_index", "arm"]
    assert not samples.duplicated(_key).any(), "duplicate (model, question, arm) rows"

    wide = (
        samples.pivot(
            index=["model", "gsm8k_test_index"], columns="arm", values="correct"
        )
        .dropna()
        .astype(bool)
    )
    short_name = {m: m.rsplit("-", 2)[-2] for m in samples.model.unique()}
    arms = [c for c in samples.arm.unique() if c in wide.columns]
    return arms, samples, short_name, wide


@app.cell(hide_code=True)
def _(mo, samples, wide):
    mo.md(f"""
    Loaded **{len(samples):,}** records --- {len(wide):,} paired
    (model, question) rows across **{samples.model.nunique()}** models and
    **{samples.arm.nunique()}** arms.
    """)
    return


@app.cell
def _(arms, by_size, pd, proportion_confint, short_name, wide):
    def accuracy_table() -> pd.DataFrame:
        """Per-arm accuracy with Wilson intervals, to match the README's format."""
        rows = []
        for model, sub in wide.groupby(level="model"):
            for arm in arms:
                hits, n = int(sub[arm].sum()), len(sub)
                lo, hi = proportion_confint(hits, n, method="wilson")
                rows.append(
                    {
                        "model": short_name[model],
                        "arm": arm,
                        "n": n,
                        "accuracy": hits / n,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
        out = pd.DataFrame(rows)
        return out.sort_values(
            "model", key=lambda c: c.map(by_size), kind="stable"
        ).reset_index(drop=True)

    accuracy = accuracy_table()
    return (accuracy,)


@app.cell(hide_code=True)
def _(accuracy, mo):
    mo.md("## Per-arm accuracy")
    _fmt = {
        "accuracy": "{:.1%}".format,
        "ci_low": "{:.1%}".format,
        "ci_high": "{:.1%}".format,
        "n": "{:,}".format,
    }
    mo.ui.table(
        accuracy,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping=_fmt,
    )
    return


@app.cell
def _(
    ALPHA,
    BASELINE,
    SINGLE,
    arms,
    by_size,
    math,
    mcnemar,
    multipletests,
    pd,
    short_name,
    wide,
):
    def paired_test(base: pd.Series, arm: pd.Series) -> dict[str, float]:
        """McNemar on one matched pair of arms, plus the paired accuracy delta.

        `broken` and `fixed` are the discordant cells and are the only thing the
        test sees. The interval on the delta is the standard Wald interval for a
        paired difference of proportions, which needs a decent number of
        discordant pairs -- fine at these counts, thin if it ever drops near zero.
        """
        n = len(base)
        broken = int((base & ~arm).sum())
        fixed = int((~base & arm).sum())
        table = [
            [int((base & arm).sum()), broken],
            [fixed, int((~base & ~arm).sum())],
        ]
        # Exact binomial when discordant pairs are too few for the chi-square
        # approximation; continuity-corrected chi-square above.
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

    def all_tests() -> pd.DataFrame:
        """Every comparison, Holm-corrected within each model."""
        masked = [a for a in arms if a not in (BASELINE, SINGLE)]
        plan = [("repeat_effect", SINGLE, BASELINE)]
        plan += [("vs_2copies", BASELINE, arm) for arm in masked]
        plan += [("vs_1copy", SINGLE, arm) for arm in masked]

        rows = []
        for model, sub in wide.groupby(level="model"):
            for family, reference, arm in plan:
                rows.append(
                    {
                        "model": short_name[model],
                        "family": family,
                        "reference": reference,
                        "arm": arm,
                        **paired_test(sub[reference], sub[arm]),
                    }
                )
        out = pd.DataFrame(rows)
        # Correct within a model: each model is an independent set of questions,
        # and the families are read together as one story per model.
        out["p_holm"] = (
            out.groupby("model").p.transform(
                lambda p: multipletests(p, method="holm")[1]
            )
        )
        out["sig"] = out.p_holm < ALPHA
        return out.sort_values(
            "model", key=lambda c: c.map(by_size), kind="stable"
        ).reset_index(drop=True)

    tests = all_tests()
    return (tests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## McNemar results

    `delta` is the paired accuracy change of `arm` relative to `reference`, in
    percentage points. `broken` / `fixed` are the discordant counts the test runs
    on. `p_holm` is Holm-corrected across all nine comparisons within a model.
    """)
    return


@app.cell(hide_code=True)
def _(mo, tests):
    def fmt_p(v: float) -> str:
        """Small p-values need an exponent; a fixed 3dp floors them all to 0.000."""
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    def fmt_pp(v: float) -> str:
        return f"{v * 100:+.2f}"

    _view = tests.assign(sig=tests.sig.map({True: "*", False: ""}))[
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
    mo.ui.table(
        _view,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "delta": fmt_pp,
            "delta_low": fmt_pp,
            "delta_high": fmt_pp,
            "p": fmt_p,
            "p_holm": fmt_p,
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Effect sizes against the two-copy baseline

    Points left of the zero line are arms that lost accuracy when the pathway was
    blocked. Bars are 95% intervals on the paired difference; an interval clear of
    zero is the same call the Holm-corrected test makes.
    """)
    return


@app.cell
def _(SERIES_COLORS, alt, by_size, pd, tests):
    def forest(df: pd.DataFrame) -> alt.LayerChart:
        """Paired deltas vs the two-copy baseline, one row per blocked pathway."""
        data = df[df.family == "vs_2copies"].assign(
            label=lambda d: d.arm.str.replace("_", " "),
            pp=lambda d: d.delta * 100,
            pp_low=lambda d: d.delta_low * 100,
            pp_high=lambda d: d.delta_high * 100,
        )
        order = (
            data.groupby("label").pp.mean().sort_values().index.tolist()
        )
        # Keep zero inside the domain with headroom, or the reference line lands
        # on the axis edge and reads as chart frame rather than as "no effect".
        span = alt.Scale(
            domain=[min(data.pp_low.min(), 0) - 0.6, max(data.pp_high.max(), 0) + 0.6],
            nice=False,
        )
        models = sorted(data.model.unique(), key=by_size)
        if len(models) > len(SERIES_COLORS):
            raise ValueError(
                f"{len(models)} models but only {len(SERIES_COLORS)} validated "
                "slots -- add a validated hue or facet instead of cycling"
            )
        colors = alt.Scale(domain=models, range=SERIES_COLORS[: len(models)])
        y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        offset = alt.YOffset("model:N", sort=models)
        color = alt.Color("model:N", scale=colors, title="model", sort=models)

        zero = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="#c3c2b7", strokeWidth=1)
            .encode(x="x:Q")
        )
        interval = (
            alt.Chart(data)
            .mark_rule(strokeWidth=2)
            .encode(
                x=alt.X(
                    "pp_low:Q",
                    title="accuracy change vs 2copies (percentage points)",
                    scale=span,
                ),
                x2="pp_high:Q",
                y=y,
                yOffset=offset,
                color=color,
            )
        )
        dot = (
            alt.Chart(data)
            .mark_point(filled=True, size=90, stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x="pp:Q",
                y=y,
                yOffset=offset,
                color=color,
                tooltip=[
                    alt.Tooltip("model:N"),
                    alt.Tooltip("arm:N"),
                    alt.Tooltip("pp:Q", format="+.2f", title="delta (pp)"),
                    alt.Tooltip("broken:Q"),
                    alt.Tooltip("fixed:Q"),
                    alt.Tooltip("p_holm:Q", format=".2e", title="p (Holm)"),
                ],
            )
        )
        return (
            (zero + interval + dot)
            .properties(width=520, height=alt.Step(34))
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False, domainColor="#c3c2b7", labelColor="#52514e")
        )

    chart = forest(tests)
    chart
    return (chart,)


@app.cell
def _(BASELINE, SINGLE, by_size, pd, tests):
    def verdicts() -> pd.DataFrame:
        """Classify each blocked pathway by how much of the gain it removes.

        Reading the two families together: an arm that is significantly below
        `2copies` but no longer distinguishable from `1copy` has had the whole
        repeat effect removed, so that pathway carries it.
        """
        vs_base = tests[tests.family == "vs_2copies"].set_index(["model", "arm"])
        vs_one = tests[tests.family == "vs_1copy"].set_index(["model", "arm"])
        # A pathway can only "carry the gain" for a model that has a gain. Where
        # the repeat effect itself is not established, an arm falling below
        # `2copies` is damage of some other kind, so say that instead.
        has_gain = (
            tests[tests.family == "repeat_effect"]
            .set_index("model")
            .apply(lambda r: bool(r.sig and r.delta > 0), axis=1)
            .to_dict()
        )

        rows = []
        for key, base_row in vs_base.iterrows():
            one_row = vs_one.loc[key]
            hurts = bool(base_row.sig and base_row.delta < 0)
            above_single = bool(one_row.sig and one_row.delta > 0)
            if not has_gain[key[0]]:
                verdict = (
                    "n/a -- no repeat gain to remove for this model"
                    if not hurts
                    else "hurts, but this model has no established repeat gain"
                )
            elif not hurts:
                verdict = f"no detectable role (indistinguishable from {BASELINE})"
            elif above_single:
                verdict = "carries part of the gain"
            else:
                verdict = f"carries the whole gain (falls back to {SINGLE})"
            rows.append(
                {
                    "model": key[0],
                    "blocked pathway": key[1],
                    f"vs {BASELINE} (pp)": base_row.delta * 100,
                    f"vs {SINGLE} (pp)": one_row.delta * 100,
                    "verdict": verdict,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["model", f"vs {BASELINE} (pp)"],
                key=lambda c: c.map(by_size) if c.name == "model" else c,
                kind="stable",
            )
            .reset_index(drop=True)
        )

    summary = verdicts()
    return (summary,)


@app.cell(hide_code=True)
def _(BASELINE, SINGLE, mo, summary):
    mo.md("## Verdict per blocked pathway")
    mo.ui.table(
        summary,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            f"vs {BASELINE} (pp)": "{:+.2f}".format,
            f"vs {SINGLE} (pp)": "{:+.2f}".format,
        },
    )
    return


@app.cell(hide_code=True)
def _(ALPHA, mo, tests):
    _repeat = tests[tests.family == "repeat_effect"]
    _lines = "\n".join(
        f"- **{r.model}**: {r.delta * 100:+.2f} pp "
        f"({r.fixed} fixed / {r.broken} broken, Holm p = {r.p_holm:.2e})"
        f"{' --- significant' if r.sig else ' --- not significant'}"
        for r in _repeat.itertuples()
    )
    _n_sig = int(tests[tests.family == "vs_2copies"].sig.sum())
    _n_tot = int((tests.family == "vs_2copies").sum())

    mo.md(f"""
    ## Read-out

    **Does repeating the prompt help?**

    {_lines}

    **Do the masks matter?** {_n_sig} of {_n_tot} blocked pathways move accuracy
    away from `2copies` at Holm-corrected alpha = {ALPHA}.

    Two cautions on reading the table above. McNemar tests the *direction* of the
    discordant pairs, so a tiny delta can be significant at this n --- read `delta`
    and its interval, not just the star. And the arms are not independent of one
    another: `strictly_past` and `past_or_aligned` differ by a single diagonal of
    the mask, so their deltas move together and Holm (which assumes nothing about
    dependence) is conservative for that pair rather than wrong.
    """)
    return


if __name__ == "__main__":
    app.run()
