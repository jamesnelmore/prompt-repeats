# /// script
# requires-python = ">=3.12"
# ///
"""Recovery-proportion analysis of the prompt-repeat eval set.

Run with:  uv run marimo edit notebooks/recovery_analysis.py
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prompt repeats: recovery proportion

        How much of what chain-of-thought buys can a model get back from seeing the
        question repeated, with reasoning structurally disabled?

        Reads every log in `logs/`. The eval set guarantees one log per
        (task, model), so nothing here de-duplicates or picks a "latest" run --
        point it at a different directory to analyse a different set.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import hashlib
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    from inspect_ai.log import read_eval_log
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    ROOT = Path(__file__).parent.parent
    LOG_DIR = ROOT / "logs"
    CACHE_DIR = ROOT / "data" / "cache"
    return (
        CACHE_DIR,
        LOG_DIR,
        alt,
        hashlib,
        mcnemar,
        multipletests,
        np,
        pd,
        read_eval_log,
    )


@app.cell
def _(CACHE_DIR, LOG_DIR, hashlib, pd, read_eval_log):
    def load_samples() -> pd.DataFrame:
        """One row per (model, arm, sample): the minimum the metric needs.

        Reading the full set takes ~60s, so results are cached under a key derived
        from the log files themselves. Adding a model or re-running an arm changes
        the key and the cache rebuilds -- there is no stale-cache failure mode, and
        no cache to remember to clear.
        """
        paths = sorted(LOG_DIR.glob("*.eval"))
        key = hashlib.sha1(
            repr([(p.name, p.stat().st_size) for p in paths]).encode()
        ).hexdigest()[:12]
        cache = CACHE_DIR / f"samples-{key}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            # Older caches used "repeats" for the same count (1 = once, 2 = twice).
            if "repeats" in df.columns and "copies" not in df.columns:
                df = df.rename(columns={"repeats": "copies"})
            return df

        rows = []
        for path in paths:
            try:
                log = read_eval_log(str(path))
            except Exception:
                # A run writing into this directory leaves half-written archives;
                # skip them rather than failing the whole notebook mid-eval.
                continue
            if log.status != "success":
                continue
            args = log.eval.task_args or {}
            for s in log.samples:
                score = next(iter(s.scores.values()))
                # Prefer prompt_copies; fall back for logs written before the rename.
                n_copies = args.get("prompt_copies", args.get("prompt_repeats", 1))
                rows.append(
                    {
                        "model": log.eval.model.split("/")[-1],
                        "use_cot": bool(args.get("use_cot", False)),
                        "copies": int(n_copies),
                        "sample_id": s.id,
                        "correct": score.value == "C",
                    }
                )

        df = pd.DataFrame(rows)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    samples = load_samples()
    return (samples,)


@app.cell
def _(mo, samples):
    _models = sorted(samples.model.unique())
    mo.md(f"""
    **{len(samples):,} rows** across **{len(_models)} models**: {", ".join(_models)}.
    Copy counts present: {sorted(samples.loc[~samples.use_cot, "copies"].unique())}
    (1 = question once, 2 = twice, …).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The metric

    Raw accuracy conflates two things: how good a model is, and how much extra
    question copies help it. Recovery proportion removes the first by rescaling
    each model onto its own span, so the models are comparable to each other:

    $$R(c) = \frac{\text{acc}_{\text{no-CoT}}(c) - \text{acc}_{\text{no-CoT}}(1)}
    {\text{acc}_{\text{CoT}} - \text{acc}_{\text{no-CoT}}(1)}$$

    where $c$ is the number of times the question appears (1 = once). **0** is
    that model's single-copy no-CoT floor, **1** is its own CoT ceiling.
    $R = 0.25$ means a second (or further) copy bought a quarter of what
    reasoning buys, whatever the model's absolute accuracy.

    Two properties worth keeping in mind. It is a *ratio of differences*, so it
    gets unstable as the CoT gap shrinks -- the `cot_gap_pp` column below is the
    denominator, and a small one means a wide interval. And it is relative by
    construction: a high $R$ on a weak model is a large share of a small prize.

    Intervals are 95% percentile bootstrap over problems (2000 resamples),
    resampling the same problem ids across every arm so the pairing survives.
    """)
    return


@app.cell
def _(np, pd, samples):
    N_BOOT = 2000

    def recovery_frame(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0):
        """Recovery proportion per (model, copy count), with bootstrap CIs."""
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
                continue  # needs both anchors to be on the scale at all

            cot = wide[(True, 1)].to_numpy()
            base = wide[(False, 1)].to_numpy()
            n = len(wide)

            # One index matrix reused across arms: resampling problems, not
            # observations, is what keeps the arms paired inside each replicate.
            idx = rng.integers(0, n, size=(n_boot, n))
            cot_b = cot[idx].mean(axis=1)
            base_b = base[idx].mean(axis=1)
            gap, gap_b = cot.mean() - base.mean(), cot_b - base_b

            for c in sorted(c for use_cot, c in wide.columns if not use_cot):
                arm = wide[(False, c)].to_numpy()
                arm_b = arm[idx].mean(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot = np.where(gap_b > 0, (arm_b - base_b) / gap_b, np.nan)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append(
                    {
                        "model": model,
                        "copies": c,
                        "accuracy": arm.mean(),
                        "base_acc": base.mean(),
                        "cot_acc": cot.mean(),
                        "cot_gap_pp": 100 * gap,
                        "recovery": (arm.mean() - base.mean()) / gap
                        if gap > 0
                        else np.nan,
                        "lo": lo,
                        "hi": hi,
                        "n": n,
                    }
                )

        return pd.DataFrame(rows)

    recovery = recovery_frame(samples)
    return (recovery,)


@app.cell
def _(mo, recovery):
    mo.as_html(
        recovery.assign(
            recovery=recovery.recovery.round(3),
            ci95=[f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(recovery.lo, recovery.hi)],
            accuracy=recovery.accuracy.round(3),
            base_acc=recovery.base_acc.round(3),
            cot_acc=recovery.cot_acc.round(3),
            cot_gap_pp=recovery.cot_gap_pp.round(1),
        )[["model", "copies", "accuracy", "recovery", "ci95",
           "base_acc", "cot_acc", "cot_gap_pp", "n"]]
    )
    return


@app.cell
def _(recovery):
    # Categorical slots in the validated reference order. Both modes pass every
    # hard gate for adjacent forms; three light-mode slots sit under 3:1 contrast,
    # so the relief rule applies -- hence the direct labels and the table above.
    SERIES = [
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    ]
    MUTED, GRID, INK = "#898781", "#e1e0d9", "#0b0b0b"

    MODELS = sorted(recovery.model.unique())
    # Slots are assigned by sorted model name, so a model keeps its hue as others
    # are added. Never cycle: past 8 the honest move is small multiples.
    PALETTE = SERIES[: len(MODELS)]

    def chart_theme(chart):
        return (
            chart.configure_view(stroke=None)
            .configure_axis(
                gridColor=GRID,
                domainColor="#c3c2b7",
                tickColor="#c3c2b7",
                labelColor=MUTED,
                titleColor="#52514e",
                labelFontSize=11,
                titleFontSize=12,
            )
            .configure_legend(labelColor=INK, titleColor="#52514e", symbolStrokeWidth=3)
            .configure_title(color=INK, fontSize=14, anchor="start")
        )

    return MODELS, MUTED, PALETTE, chart_theme


@app.cell
def _(MODELS, MUTED, PALETTE, alt, chart_theme, mo, pd, recovery):
    def recovery_chart(df: pd.DataFrame):
        d = df.copy()

        # Constant-pixel horizontal dodge on the log axis, so overlapping error
        # bars stay readable as models are added.
        rank = {m: i for i, m in enumerate(MODELS)}
        spread = 0 * (len(MODELS) - 1)
        d["x"] = d.copies * (
            1 + spread * (d.model.map(rank) - (len(MODELS) - 1) / 2) / max(len(MODELS) - 1, 1)
        )

        color = alt.Color(
            "model:N",
            scale=alt.Scale(domain=MODELS, range=PALETTE),
            legend=alt.Legend(title="Model", orient="bottom", columns=2),
        )
        x = alt.X(
            "x:Q",
            scale=alt.Scale(type="log", base=2, nice=False, padding=34),
            axis=alt.Axis(
                values=sorted(d.copies.unique()),
                title="Question copies in prompt (1 = once)",
                labelExpr="datum.value",
            ),
        )
        y_scale = alt.Scale(domain=[min(-0.12, d.lo.min() - 0.07), max(1.05, d.hi.max() + 0.05)])
        y = alt.Y("recovery:Q", scale=y_scale, axis=alt.Axis(format="%", title="Recovery of the CoT gap"))

        # label_y sits the baseline caption *below* its rule: every model is
        # pinned to 0 at 1 copy by construction, so the space above it is
        # always occupied.
        anchors = pd.DataFrame(
            {
                "y": [0.0, 1.0],
                "label_y": [-0.055, 1.0],
                "label": ["no-CoT, 1 copy", "CoT ceiling"],
            }
        )
        anchor_rules = (
            alt.Chart(anchors)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1, color=MUTED)
            .encode(y=alt.Y("y:Q", scale=y_scale))
        )
        anchor_labels = (
            alt.Chart(anchors)
            .mark_text(align="left", dy=-6, fontSize=11, color=MUTED)
            .encode(x=alt.value(4), y=alt.Y("label_y:Q", scale=y_scale), text="label:N")
        )

        bars = (
            alt.Chart(d)
            .mark_rule(strokeWidth=1.5, opacity=0.65)
            .encode(x=x, y=alt.Y("lo:Q", scale=y_scale, title=""), y2="hi:Q", color=color)
        )
        line = (
            alt.Chart(d)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=55, filled=True))
            .encode(
                x=x,
                y=y,
                color=color,
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("copies:Q", title="Copies"),
                    alt.Tooltip("recovery:Q", format=".1%", title="Recovery"),
                    alt.Tooltip("lo:Q", format=".1%", title="CI low"),
                    alt.Tooltip("hi:Q", format=".1%", title="CI high"),
                    alt.Tooltip("accuracy:Q", format=".1%", title="Raw accuracy"),
                    alt.Tooltip("cot_gap_pp:Q", format=".1f", title="CoT gap (pp)"),
                ],
            )
        )

        # Direct labels at the right end. Three light-mode slots are sub-3:1 on the
        # surface, so labels are the required relief, not decoration -- nudged apart
        # so they stay legible however the lines land.
        ends = d.sort_values("copies").groupby("model", as_index=False).last()
        ends = ends.sort_values("recovery", ascending=False).reset_index(drop=True)
        label_y, last = [], None
        for v in ends.recovery:
            v = v if last is None else min(v, last - 0.045)
            label_y.append(v)
            last = v
        ends["label_y"] = label_y
        labels = (
            alt.Chart(ends)
            .mark_text(align="left", dx=8, fontSize=11)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(type="log", base=2, nice=False, padding=34)),
                y=alt.Y("label_y:Q", scale=y_scale),
                text="model:N",
                color=alt.Color("model:N", scale=alt.Scale(domain=MODELS, range=PALETTE), legend=None),
            )
        )

        return chart_theme(
            (anchor_rules + anchor_labels + bars + line + labels).properties(
                width=520,
                height=340,
                title="Share of each model's own CoT gap recovered by repeating the question",
            )
        )

    warning = (
        mo.md(
            f"""/// warning | Too many series
            {len(MODELS)} models exceeds the 8-slot categorical ceiling. Hues are
            not cycled, so the chart below is missing models -- switch to small
            multiples before reading it.
            ///"""
        )
        if len(MODELS) > 8
        else mo.md("")
    )
    mo.vstack([warning, recovery_chart(recovery)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Is the intervention doing anything?

    Every arm scores the same problems, so the arms are paired and the items both
    arms get right (or both get wrong) carry no information about the effect.
    **McNemar's test** is the one that matches this design: it conditions on the
    problems the two arms *disagree* on and asks whether the fixes outnumber the
    breaks beyond chance. An unpaired proportion test would throw the pairing away
    and understate the evidence.

    One row per model, testing **1 copy against 2** -- the jump where the effect
    either exists or does not. `p` is the raw two-sided p-value, `p_holm` is
    Holm-corrected across the models, and **`sig` stars anything below 1/1000**.
    Effect sizes are in the chart above; `fixed` and `broken` are the discordant
    counts the test actually runs on.
    """)
    return


@app.cell
def _(mcnemar, multipletests, pd, samples):
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
                continue  # needs both arms

            base, arm = wide[1], wide[2]
            fixed = int((~base & arm).sum())
            broken = int((base & ~arm).sum())
            table = [
                [int((base & arm).sum()), broken],
                [fixed, int((~base & ~arm).sum())],
            ]
            # Exact binomial when the discordant pairs are too few for the
            # chi-square approximation; continuity-corrected chi-square above.
            res = mcnemar(table, exact=(fixed + broken) < 25, correction=True)
            rows.append(
                {
                    "model": model,
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
    def fmt_p(v: float) -> str:
        """Small p-values need an exponent; a fixed 3dp floors them all to 0.000."""
        return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"

    tests_table = mo.ui.table(
        tests,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={"p": fmt_p, "p_holm": fmt_p, "n": "{:,}"},
        # Numbers right-align so magnitudes line up column-wise; labels stay left.
        text_justify_columns={
            "model": "left",
            "fixed": "right",
            "broken": "right",
            "p": "right",
            "p_holm": "right",
            "sig": "center",
            "n": "right",
        },
        label="McNemar: 1 copy vs 2 copies, per model",
    )
    tests_table
    return


@app.cell
def _(mo, tests):
    _starred = int((tests.sig == "*").sum()) if not tests.empty else 0
    _flipped = (
        int(((tests.p < 1e-3) != (tests.p_holm < 1e-3)).sum()) if not tests.empty else 0
    )
    mo.md(f"""
    **{_starred} of {len(tests)}** models are starred (p < 0.001); Holm correction
    changes that for **{_flipped}** of them.

    A star only says two copies beat one. It says nothing about how much (the
    chart above) or about closing the distance to CoT --- and at
    n={tests.n.iloc[0] if not tests.empty else "?"} paired items, roughly a
    one-point shift is enough to earn one.
    """)
    return


if __name__ == "__main__":
    app.run()
