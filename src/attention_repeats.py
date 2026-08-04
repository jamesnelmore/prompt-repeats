"""Where does gemma-3-4b-it actually look when the question is repeated?

Runs the no-CoT prompt from `task.py` at several repeat counts through
transformer-lens on one GSM8K test question, and writes figures showing how the
attention at the final prompt position (the token that produces the answer) is
split across the copies of the question.

The prompt text is built with the same template and repeat logic the eval uses,
then wrapped in Gemma's own chat template -- OpenRouter applies that template
server-side, so this reproduces what the served model saw. The eval additionally
constrains decoding with a JSON response schema; that changes the sampled tokens,
not the prompt, so it does not affect the attention measured here.

Requires access to the gated `google/gemma-3-4b-it` repo: accept the licence on
huggingface.co, then either run `hf auth login` or put HF_TOKEN=... in .env
(loaded here the same way src/gemma_eval.py loads OpenRouter credentials).

Run with:  python src/attention_repeats.py --out results/attention
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: this is meant to be run on a remote GPU box.

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from tqdm import tqdm
from transformer_lens import HookedTransformer

from task import DATASET_PATH, GSM8K_DATASET_REVISION, NO_COT_PROMPT_TEMPLATE, repeated_template

MODEL_NAME = "google/gemma-3-4b-it"

# Palette: categorical slots 1-8 for the question copies, neutrals for the
# non-question text. Sequential blue ramp for the per-head heatmaps.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2db"
COPY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INSTRUCTION_COLOR = "#a9a89f"
SCAFFOLD_COLOR = "#d6d5cd"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
# Diverging: blue <-> orange around a neutral gray midpoint. The poles reuse the
# copy 1 / copy 2 series hues instead of the generic blue/red pair, so a reader
# carries one colour meaning across the whole figure set.
DIVERGING = LinearSegmentedColormap.from_list(
    "copy_diff",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f7cbc4", "#f0a184", "#eb6834", "#b8410f"],
)

INSTRUCTIONS = "instructions"
SCAFFOLD = "chat scaffold"


@dataclass
class RepeatRun:
    """Everything measured for one repeat count."""

    repeats: int
    tokens: list[str]
    segments: list[str]  # One label per token position.
    segment_names: list[str]  # Plot order: copy 1..n, instructions, scaffold.
    # Attention from the final prompt position, [layer, head, segment].
    final_attention: np.ndarray
    # Attention from the final prompt position, [key position].
    final_profile: np.ndarray
    # Attention from the final prompt position per layer, [layer, key position].
    layer_profile: np.ndarray
    completion: str

    def copy_positions(self, copy: int) -> np.ndarray:
        """Token positions belonging to `copy` (1-indexed)."""
        return np.flatnonzero(np.array([label == f"copy {copy}" for label in self.segments]))


def gsm8k_question(index: int) -> tuple[str, str]:
    """The question and reference answer at `index` of the pinned GSM8K test split."""
    dataset = load_dataset(
        DATASET_PATH, "main", split="test", revision=GSM8K_DATASET_REVISION
    )
    record = dataset[index]
    return record["question"], record["answer"].split("####")[-1].strip()


def build_prompt(tokenizer, question: str, repeats: int) -> tuple[str, str]:
    """The chat-templated prompt, and the user turn's text inside it."""
    user_text = repeated_template(NO_COT_PROMPT_TEMPLATE, repeats).format(prompt=question)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, user_text


def label_positions(
    tokenizer, prompt: str, user_text: str, question: str, repeats: int
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Tokenise `prompt` and label every token position with its segment.

    Segments are the individual question copies, the surrounding instruction
    text, and the chat template's own turn markers. A token that straddles a
    boundary is assigned to whichever segment covers most of its characters.
    """
    # add_special_tokens=False: Gemma's chat template already emits <bos>.
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    user_start = prompt.index(user_text)
    user_end = user_start + len(user_text)

    copy_spans: list[tuple[int, int]] = []
    search_from = user_start
    for _ in range(repeats):
        start = prompt.index(question, search_from)
        copy_spans.append((start, start + len(question)))
        search_from = start + len(question)

    def label_for(start: int, end: int) -> str:
        if end <= start:  # Zero-width offsets: special tokens.
            return SCAFFOLD
        overlaps = {
            f"copy {i + 1}": max(0, min(end, span_end) - max(start, span_start))
            for i, (span_start, span_end) in enumerate(copy_spans)
        }
        overlaps[INSTRUCTIONS] = max(0, min(end, user_end) - max(start, user_start)) - sum(
            overlaps.values()
        )
        overlaps[SCAFFOLD] = (end - start) - sum(overlaps.values())
        return max(overlaps, key=lambda name: overlaps[name])

    segments = [label_for(start, end) for start, end in offsets]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    token_tensor = torch.tensor([encoding["input_ids"]])
    return token_tensor, tokens, segments


def load_model(device: str, dtype: torch.dtype) -> HookedTransformer:
    """Load gemma-3-4b-it into transformer-lens (needs HF auth; see module docstring)."""
    return HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=dtype,
    )


def measure(model: HookedTransformer, question: str, repeats: int, max_new_tokens: int) -> RepeatRun:
    """Cache attention for one repeat count and summarise it by segment."""
    prompt, user_text = build_prompt(model.tokenizer, question, repeats)
    tokens, token_strings, segments = label_positions(
        model.tokenizer, prompt, user_text, question, repeats
    )
    tokens = tokens.to(model.cfg.device)

    _, cache = model.run_with_cache(
        tokens, names_filter=lambda name: name.endswith("hook_pattern"), return_type=None
    )
    # [layer, head, key position], attention from the last prompt token -- the
    # position whose residual stream produces the first answer token.
    final = torch.stack(
        [cache["pattern", layer][0, :, -1, :] for layer in range(model.cfg.n_layers)]
    ).float().cpu()
    del cache

    segment_names = [f"copy {i + 1}" for i in range(repeats)] + [INSTRUCTIONS, SCAFFOLD]
    membership = torch.zeros(len(segments), len(segment_names))
    for position, label in enumerate(segments):
        membership[position, segment_names.index(label)] = 1.0

    with torch.inference_mode():
        completion_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_at_eos=True,
            verbose=False,
        )
    completion = model.tokenizer.decode(completion_tokens[0, tokens.shape[1] :])

    return RepeatRun(
        repeats=repeats,
        tokens=token_strings,
        segments=segments,
        segment_names=segment_names,
        final_attention=(final @ membership).numpy(),
        final_profile=final.mean(dim=(0, 1)).numpy(),
        layer_profile=final.mean(dim=1).numpy(),
        completion=completion,
    )


def segment_color(name: str) -> str:
    if name == INSTRUCTIONS:
        return INSTRUCTION_COLOR
    if name == SCAFFOLD:
        return SCAFFOLD_COLOR
    return COPY_COLORS[int(name.split()[1]) - 1]


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.xaxis.label.set_color(INK_MUTED)


def new_figure(width: float, height: float, title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    figure, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    # y=1, va="bottom" keeps the title clear of the subtitle on short figures;
    # bbox_inches="tight" at save time pulls it back inside the image.
    figure.suptitle(
        title, x=0.02, y=1.0, ha="left", va="bottom", color=INK, fontsize=13, fontweight="semibold"
    )
    ax.set_title(subtitle, loc="left", color=INK_MUTED, fontsize=9, pad=10)
    style_axes(ax)
    return figure, ax


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    tqdm.write(f"wrote {path}")  # tqdm.write so the bars stay intact.


def plot_by_layer(run: RepeatRun, out: Path) -> None:
    """Stacked bars: where each layer's attention goes, averaged over heads."""
    shares = run.final_attention.mean(axis=1)  # [layer, segment]
    layers = np.arange(shares.shape[0])

    figure, ax = new_figure(
        9,
        4.2,
        f"Attention from the answer position, by layer ({run.repeats} copies)",
        "Share of the final token's attention, averaged over the 8 heads in each layer",
    )
    bottom = np.zeros(len(layers))
    for index, name in enumerate(run.segment_names):
        ax.bar(
            layers,
            shares[:, index],
            bottom=bottom,
            width=0.82,
            color=segment_color(name),
            edgecolor=SURFACE,
            linewidth=1.2,
            label=name,
        )
        bottom += shares[:, index]
    ax.set_xlabel("layer")
    ax.set_ylabel("attention share")
    ax.set_xlim(-0.8, len(layers) - 0.2)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncol=len(run.segment_names),
        fontsize=8,
        labelcolor=INK_MUTED,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    save(figure, out / f"attention_by_layer_repeat{run.repeats}.png")


def plot_heads(run: RepeatRun, out: Path) -> None:
    """Per-head heatmap of attention to each copy: which heads read which copy."""
    copies = [name for name in run.segment_names if name.startswith("copy")]
    values = run.final_attention[:, :, : len(copies)]
    vmax = float(values.max())

    figure, axes = plt.subplots(
        1,
        len(copies),
        figsize=(2.1 * len(copies) + 1.6, 6),
        facecolor=SURFACE,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    figure.suptitle(
        f"Which heads read which copy ({run.repeats} copies)",
        x=0.02,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="semibold",
    )
    for index, (ax, name) in enumerate(zip(axes, copies)):
        image = ax.imshow(
            values[:, :, index],
            cmap=SEQUENTIAL,
            vmin=0,
            vmax=vmax,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(name, loc="left", color=INK, fontsize=10, pad=6)
        ax.set_xlabel("head")
        ax.set_xticks(range(values.shape[1]))
        style_axes(ax)
    axes[0].set_ylabel("layer")
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("attention share from the answer position", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    save(figure, out / f"attention_by_head_repeat{run.repeats}.png")


def plot_profile(run: RepeatRun, out: Path) -> None:
    """Attention against key position, with the copies shaded."""
    profile = run.final_profile
    positions = np.arange(len(profile))

    figure, ax = new_figure(
        10,
        3.6,
        f"Attention from the answer position, token by token ({run.repeats} copies)",
        "Averaged over every layer and head; shaded bands are the repeated question. "
        "Log scale: the <bos> sink at position 0 is an order of magnitude above everything else.",
    )
    ax.set_yscale("log")
    floor = max(profile[profile > 0].min(), 1e-5)
    ceiling = profile.max()
    for name in run.segment_names:
        if not name.startswith("copy"):
            continue
        edges = np.flatnonzero(np.array([label == name for label in run.segments]))
        ax.axvspan(
            edges[0] - 0.5,
            edges[-1] + 0.5,
            color=segment_color(name),
            alpha=0.12,
            linewidth=0,
        )
        ax.text(
            (edges[0] + edges[-1]) / 2,
            ceiling * 1.6,
            name,
            ha="center",
            color=segment_color(name),
            fontsize=9,
        )
    ax.plot(positions, np.clip(profile, floor, None), color=INK_MUTED, linewidth=1.6)
    ax.set_xlabel("token position")
    ax.set_ylabel("mean attention (log)")
    ax.set_xlim(-1, len(profile))
    ax.set_ylim(floor * 0.8, ceiling * 3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    save(figure, out / f"attention_profile_repeat{run.repeats}.png")


def token_labels(run: RepeatRun, positions: np.ndarray) -> list[str]:
    """Readable strings for the tokens at `positions` (Gemma marks word starts with U+2581)."""
    return [run.tokens[p].replace("▁", " ").replace("\n", "\\n") for p in positions]


def plot_layer_token_map(run: RepeatRun, out: Path) -> None:
    """The layer sweep: every layer's attention to every token, copies marked."""
    values = run.layer_profile
    # A position under 1e-3 contributes nothing at this sequence length; clipping
    # there spends the colour range on the structure instead of on the noise.
    floor = 1e-3

    figure, ax = new_figure(
        13,
        5.4,
        f"Attention from the answer position: every layer, every token ({run.repeats} copies)",
        "Head-averaged attention from the final prompt token. Log colour scale; "
        f"values below {floor:g} are clipped.",
    )
    image = ax.imshow(
        np.clip(values, floor, None),
        cmap=SEQUENTIAL,
        norm=LogNorm(vmin=floor, vmax=values.max()),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    for copy in range(1, run.repeats + 1):
        positions = run.copy_positions(copy)
        color = segment_color(f"copy {copy}")
        for edge in (positions[0] - 0.5, positions[-1] + 0.5):
            ax.axvline(edge, color=color, linewidth=1.2, alpha=0.9)
        ax.text(
            positions.mean(),
            values.shape[0] - 1.5,
            f"copy {copy}",
            ha="center",
            va="center",
            color=color,
            fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    bar = figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("attention share (log)", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    save(figure, out / f"attention_layer_token_repeat{run.repeats}.png")


def plot_copy_alignment(run: RepeatRun, out: Path) -> None:
    """Copy 2 minus copy 1, token-aligned: which copy each layer prefers, word by word."""
    first, second = run.copy_positions(1), run.copy_positions(2)
    width = min(len(first), len(second))  # Identical text, so these line up 1:1.
    difference = run.layer_profile[:, second[:width]] - run.layer_profile[:, first[:width]]
    limit = float(np.percentile(np.abs(difference), 99)) or float(np.abs(difference).max())

    figure, ax = new_figure(
        13,
        5.4,
        "Which copy does each layer read? (copy 2 minus copy 1, token by token)",
        "Head-averaged attention from the answer position; the copies are token-for-token identical, "
        f"so positions line up 1:1. Blue = the layer reads copy 1, orange = copy 2. Clipped at +/-{limit:.4f}.",
    )
    image = ax.imshow(
        difference,
        cmap=DIVERGING,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    step = max(1, width // 24)
    ax.set_xticks(range(0, width, step))
    ax.set_xticklabels(token_labels(run, first[:width:step]), rotation=90, fontsize=6)
    ax.set_xlabel("token of the question (both copies)")
    ax.set_ylabel("layer")
    bar = figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("attention difference (copy 2 - copy 1)", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    save(figure, out / f"attention_copy_alignment_repeat{run.repeats}.png")


def plot_copy_by_layer(run: RepeatRun, out: Path) -> None:
    """Per-layer totals for each copy, as lines rather than stacked shares."""
    figure, ax = new_figure(
        9,
        4,
        f"Attention to each copy, layer by layer ({run.repeats} copies)",
        "Total head-averaged attention from the answer position onto each copy of the question",
    )
    layers = np.arange(run.layer_profile.shape[0])
    for copy in range(1, run.repeats + 1):
        totals = run.layer_profile[:, run.copy_positions(copy)].sum(axis=1)
        color = segment_color(f"copy {copy}")
        ax.plot(layers, totals, color=color, linewidth=2, label=f"copy {copy}")
        ax.annotate(
            f"copy {copy}",
            (layers[-1], totals[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("attention share")
    ax.set_xlim(-0.5, len(layers) + 3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=run.repeats, loc="upper left")
    save(figure, out / f"attention_copy_by_layer_repeat{run.repeats}.png")


def plot_copy_totals(runs: list[RepeatRun], out: Path) -> None:
    """Grouped bars: total attention each copy gets, across repeat counts."""
    figure, ax = new_figure(
        7.5,
        4,
        "How much attention each copy of the question gets",
        "Total attention share from the answer position, averaged over every layer and head. "
        "Copies are token-for-token identical, so the shares are directly comparable.",
    )
    width = 0.8 / max(run.repeats for run in runs)
    for group, run in enumerate(runs):
        copies = [name for name in run.segment_names if name.startswith("copy")]
        totals = run.final_attention.mean(axis=(0, 1))[: len(copies)]
        offsets = (np.arange(len(copies)) - (len(copies) - 1) / 2) * width
        for index, name in enumerate(copies):
            bar = ax.bar(
                group + offsets[index],
                totals[index],
                width=width * 0.86,
                color=COPY_COLORS[index],
                edgecolor=SURFACE,
                linewidth=1.2,
                label=name if group == len(runs) - 1 else None,
            )
            ax.bar_label(bar, fmt="%.3f", color=INK_MUTED, fontsize=7, padding=2)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{run.repeats} copies" if run.repeats > 1 else "1 copy" for run in runs])
    ax.set_ylabel("attention share")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_MUTED,
        ncol=max(run.repeats for run in runs),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    save(figure, out / "attention_by_copy.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/attention", help="Directory for the figures.")
    parser.add_argument(
        "--repeats",
        type=int,
        nargs="+",
        default=[2],
        help="Repeat counts to run, matching the eval's no-CoT arms.",
    )
    parser.add_argument(
        "--question-index", type=int, default=0, help="Index into the GSM8K test split."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="bfloat16 keeps the 4b model comfortably inside a 24GB card.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Greedy continuation recorded alongside the figures as a sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(find_dotenv(usecwd=True))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    question, reference = gsm8k_question(args.question_index)
    model = load_model(args.device, getattr(torch, args.dtype))

    runs = []
    for repeats in tqdm(sorted(args.repeats), desc="repeat counts", unit="arm"):
        run = measure(model, question, repeats, args.max_new_tokens)
        runs.append(run)
        figures = [plot_by_layer, plot_heads, plot_profile, plot_layer_token_map, plot_copy_by_layer]
        if run.repeats >= 2:
            figures.append(plot_copy_alignment)
        for plot in tqdm(figures, desc=f"figures ({repeats} copies)", unit="fig", leave=False):
            plot(run, out)
        # The measured arrays, so the plots can be reworked without a GPU pass.
        np.savez(
            out / f"attention_repeat{run.repeats}.npz",
            layer_profile=run.layer_profile,
            final_attention=run.final_attention,
            segments=np.array(run.segments),
            tokens=np.array(run.tokens),
            segment_names=np.array(run.segment_names),
        )
    plot_copy_totals(runs, out)

    summary = {
        "model": MODEL_NAME,
        "question_index": args.question_index,
        "question": question,
        "reference_answer": reference,
        "runs": [
            {
                "repeats": run.repeats,
                "n_tokens": len(run.tokens),
                "completion": run.completion,
                "attention_share": dict(
                    zip(run.segment_names, run.final_attention.mean(axis=(0, 1)).round(4).tolist())
                ),
            }
            for run in runs
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["runs"], indent=2))


if __name__ == "__main__":
    main()
