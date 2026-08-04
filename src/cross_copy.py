"""How does the second copy of the question read the first one?

`attention_repeats.py` asks where the *answer* position looks. This asks the
question behind the repeat effect: when the model re-reads the question, do the
copy-2 tokens attend back to copy 1, and with what pattern? If they do, copy 2 is
being processed on top of copy 1's already-transformed representations, which
buys the model extra serial depth over the question without any reasoning tokens.

Both copies are token-for-token identical, so a copy-2 query at index i has an
exactly aligned key at copy-1 index i. That makes the offset j - i meaningful:

    offset  0  -- the same token in the earlier copy ("where was I?")
    offset +1  -- the token *after* the match, i.e. the induction pattern
                  (attend to what followed last time, and predict it again)

Run with:  python src/cross_copy.py --out results/cross_copy
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

from attention_repeats import (
    COPY_COLORS,
    GRID,
    INK,
    INK_MUTED,
    SEQUENTIAL,
    SURFACE,
    build_prompt,
    gsm8k_question,
    label_positions,
    load_model,
    new_figure,
    save,
    style_axes,
)


@dataclass
class CrossCopy:
    """Copy-2-query x copy-1-key attention, per layer and head."""

    # [layer, head, copy-2 index, copy-1 index]
    block: np.ndarray
    # [layer, head]: mean share of a copy-2 token's attention landing on copy 1.
    mass: np.ndarray
    # [layer, head, offset]: attention by key offset (j - i), offsets in `offsets`.
    by_offset: np.ndarray
    offsets: np.ndarray
    tokens: list[str]
    copy1: np.ndarray
    copy2: np.ndarray

    def top_heads(self, count: int) -> list[tuple[int, int]]:
        """(layer, head) pairs with the most copy-2 -> copy-1 attention."""
        order = np.argsort(self.mass, axis=None)[::-1][:count]
        return [(int(l), int(h)) for l, h in zip(*np.unravel_index(order, self.mass.shape))]


def measure_cross_copy(model, question: str, max_offset: int) -> CrossCopy:
    """One forward pass; keep only the copy-2 x copy-1 block of each head."""
    prompt, user_text = build_prompt(model.tokenizer, question, 2)
    tokens, token_strings, segments = label_positions(model.tokenizer, prompt, user_text, question, 2)
    tokens = tokens.to(model.cfg.device)

    copy1 = np.flatnonzero(np.array([s == "copy 1" for s in segments]))
    copy2 = np.flatnonzero(np.array([s == "copy 2" for s in segments]))
    width = min(len(copy1), len(copy2))
    copy1, copy2 = copy1[:width], copy2[:width]

    _, cache = model.run_with_cache(
        tokens, names_filter=lambda name: name.endswith("hook_pattern"), return_type=None
    )
    # The full patterns are [head, query, key] per layer; only the copy-2 rows and
    # copy-1 columns are needed, which is ~64x64 per head instead of 205x205.
    block = torch.stack(
        [
            cache["pattern", layer][0][:, copy2, :][:, :, copy1].float().cpu()
            for layer in tqdm(range(model.cfg.n_layers), desc="layers", unit="layer", leave=False)
        ]
    ).numpy()
    del cache

    offsets = np.arange(-max_offset, max_offset + 1)
    by_offset = np.stack(
        [np.diagonal(block, offset=int(d), axis1=2, axis2=3).mean(axis=2) for d in offsets],
        axis=-1,
    )

    return CrossCopy(
        block=block,
        mass=block.sum(axis=3).mean(axis=2),
        by_offset=by_offset,
        offsets=offsets,
        tokens=token_strings,
        copy1=copy1,
        copy2=copy2,
    )


def plot_mass(data: CrossCopy, out: Path) -> None:
    """Layer x head: which heads read the earlier copy at all."""
    figure, ax = new_figure(
        7,
        6.5,
        "Which heads read the earlier copy?",
        "Share of a copy-2 token's attention landing anywhere in copy 1, averaged over copy-2 tokens",
    )
    image = ax.imshow(
        data.mass,
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=data.mass.max(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(data.mass.shape[1]))
    bar = figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("attention to copy 1", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    save(figure, out / "cross_copy_mass.png")


def plot_head_patterns(data: CrossCopy, out: Path, count: int) -> None:
    """The head patterns themselves: copy-2 queries against copy-1 keys."""
    heads = data.top_heads(count)
    columns = 4
    rows = int(np.ceil(len(heads) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.5 * columns, 2.7 * rows), facecolor=SURFACE
    )
    figure.suptitle(
        "How copy 2 attends to copy 1, head by head",
        x=0.02,
        y=1.0,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=13,
        fontweight="semibold",
    )
    width = data.block.shape[2]
    floor = 1e-3
    norm = LogNorm(vmin=floor, vmax=1.0)

    for index, ax in enumerate(np.atleast_1d(axes).ravel()):
        if index >= len(heads):
            ax.axis("off")
            continue
        layer, head = heads[index]
        image = ax.imshow(
            np.clip(data.block[layer, head], floor, None),
            cmap=SEQUENTIAL,
            norm=norm,
            aspect="equal",
            origin="lower",
            interpolation="nearest",
        )
        # The two diagonals worth naming: same token (offset 0) and induction (+1).
        ax.plot([0, width - 1], [0, width - 1], color=INK_MUTED, linewidth=0.5, alpha=0.35)
        ax.set_title(
            f"L{layer} H{head} · {data.mass[layer, head]:.2f}",
            loc="left",
            color=INK,
            fontsize=9,
            pad=4,
        )
        style_axes(ax)
        ax.tick_params(labelsize=7)
        if index % columns == 0:
            ax.set_ylabel("copy 2 token", fontsize=8)
        if index >= len(heads) - columns:
            ax.set_xlabel("copy 1 token", fontsize=8)

    bar = figure.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.set_label("attention (log)", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    figure.text(
        0.02,
        -0.01,
        "Panel label: layer, head, and that head's total copy-2 -> copy-1 mass. "
        "The faint line is the aligned diagonal (offset 0); an induction head sits one step above it.",
        color=INK_MUTED,
        fontsize=8,
    )
    save(figure, out / "cross_copy_head_patterns.png")


def plot_offsets(data: CrossCopy, out: Path, count: int) -> None:
    """Attention by key offset: aligned (0) versus induction (+1) versus diffuse."""
    figure, ax = new_figure(
        9,
        4.2,
        "Where in copy 1 does a copy-2 token look?",
        "Attention by offset between the copy-2 query and the copy-1 key. "
        "0 is the identical token; +1 is the induction pattern.",
    )
    heads = data.top_heads(count)
    for index, (layer, head) in enumerate(heads):
        ax.plot(
            data.offsets,
            data.by_offset[layer, head],
            color=COPY_COLORS[index],
            linewidth=2,
            label=f"L{layer} H{head}",
        )
    ax.plot(
        data.offsets,
        data.by_offset.mean(axis=(0, 1)),
        color=INK_MUTED,
        linewidth=2,
        linestyle=(0, (4, 2)),
        label="all heads (mean)",
    )
    for offset, name in ((0, "aligned"), (1, "induction")):
        ax.axvline(offset, color=GRID, linewidth=1)
        ax.text(offset, ax.get_ylim()[1], f" {name}", color=INK_MUTED, fontsize=8, va="top")
    ax.set_xlabel("key offset (copy-1 index minus copy-2 index)")
    ax.set_ylabel("mean attention")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK_MUTED,
        ncol=count + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    save(figure, out / "cross_copy_offsets.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/cross_copy")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--top-heads", type=int, default=12, help="Heads shown in the pattern grid.")
    parser.add_argument(
        "--offset-lines", type=int, default=4, help="Heads drawn on the offset profile."
    )
    parser.add_argument(
        "--max-offset", type=int, default=24, help="Offset range on the profile figure."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    question, _ = gsm8k_question(args.question_index)
    model = load_model(args.device, getattr(torch, args.dtype))
    data = measure_cross_copy(model, question, args.max_offset)

    plot_mass(data, out)
    plot_head_patterns(data, out, args.top_heads)
    plot_offsets(data, out, args.offset_lines)
    np.savez(
        out / "cross_copy.npz",
        block=data.block,
        mass=data.mass,
        by_offset=data.by_offset,
        offsets=data.offsets,
    )

    aligned = data.by_offset[:, :, data.offsets == 0].squeeze(-1)
    induction = data.by_offset[:, :, data.offsets == 1].squeeze(-1)
    report = {
        "question_index": args.question_index,
        "copy_tokens": int(data.block.shape[2]),
        "mean_cross_copy_mass": float(data.mass.mean()),
        "max_cross_copy_mass": float(data.mass.max()),
        "top_heads": [
            {
                "layer": layer,
                "head": head,
                "cross_copy_mass": round(float(data.mass[layer, head]), 4),
                "aligned_offset_0": round(float(aligned[layer, head]), 4),
                "induction_offset_1": round(float(induction[layer, head]), 4),
            }
            for layer, head in data.top_heads(args.top_heads)
        ],
    }
    (out / "cross_copy.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
