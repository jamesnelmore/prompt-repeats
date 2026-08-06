"""Named intervention presets: which conditions to run, and how to describe them."""

from __future__ import annotations

from dataclasses import dataclass

from ablation.masks import BASELINE, Condition


@dataclass(frozen=True)
class Intervention:
    """One ablation study: baselines plus the masked arms that answer its question."""

    name: str
    description: str
    conditions: tuple[Condition, ...]
    verify_scopes: tuple[str, ...]
    notes: tuple[str, ...]
    default_out: str


PATHWAYS = Intervention(
    name="pathways",
    description=(
        "Sever each copy-1→answer path alone: mask_copy2 = c1→answer only "
        "(copy 2 blind to copy 1); mask_answer = c1→c2 only (answer blind to copy 1)."
    ),
    conditions=(
        Condition("1copy", 1, None),
        Condition(BASELINE, 2, None),
        Condition("2copies_mask_copy2", 2, "copy2"),
        Condition("2copies_mask_answer", 2, "answer"),
    ),
    verify_scopes=("copy2", "answer"),
    notes=(
        "mask_copy2: copy2 cannot attend to copy1; answer still can (c1 → answer only)",
        "mask_answer: answer (after copy2) cannot attend to copy1; copy2 still can (c1 → c2 only)",
    ),
    default_out="results/cross_copy_ablation_pathways",
)

CAUSAL_CROSS_COPY = Intervention(
    name="causal_cross_copy",
    description=(
        "Keep two question copies but stop copy 2 from reading aligned/future keys in "
        "copy 1: copy2[i] ↛ copy1[j≥i]. Full mask_copy2 is the no-cross-copy floor."
    ),
    conditions=(
        Condition("1copy", 1, None),
        Condition(BASELINE, 2, None),
        Condition("2copies_mask_copy2", 2, "copy2"),
        Condition("2copies_causal_copy2", 2, "causal_copy2"),
    ),
    verify_scopes=("copy2", "causal_copy2"),
    notes=(
        "mask_copy2: copy2 cannot attend to any of copy1",
        "causal_copy2: copy2[i] may attend copy1[j] only for j < i "
        "(no aligned/future keys in copy 1)",
    ),
    default_out="results/cross_copy_ablation_causal",
)

INTERVENTIONS: dict[str, Intervention] = {
    PATHWAYS.name: PATHWAYS,
    CAUSAL_CROSS_COPY.name: CAUSAL_CROSS_COPY,
}
