"""Does copy 2 of a repeated prompt read copy 1? (circuitsvis)

Small self-contained experiment: run the no-CoT prompt from `task.py` at 2 repeats
through a model on 10 GSM8K test questions, then write one HTML page of circuitsvis
attention heads restricted to the two question copies, so the copy-2 -> copy-1
quadrant can be read off directly.

Defaults to Qwen/Qwen3-0.6B so the whole thing runs on a laptop CPU.

Prompts, constrained decoding and scoring all come from `nocot_eval`, so these
are the prompts the eval sends and the completions are read exactly as the eval
read them -- the grammar constrains the sampled tokens only, not the prompt, so
it does not affect the attention measured here.

Heads are ranked by cross-copy mass, averaged over all 10 questions: the share of
a copy-2 token's attention that lands anywhere in copy 1. The viz then shows the
top heads for every question.

Run with:  python src/cross_copy_vis.py --out results/cross_copy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from circuitsvis.attention import attention_heads
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from transformer_lens import HookedTransformer

from nocot_eval import AnswerGrammar, build_grammar, build_prompt, questions, scored

MODEL_NAME = "Qwen/Qwen3-0.6B"
REPEATS = 2


def pick_device() -> str:
    # Not mps: transformer-lens warns it can be silently wrong on this torch, and
    # 0.6B runs on cpu in about a minute per question anyway. Pass --device mps
    # to override.
    return "cuda" if torch.cuda.is_available() else "cpu"


def copy_positions(
    tokenizer, prompt: str, question: str
) -> tuple[torch.Tensor, list[str], np.ndarray, np.ndarray]:
    """Tokenise, and find the token positions of each copy of the question.

    A token is assigned to a copy only if its character span lies entirely inside
    that copy, which keeps the two copies exactly aligned token-for-token (the
    boundary tokens that straddle question and instruction text are dropped).
    """
    encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)

    spans: list[tuple[int, int]] = []
    search_from = 0
    for _ in range(REPEATS):
        start = prompt.index(question, search_from)
        spans.append((start, start + len(question)))
        search_from = start + len(question)

    inside = [
        np.array(
            [
                index
                for index, (start, end) in enumerate(encoding["offset_mapping"])
                if end > start and start >= span_start and end <= span_end
            ]
        )
        for span_start, span_end in spans
    ]
    width = min(len(part) for part in inside)
    copy1, copy2 = inside[0][:width], inside[1][:width]

    tokens = [tokenizer.decode([token_id]) for token_id in encoding["input_ids"]]
    return torch.tensor([encoding["input_ids"]]), tokens, copy1, copy2


def constrained_answer(
    model: HookedTransformer,
    tokens: torch.Tensor,
    grammar: AnswerGrammar,
    max_digits: int,
) -> str:
    """Greedy-decode one ANSWER_SCHEMA object, masking logits to the grammar.

    One full forward pass per step: HookedTransformer exposes no incremental
    cache for a custom decode loop, and at ~5 steps that is cheap enough.
    """
    prompt_length = tokens.shape[1]
    tokens = torch.cat([tokens, grammar.prefix], dim=1)
    with torch.inference_mode():
        for emitted in range(max_digits):
            logits = model(tokens, return_type="logits")[0, -1]
            allowed = grammar.allowed(emitted)
            next_id = allowed[int(logits[allowed].argmax())]
            tokens = torch.cat([tokens, next_id.view(1, 1)], dim=1)
            if int(next_id) == grammar.close:
                break
    return model.tokenizer.decode(tokens[0, prompt_length:])


def measure(
    model: HookedTransformer, question: str, grammar: AnswerGrammar, max_digits: int
) -> dict:
    """One forward pass plus a greedy completion, for a single question."""
    prompt = build_prompt(model.tokenizer, question, REPEATS)
    token_tensor, tokens, copy1, copy2 = copy_positions(
        model.tokenizer, prompt, question
    )
    token_tensor = token_tensor.to(model.cfg.device)

    _, cache = model.run_with_cache(
        token_tensor,
        names_filter=lambda name: name.endswith("hook_pattern"),
        return_type=None,
    )
    # [layer, head, query, key] restricted to the copies: copy-2 queries against
    # copy-1 keys is the block of interest, and the copy-1 rows come along for
    # free as the baseline the viz shows above it.
    keep = np.concatenate([copy1, copy2])
    patterns = torch.stack(
        [
            cache["pattern", layer][0][:, keep, :][:, :, keep].float().cpu()
            for layer in range(model.cfg.n_layers)
        ]
    ).numpy()
    del cache

    width = len(copy1)
    # Share of each copy-2 token's (full-prompt) attention landing in copy 1,
    # averaged over copy-2 tokens.
    mass = patterns[:, :, width:, :width].sum(axis=3).mean(axis=2)

    return {
        "completion": constrained_answer(model, token_tensor, grammar, max_digits),
        "patterns": patterns,  # [layer, head, 2*width, 2*width]
        "mass": mass,  # [layer, head]
        "tokens": [tokens[index] for index in keep],
        "width": width,
    }


def top_heads(mass: np.ndarray, count: int) -> list[tuple[int, int]]:
    order = np.argsort(mass, axis=None)[::-1][:count]
    return [
        (int(layer), int(head))
        for layer, head in zip(*np.unravel_index(order, mass.shape))
    ]


def write_html(
    runs: list[dict],
    mean_mass: np.ndarray,
    heads: list[tuple[int, int]],
    model_name: str,
    out: Path,
) -> Path:
    """One page: for each question, the selected heads over copy 1 + copy 2."""
    sections = []
    for index, run in enumerate(runs):
        width = run["width"]
        attention = np.stack([run["patterns"][layer, head] for layer, head in heads])
        names = [
            f"L{layer}H{head} (mean {mean_mass[layer, head]:.2f} | here {run['mass'][layer, head]:.2f})"
            for layer, head in heads
        ]
        viz = attention_heads(
            attention=attention, tokens=run["tokens"], attention_head_names=names
        )
        sections.append(
            f"<h2>Question {index}</h2>"
            f"<p>Tokens 0&ndash;{width - 1} are copy 1; {width}&ndash;{2 * width - 1} are copy 2 "
            f"(the same tokens again). The lower-left quadrant is copy 2 attending to copy 1. "
            f"Completion: <code>{run['completion'].strip()[:120]}</code></p>"
            f"{viz.__html__()}"
        )

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cross-copy attention &mdash; {model_name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#0b0b0b}}
p{{color:#52514e;line-height:1.5}} h2{{margin-top:2.5rem}}</style></head>
<body>
<h1>Does copy 2 read copy 1?</h1>
<p>{model_name}, no-CoT GSM8K prompt at {REPEATS} repeats, {len(runs)} questions. Attention is
sliced to the two question copies but <em>not</em> renormalised, so a row sums to less than 1:
the rest of that token's attention goes to the instruction text and chat scaffold.
Heads shown are the {len(heads)} with the highest copy-2&nbsp;&rarr;&nbsp;copy-1 mass averaged over
all questions; each head's mean mass and its mass on the question shown are in the head label.</p>
{"".join(sections)}
</body></html>
"""
    path = out / "cross_copy.html"
    path.write_text(page)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/cross_copy")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Any transformer-lens model. The point of the local Qwen default is "
        "to rehearse a gemma run on a GPU box without editing this file.",
    )
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument(
        "--top-heads", type=int, default=8, help="Heads rendered per question."
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=8,
        help="Cap on decoded digit tokens before the object is closed anyway.",
    )
    parser.add_argument("--device", default=pick_device())
    parser.add_argument(
        "--dtype", default="float32", choices=["bfloat16", "float16", "float32"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv(usecwd=True))  # HF_TOKEN, if the cache is cold.

    print(f"Loading {args.model} on {args.device}...")
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=getattr(torch, args.dtype)
    )

    print(f"Building grammar on {args.device}")
    grammar = build_grammar(model.tokenizer, args.device)

    print("starting eval")
    runs, correct = [], []
    for index, (question, reference) in tqdm(enumerate(questions(args.questions))):
        run = measure(model, question, grammar, args.max_digits)
        runs.append(run)
        correct.append(scored(run["completion"], reference))
        print(
            f"[{index}] mass max {run['mass'].max():.3f}  "
            f"{run['completion']!r} vs {reference}  correct={correct[-1]}"
        )

    mean_mass = np.mean([run["mass"] for run in runs], axis=0)
    heads = top_heads(mean_mass, args.top_heads)
    path = write_html(runs, mean_mass, heads, args.model, out)

    report = {
        "model": args.model,
        "repeats": REPEATS,
        "questions": args.questions,
        "accuracy": sum(correct) / len(correct),
        "mean_cross_copy_mass": float(mean_mass.mean()),
        "top_heads": [
            {
                "layer": layer,
                "head": head,
                "mean_cross_copy_mass": round(float(mean_mass[layer, head]), 4),
            }
            for layer, head in heads
        ],
    }
    (out / "cross_copy.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
