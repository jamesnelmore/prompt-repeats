# What I need to start writing the paper (36h freeze)

## Must run
- **Reverse arm (local GPU, white-box):** future-only mask — copy2 may attend only to *future* (± aligned) tokens in copy1; past tokens blocked. (Past-only is already done and kills the gain; this is the discriminating test for “look into the future.”)
- **CoT × repeats (OpenRouter API, no GPU):** Inspect arms for CoT with 2/4/8 copies (1-copy CoT already logged). Expect little or no lift vs CoT×1.
- **Stats freeze:** type-I–corrected paired tests (McNemar + Holm) for 1→2 no-CoT and all ablation contrasts; numbers from `ablation_stats` / `early_results` + new arms.

## Cut for freeze
- Length-matching / pad-with-other-question control — cite 4/8-copy plateau as suggestive; list as limitation.

## Writing (parallel)
- Related work: deep research on Echo encoding + Redwood Think Fast + Greenblatt blog; keep short (5-page body).
