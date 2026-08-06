# Testing Usefullness of Repeating Prompts in Open Weight LLMs and examining mechanistic explantions of why

Preliminary results (see `notebooks/blackbox_analysis.py for more info`)
- Sending a prompt twice shows statistically significent improvement for GSM8k on all tested models with more than 4b parameters
- Repeating more than twice doesn't appear to add anything.

## Cross-copy pathways ablatio (Agent summary)

Does the 2-repeat gain need copy 2 to read copy 1, or the answer to read copy 1 directly? On no-CoT GSM8K (full test, n=1319), we zero each path alone (`src/cross_copy_ablation.py`; results in `results/cross_copy_ablation_pathways_*`). Accuracies with Wilson 95% CIs:

| condition | gemma-3-12b-it | gemma-3-27b-it |
|---|---:|---:|
| repeat1 | 21.8% [19.7, 24.1] | 31.2% [28.8, 33.8] |
| repeat2 | 25.6% [23.3, 28.0] | 37.5% [35.0, 40.2] |
| mask_copy2 (c1→answer only) | 20.1% [18.0, 22.3] | 30.3% [27.8, 32.8] |
| mask_answer (c1→c2 only) | 20.9% [18.7, 23.1] | 35.8% [33.2, 38.4] |

On 27b, `mask_copy2` clearly drops accuracy (to ~repeat1); `mask_answer` is not clearly below `repeat2` (CIs overlap). So what matters is a **c1 → c2 → answer** chain — answer attending to c1 directly is not important.
