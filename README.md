# Testing Usefullness of Repeating Prompts in Open Weight LLMs and examining mechanistic explantions of why

- Desrisk step one: use gsm8k + inspect to test every frontier open weight LLM on OpenRouter to see which ones can benefit from prompt repeats.
- Plot attention values on each token for each repeat and compare between models that can do this and models that can't


## Step 1: GSM8K results on OpenRouter
- Prompt repeats improve performance on Gemma family at n=500 questions

```
openrouter/google/gemma-3-12b-it  (CoT=0.938, noCoT@1=0.216)
repeats    acc  recovery  p(McNemar)
      1  0.216  baseline           -
      2  0.226      0.01      0.5966
      4  0.254      0.05      0.0319
      8  0.268      0.07      0.0054

openrouter/google/gemma-3-27b-it  (CoT=0.950, noCoT@1=0.334)
repeats    acc  recovery  p(McNemar)
      1  0.334  baseline           -
      2  0.390      0.09      0.0030
      4  0.400      0.11      0.0010
      8  0.396      0.10      0.0015

openrouter/google/gemma-3-4b-it  (CoT=0.882, noCoT@1=0.104)
repeats    acc  recovery  p(McNemar)
      1  0.104  baseline           -
      2  0.138      0.04      0.0213
      4  0.126      0.03      0.1352
      8  0.122      0.02      0.2806
```

- [ ] Test other families
- [ ] Test with full benchmark
- [ ] Apply corrections to account for multiple significence tests

## White Box Analysis
-[ ] Compare activations and attention heads in Gemma 4b and Gemma 27b
