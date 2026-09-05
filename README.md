# Copying a prompt once is useful for nonreasoning models

Static demo: https://jamesnelmore.github.io/prompt-repeats/

To run locally: `uv sync --frozen` then `uv run marimo edit notebooks/demo.py`.

## Headline Results (tenative)
- LLM nonreasoning performance on GSM8k improves when a model is shown 2 copies of the prompt. [Past work](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens) found a similar result for frontier models, this repo shows it replicates to open weight models as well, specifically the gemma 3/4 family above 4 billion parameters
- Attention knockout experments show that blocking copy 2 of the prompt from attending to copy 1 degrades performance to roughly single copy levels. 
- Current hypothesis being tested is that the model uses the first copy to let the second copy "look into the future" and get around attention masking to gain information 

## More information

[Past work](https://blog.redwoodresearch.org/p/recent-llms-can-use-filler-tokens) by Ryan Greenblatt found that non-reasoning performance for frontier models is improved by copying a prompt multiple times. Normally, an LLM input might look something like this:

```
User:
	[Instructions]

	[Question]

Assistant:
	ANSWER:
```

He found that this:

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
