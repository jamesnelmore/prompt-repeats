# Testing Usefullness of Repeating Prompts in Open Weight LLMs and examining mechanistic explantions of why

Preliminary results (see `notebooks/blackbox_analysis.py for more info`)
- Sending a prompt twice shows statistically significent improvement for GSM8k on all tested models with more than 4b parameters
- Repeating more than twice doesn't appear to add anything.
