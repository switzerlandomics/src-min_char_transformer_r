# Input texts

`input.txt` is a small synthetic, deliberately repetitive UTF-8 corpus bundled
for offline smoke tests. It is not evidence of performance on natural language.

The normal experiment uses `tiny_shakespeare.txt`. If it is missing, the runner
retrieves Karpathy's Tiny Shakespeare text from:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The downloaded text is ignored by Git. The runner records an MD5 checksum of
the exact input bytes and uses contiguous training, validation and test splits.
For a different UTF-8 corpus, pass `--input=PATH`. Unseen held-out characters
cause an explicit error rather than being discarded or mapped silently.
