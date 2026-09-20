# From a character RNN to a minimal Transformer

The preceding `min-char-rnn` project reads text from left to right and updates a
hidden state after every input character. Its prediction at a position depends
on the current character and the information retained in that recurrent state.

This project keeps next-character prediction on Tiny Shakespeare, but replaces
the recurrent information path with **causal self-attention**. Every position
uses learned queries and keys to weight the value representations of permitted
preceding positions. The causal mask prevents the model from accessing future
input characters when predicting the next one.

The original Transformer paper (Vaswani et al., 2017) used an encoder-decoder
architecture. This repository implements one **decoder-style** block, one
attention head and character tokens. GPT-2 (2019) and later large language
models use decoder-only Transformer architectures at a far larger scale. This
project is intended to expose one architectural idea through a small model
whose mathematics, training loop and results can be inspected in R.

The project does not introduce a new language-model architecture. Nor does the
presence of an attention matrix establish that any single attention weight is
a complete explanation of a prediction. Evidence of correctness comes first
from mathematical tests and then from recorded real-text experiments.

References: [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762),
[Karpathy (2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/),
[GPT-2 (2019)](https://openai.com/research/better-language-models).
