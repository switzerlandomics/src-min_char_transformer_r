# Comparing recurrence and causal self-attention

Both projects solve the same conceptual task: predict each next character in
an input sequence. The earlier vanilla RNN updates one recurrent hidden state
in sequence, passing information to later positions. This project combines
value representations from permitted positions through learned causal
attention weights. Neither model may read its future target characters.

The architectures differ in their access to history. The existing RNN carries
hidden state across adjacent training windows, even though its backward-pass
calculation is truncated within each window. This Transformer receives only
its explicit context window and recomputes representations within it. Giving
both models 32-character windows does not equalise that distinction.

The historical RNN experiment used a 90/10 split, an AdaGrad optimiser and a
100-unit hidden state. The reference Transformer proposal uses 80/10/10,
Adam and a one-block 32-wide network. Historic loss curves cannot therefore
serve as a controlled head-to-head architectural benchmark.

For a comparison, use **identical corpus bytes, held-out character positions,
next-character targets and evaluation context rules** where feasible. Report
parameter counts, characters processed, CPU time, validation and final test
losses, and generated examples with the same prompt and sampling settings.
Explain the remaining differences in context and optimisation explicitly.
A result from these small configurations is evidence about these particular
experiments, not a general verdict about all RNNs and Transformers.
