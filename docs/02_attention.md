# Attention, tensor shapes and manual gradients

The implementation uses one head and one Transformer block. For a window of
`L` characters, vocabulary size `V`, model width `d` and feed-forward width
`f`, rows represent positions and columns represent features. The model's
input embedding matrix is `L × d`; logits and next-character probabilities
are `L × V`.

## Forward pass

`X = token_embedding[indices, ] + position_embedding[1:L, ]`. Learned absolute
positions restart at the beginning of each supplied context window. Define
`N1 = LN1(X)` and let the four attention projections have shape `d × d`:

```text
Q = N1 Wq                   (L × d)
K = N1 Wk                   (L × d)
V = N1 Wv                   (L × d)
S = Q Kᵀ / sqrt(d)           (L × L)
S[i,j] = -Inf               whenever j > i
A = row_softmax(S)           (L × L)
Z = A V                     (L × d)
R = X + Z Wo                (L × d)
N2 = LN2(R)                 (L × d)
H = ReLU(N2 W1 + b1)        (L × f)
U = R + H W2 + b2           (L × d)
F = LN3(U)                 (L × d)
logits = F Wout + bout      (L × V)
```

Each softmax row sums to one, and its masked upper-triangular entries are
exactly zero. The position `i` sees input indices `1:i` when predicting the
next-character target at `i+1`. It cannot access the target through the input
or attention mechanism.

All three layer norms normalise across the *features of one position*, not
across time. For a row `x` with mean `μ`, population variance `σ²`, gain `γ`
and bias `β`, the implementation calculates
`z = (x - μ) / sqrt(σ² + ε)` and `y = γ * z + β`, with `ε = 1e-5` by default.
Residual connections are added **before** the respective next normalisation;
this is the precise pre-normalised block implemented, rather than a mixture
of pre-norm and post-norm variants.

The total number of parameters without output-weight tying is
`2*V*d + L*d + 4*d*d + 2*d*f + f + 7*d + V`.
At `V=65, L=32, d=32, f=64` this gives **13,729 parameters**. The two
embedding tables and output projection are independent trainable matrices.

## Loss and backward pass

For each position, the target is the *next* character, and the average loss is
`mean(log_sum_exp(logits[i, ]) - logits[i, target[i]])`. A row-wise
maximum is subtracted before exponentiation for numerical stability. With
`P = row_softmax(logits)`, the derivative for the **mean** loss is
`dlogits = (P - one_hot(targets)) / L`.

For a row-wise softmax `A = softmax(S)` and incoming gradient `dA`, its
Jacobian-vector product is
`dS = A * (dA - rowSums(dA * A))`, interpreted row by row. Masked entries
remain zero. For `Z = A V`, `dA = dZ Vᵀ` and `dV = Aᵀ dZ`.
For `S = Q Kᵀ / sqrt(d)`,
`dQ = dS K / sqrt(d)` and `dK = dSᵀ Q / sqrt(d)`.
Gradients for `Wq`, `Wk`, `Wv` follow by multiplying each projection's input
transpose by the corresponding output gradient; the upstream gradient to the
normalised input is the sum of contributions from all three projections.

For layer norm, let `g = dY * γ` and `z` be the normalised input. For each
position independently, the derivative is
`dX = (g - mean(g) - z * mean(g*z)) / sqrt(σ² + ε)`.
Gain and bias derivatives are column sums of `dY*z` and `dY`, respectively.
A residual addition sends the same upstream gradient down **both** paths;
the contributions must be summed. An embedding index can appear repeatedly,
so gradients for its row in the character table must **accumulate**, not
replace one another. The position table receives a gradient at every used
position.

Every parameter group is checked against central finite differences on a tiny
model in `tests/test_gradients.R`. Both absolute and relative error are
reported because a near-zero derivative can have unstable relative error.
Gradient clipping is applied to the combined global norm **after** manual
differentiation and is not part of numerical gradient checking.

## Why this remains a minimal model

There is one causal attention head, one block, a ReLU feed-forward network,
learned absolute positions and no dropout or weight tying. Sampling recomputes
the block on the most recent context at each step. The code makes no claim to
be computationally equivalent to an optimised Transformer inference engine.
