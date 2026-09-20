# Project 2: Building a minimal Transformer from first principles in R

The clearest continuation of `min-char-rnn` is to keep the language-modelling problem the same and change the architecture that solves it. This gives us a controlled way to understand what a Transformer does differently from a recurrent neural network, rather than beginning an unrelated AI project.

Our first project established that a small network can learn to predict the next character from ordinary text. We implemented the forward pass, loss, gradients and parameter updates in base R, trained the network on Tiny Shakespeare and observed its predictions improve. We also saw that better training loss does not guarantee consistently better validation performance or coherent generated prose.

The next project should preserve that transparency while answering a new question: How can a network use attention to represent preceding text without passing everything through a single recurrent hidden state?

That is the architectural change we want to make visible.

## 1. Where this fits in AI history

I would use three reference points, each with a precise role.

2015 · Our starting point

Karpathy's minimal character-level RNN

The model reads characters sequentially. At each position, it combines the current character with its previous hidden state and predicts the next character. The hidden state carries information forward through the sequence.

Our first R project implements this mechanism directly.

2017 · The architectural change

Attention Is All You Need

The Transformer replaces the recurrent sequence-processing mechanism with attention-based layers. At each position, attention can combine information from other permitted positions in the sequence without passing it through every intervening recurrent step.

The original paper presented an encoder-decoder Transformer for sequence-to-sequence tasks. Our next project will use only the causal, decoder-style language-modelling part of the architecture.

2019 onwards · Scaling the approach

GPT-2 and subsequent large language models

GPT-2 demonstrated autoregressive text generation using a decoder-only Transformer trained at substantially greater parameter, data and computational scale.

Our project will implement a very small model from this architectural family. It will not reproduce GPT-2's scale, subword tokenisation, multiple Transformer blocks or training regime.

AlexNet can remain a brief historical reference in the accompanying article because it illustrates the importance of data and computation in deep learning. However, it should not be presented as a direct architectural predecessor of either our RNN or the Transformer. The useful sequence for this project is specifically recurrent language modelling → attention-based sequence modelling → decoder-only Transformer language models.

## 2. Keep the experiment familiar

We should retain Tiny Shakespeare, the character vocabulary, the next-character prediction objective and the basic training/validation distinction from `min-char-rnn`.

The input sequence remains straightforward:

```
Input:   T  h  e  [space]  k  i  n
Target:  h  e  [space]  k  i  n  g
```

The new model is still asked to predict a probability distribution for the next character at every position. We have not introduced a new objective or asked the network to answer questions, follow instructions or classify text.

This continuity is essential. It allows someone who has read the first project to concentrate on what changed inside the network rather than having to understand a new dataset and evaluation task at the same time.

It also provides a direct experimental comparison. We can evaluate both models on the same held-out text and examine how they learn and generate passages under clearly documented training conditions.

## 3. What exactly are we going to implement?

I would build a minimal decoder-only Transformer with one causal self-attention head and one Transformer block.

This is deliberately smaller than GPT-2, but it includes the architectural components needed to demonstrate how Transformer language modelling works.

## The complete model

Input characters ↓ Character embeddings + position embeddings ↓ Causal self-attention ↓ Residual connection and normalisation ↓ Feed-forward neural network ↓ Residual connection and normalisation ↓ Output scores + softmax ↓ Next-character probabilities

One Transformer block, one attention head and a character-level output vocabulary. The same block parameters are applied across sequence positions.

We would implement the numerical operations directly with base R arrays and matrices. As with the RNN, no pretrained weights, Python bridge, deep-learning framework or automatic differentiation would be involved.

Here is the learning progression I would use to make each component understandable.

### Character embeddings: from identity to a learned representation

In the RNN, a one-hot vector identifies the input character. The Transformer can instead use a learned embedding: a short numerical vector associated with each character.

Initially these vectors are random. Training changes them alongside the other model parameters.

The important distinction is that a character's embedding is not yet its contextual meaning. It represents the character before the attention layer has combined information from the preceding sequence.

We should show the dimensions explicitly. For example, with 25 character positions and an embedding width of 64, the initial sequence representation is a 25×6425\times6425×64 matrix.

This gives readers a concrete object to follow through the rest of the network.

### Position embeddings: retaining sequence order

A Transformer processes several character representations together rather than updating one recurrent state after another. It therefore needs a way to represent character positions.

We can add a learned positional vector to each character embedding. The representation for `h` in position two is then distinguishable from the representation for `h` in another position.

For the first project, I would use learned absolute position embeddings because their implementation is easy to inspect. More sophisticated positional methods would add complexity without helping us establish the basic mechanism.

### Causal self-attention: the central experiment

This is the part that should receive the most explanation and the clearest figure.

Each character representation is transformed into three vectors, conventionally called its query, key and value.

At a given position, the query is compared with keys from the permitted positions. Those comparisons produce attention scores. Softmax converts the scores into weights, which determine how the corresponding value vectors are combined.

The central calculation is:

Attention⁡(Q,K,V)=softmax⁡(QKTdk+M)V\operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^\mathsf{T}}{\sqrt{d_k}}+M \right)VAttention(Q,K,V)=softmax(dkQKT+M)V

Here MMM is a causal mask. It prevents a position from accessing characters that occur later in the sequence.

For example, when predicting the character after `The k`, the model can use the preceding characters in `The k`. It must not see the subsequent `ing` from the target text.

This is the important contrast with the RNN: rather than relying solely on information that has been repeatedly transformed and carried forward through one recurrent state, the attention calculation directly combines representations from the available preceding positions.

We can make this visible with an attention matrix. Rows represent the position making the prediction; columns represent the positions it is allowed to attend to. The future-facing portion of the matrix is masked.

An attention heatmap is useful for showing where the model places attention weight. We should not present those weights as a complete explanation of why the model made a particular prediction. The final prediction also depends on the value vectors, other learned transformations and the output layer.

### The remaining block: transforming the attention result

Attention alone is not the complete Transformer block.

We should include a small feed-forward network, residual connections and layer normalisation. These allow the contextual representation to undergo additional learned transformations while preserving information through the block.

For this educational implementation, a simple two-layer feed-forward network with a ReLU activation is sufficient.

We can avoid dropout, multiple attention heads, multiple stacked blocks and more elaborate optimisation techniques in the initial version. They can be introduced later as separate experiments, after the single-block model is demonstrably correct.

## 4. The part that must genuinely be built from first principles

The architectural code should not simply call an attention function supplied by an R package. Nor should we delegate training to an automatic-differentiation system.

As in `min-char-rnn`, the central model file should contain the forward calculations and their corresponding backward calculations.

The greatest new implementation work will be the gradients through attention, including the query, key and value projections, the masked softmax, the weighted combination of values, and the subsequent transformations. We will also need derivatives for the embeddings, normalisation and feed-forward network.

I would preserve the gradient-checking principle from the first project. For small test inputs and a tiny model, compare the manually calculated gradients with finite-difference estimates.

This is particularly important here. A Transformer might produce plausible-looking text despite an error in part of its backward pass. A decreasing training loss alone would not establish that we implemented the intended mathematics correctly.

We should also test that changing a future character cannot change the predictions at preceding positions. That provides a direct test of the causal mask and catches a particularly consequential class of language-model implementation errors.

## 5. What should we hold constant when comparing the models?

Our central scientific question is architectural, so the comparison should be designed to prevent obvious differences in the experiment from being mistaken for differences between recurrence and attention.

|
Element

|

Proposed approach

|
| --- | --- |
|

Dataset

|

Same Tiny Shakespeare file and character vocabulary

|
|

Prediction task

|

Same next-character targets and cross-entropy loss

|
|

Data separation

|

Same contiguous training and held-out portions

|
|

Context

|

Begin with the same 25-character training-window length

|
|

Validation

|

Evaluate both on several fixed held-out passages, with no parameter updates

|
|

Initial comparison

|

Similar parameter count where practical, with exact counts reported

|
|

Training

|

Report both parameter updates and characters processed

|
|

Results

|

Held-out loss, learning curves, generated samples, runtime and memory use

|

There are two important qualifications.

First, the same 25-character window does not give the two architectures identical access to history. Our RNN can carry its hidden state between successive training windows. A standard Transformer with a 25-character context cannot access characters outside that context unless we explicitly provide a longer or overlapping input. We should document this difference, not describe the experiments as having identical memory.

Second, the same number of parameter updates does not necessarily mean the same computational work or the same amount of training data. We should report the number of character positions used for learning, approximate corpus passes, parameter count and measured CPU runtime.

I would make the first comparison simple and transparent rather than claim it is a comprehensive benchmark of RNNs against Transformers.

## 6. Use stronger evaluation from the beginning

Our longer RNN run revealed why a single validation curve can be difficult to interpret. The network's measured validation loss changed abruptly, while the training curve remained much smoother.

For the Transformer project, I would retain the fixed held-out split but evaluate several fixed passages distributed through it. We can report both the aggregate loss and the individual passage losses.

This makes it possible to distinguish deterioration affecting many held-out passages from deterioration concentrated in one passage.

We should also preserve a separate untouched test portion if we intend to select model settings or checkpoints using validation loss. The test set would be evaluated only after those decisions are made.

For generated-text comparisons, we should use identical seed prompts and document the sampling temperature and random seed. A small number of generated passages is useful for seeing how the models behave, but it is not a replacement for a quantitative held-out evaluation.

## 7. What would make the finished project convincing?

The project should demonstrate more than the existence of an `attention()` function in R.

I would aim for four concrete outputs.

An inspectable architecture figure.

Show the character embeddings, positional information, causal attention, feed-forward block and output predictions. Accompany it with a small attention matrix for `The king` so the reader can connect the abstract calculation to actual character positions.

A tested implementation.

Include finite-difference gradient checks, matrix-dimension tests, numerical-stability checks, a causal-mask test and a small end-to-end training test. These establish that the implementation behaves as intended.

A real training experiment.

Train on Tiny Shakespeare, save checkpoints and report held-out loss, generated text and CPU runtime. Include a baseline predictor so the result can be judged against something simpler than a neural network.

A documented RNN comparison.

Run the previous model and the Transformer under explicitly stated conditions. Explain what the comparison demonstrates and which differences in context, computation and training procedure prevent broader conclusions.

The repository can follow the same design as our first project: one readable model implementation, separate data preparation and experiment scripts, tests, concise technical documentation and a small collection of selected figures. The downloaded corpus, full experiment outputs and checkpoints should remain outside Git.

## 8. What not to attempt in the first version

I would not attempt to reproduce GPT-2 or construct a multi-layer, multi-head Transformer immediately.

That would introduce too many simultaneous sources of complexity: multiple attention heads, repeated blocks, large parameter matrices, substantial training requirements and difficult numerical debugging.

Instead, we should establish that the smallest complete decoder-style Transformer can learn from real text using our own R implementation. Once it is correct, we can ask controlled questions: Does adding a second head change performance? Does extending the context help? What happens when we stack another block?

Each change then has an interpretable purpose and a measured outcome.

We should also avoid promising that the Transformer will generate better text than the RNN. A small attention-based model with limited training may perform similarly or worse. The value of the experiment is that we will be able to see and measure how the two architectures operate.

## The project in plain English

Our first project built a small language model in R that reads text one character at a time and carries information forward in a recurrent hidden state. For the next project, we will keep the same Shakespeare text and the same task of predicting the next character, but build a different neural network from first principles in R. Instead of relying on a recurrent hidden state, the new model will use attention to combine information from the preceding characters when making each prediction. We will implement the calculations and training ourselves, then compare the two models on the same held-out text and examine the passages they generate.

This follows an important development in AI history. Recurrent networks were widely used for sequence modelling before the Transformer architecture was introduced in 2017. GPT-2 and subsequent large language models demonstrated how decoder-only Transformers could be trained at much greater scale. Our project will not reproduce a modern frontier model, but it will implement the central attention-based language-modelling mechanism in a small, complete R program. By keeping the experiment familiar and changing the architecture, we can see precisely what the Transformer introduced and how that mechanism differs from the RNN we have already built.

