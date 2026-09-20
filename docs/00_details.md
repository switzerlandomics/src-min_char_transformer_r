# Project 2: A minimal Transformer from first principles in R

The finished project should demonstrate one specific development in language modelling: how causal self-attention can replace recurrent state as the mechanism for combining information from preceding characters.

We already have a working character-level RNN. Rather than changing the dataset, prediction task and development approach at the same time, we should build a small decoder-only Transformer that learns from the same Tiny Shakespeare text. The new project will retain the strengths of the RNN implementation: explicit mathematics, readable base R code, numerical tests, useful experiment logs, reproducible results and figures designed for publication.

The main design constraint is that this must be a complete, genuinely trainable model, not an isolated demonstration of an attention matrix. A reader should be able to clone the repository, run a smoke test, train on real text, observe the learning curves, inspect the attention mechanism and generate text from the saved model on an ordinary laptop CPU.

I would organise the project around five principles:

1. One new concept at a time. Keep character-level prediction familiar and make attention the central architectural change.

2. No hidden learning algorithm. Implement the forward calculations, backward calculations and parameter updates directly in base R.

3. Correctness before long training. Test numerical gradients, causal masking, save/load behaviour and the complete experiment pipeline before producing results.

4. A small model with meaningful outputs. Choose an initial configuration that is practical on a CPU, then measure its actual learning and runtime before deciding on the public demonstration.

5. An honest comparison with the RNN. Report precisely what is comparable and what differs, especially the handling of context between training windows.

The following is the implementation plan I would use.

## 1. Define the finished product before writing the code

The public demonstration should answer four questions in order.

|
Question

|

What the project demonstrates

|
| --- | --- |
|

What is the model trying to learn?

|

Predicting the next character in real text

|
|

What changes relative to the RNN?

|

Attention combines representations of available preceding positions rather than propagating a recurrent state

|
|

How does it learn?

|

Explicit forward pass, cross-entropy loss, manually calculated gradients and parameter updates

|
|

Does it work?

|

Tests, held-out prediction loss, generated text and inspectable attention weights

|

These questions should determine the code, figures and README structure. We should not add architectural features merely because they appear in contemporary Transformer implementations.

The initial implementation should have one causal self-attention head, one Transformer block and a character-level vocabulary. It should be a small but complete decoder-style language model, including learned character and position embeddings, attention, a feed-forward network, residual connections, layer normalisation and an output layer.

That is enough to demonstrate the architectural mechanism introduced by the Transformer family while keeping every operation accessible.

## 2. Choose a CPU-sized reference model

I would start with this configuration as a candidate, not an untested promise of performance.

|
Setting

|

Initial choice

|

Reason

|
| --- | --- | --- |
|

Vocabulary

|

Characters present in Tiny Shakespeare

|

Same prediction units as the RNN

|
|

Context length

|

32 characters

|

Small attention matrix that can be visualised and differentiated directly

|
|

Embedding width

|

32

|

Keeps the main matrix operations small

|
|

Attention heads

|

1

|

Makes the attention calculation and its gradients inspectable

|
|

Transformer blocks

|

1

|

Avoids obscuring the central mechanism with depth

|
|

Feed-forward width

|

64

|

Provides a small learned transformation after attention

|
|

Training batch

|

Start with one 32-character window

|

Simplifies indexing, gradients and debugging

|
|

Arithmetic

|

Base R matrices

|

No automatic differentiation or framework dependency

|

With a 65-character vocabulary, this design would have approximately 14,000 trainable parameters, depending on the precise placement of normalisation layers and whether the output embeddings are tied. The exact count should be calculated from the implemented model and checked by a unit test.

The previous RNN has 23,165 parameters under its reference configuration. The two models will therefore not initially have identical capacity. We should report this rather than imply that we have constructed a parameter-matched benchmark.

A 32-character context is useful pedagogically, but it is a substantial limitation. The Transformer will not be able to consult text outside the context window unless we explicitly provide a longer context. The RNN's hidden state can carry information between its training windows, so equal window lengths do not imply identical access to historical information.

I would first establish that the small Transformer is correct and learns. Only then would I consider increasing context length, embedding width or batch size.

# 3. The repository structure

I would name the project `min-char-transformer`. This makes the relationship to `min-char-rnn` immediately apparent without suggesting that we have implemented a production-scale GPT model.

The directory structure should be familiar to anyone who has just examined the RNN repository:

```
min-char-transformer/
├── .gitignore
├── README.md
│
├── R/
│   ├── data.R
│   ├── model.R
│   ├── optimiser.R
│   ├── plots.R
│   └── progress.R
│
├── data/
│   ├── input.txt
│   └── README.md
│
├── experiments/
│   ├── run.R
│   ├── plot_results.R
│   └── inspect_attention.R
│
├── tests/
│   ├── test_model.R
│   ├── test_gradients.R
│   ├── test_training.R
│   └── test_monitoring.R
│
├── docs/
│   ├── 01_about.md
│   ├── 02_attention.md
│   ├── 03_validation.md
│   ├── 04_results.md
│   ├── 05_rnn_comparison.md
│   └── figures/
│       ├── transformer_model.svg
│       └── causal_attention.svg
│
├── scripts/
│   └── create_zip.R
│
└── output/
    └── .gitkeep
```

The downloaded Shakespeare corpus and generated experiment files remain local and are excluded by `.gitignore`. The repository contains the small smoke-test corpus, readable source code, tests, documentation and selected publication figures.

I would not create a separate package or a complicated module framework. The RNN project's approach of sourcing a small number of explicit R files is sufficient.

## The main code files

### `R/data.R`: corpus and vocabulary

This can retain much of the previous project's approach: UTF-8 input validation, deterministic character indexing, encoding and decoding, checksum recording and automatic download of Tiny Shakespeare.

There is one deliberate improvement to make at this stage: separate training, validation and test data before using them for their respective purposes.

I propose an 80% training, 10% validation and 10% test split, with contiguous regions and no prediction target crossing a split boundary. Validation would support checkpoint selection and experiment development; the test portion would be reserved for the final selected model.

Because this differs from the RNN's earlier 90/10 split, its published results would not be directly comparable without a new RNN run under the same data protocol. We should not silently place the two historical learning curves side by side.

The vocabulary should preferably be established from training data alone. If held-out text contains a character absent from that vocabulary, the program should handle the situation explicitly rather than silently discard it. For Tiny Shakespeare, we can check this condition before training and document the actual result.

### `R/model.R`: the complete Transformer mathematics

This should remain the intellectual centre of the repository.

It will contain clearly named functions for model initialisation, the forward pass, cross-entropy loss, the backward pass, sampling, parameter counting, saving and loading.

The main functions should expose intermediate calculations where useful for testing and figures, but we should not expose every temporary matrix through a large public API.

A useful convention is to represent one sequence as a matrix with:

* Rows corresponding to character positions.

* Columns corresponding to embedding dimensions.

For a context of 32 characters and an embedding width of 32, the sequence representation is therefore a 32×3232\times3232×32 matrix.

Using this convention consistently will make the attention calculations much easier to follow and reduce the chance of subtle R indexing errors.

### `R/optimiser.R`: explicit parameter updates

The RNN implements AdaGrad directly. We can begin with the same optimiser for continuity, keeping its accumulators and updates visible.

However, we should treat the optimiser as a separate experimental decision. If the small Transformer does not train reliably with AdaGrad, a manually implemented Adam optimiser could be introduced and tested. We should not change the optimiser silently while claiming that the experiment isolates architecture.

This file should contain only the optimiser state and update calculations. It should not contain model mathematics or experiment orchestration.

### `R/progress.R`: familiar human-readable feedback

Preserve the existing logging conventions: labelled stages, timestamped messages, configuration details, progress bars, elapsed time and ETA.

The Transformer adds a few important quantities worth reporting at startup: context length, embedding width, attention-head count, feed-forward width, block count and exact parameter count.

The reporting should remain concise during training. A reader should be able to see whether the model is learning without being overwhelmed by matrix-level diagnostics on every update.

### `R/plots.R`: publication-quality visualisation

The plotting module should remain an optional observer, using `ggplot2` only for visualisation. Training, validation, sampling and checkpointing must work with `--no-plot`.

Its responsibilities will be broader than in the RNN project because we also want to display attention, but the same principle applies: plot rendering failures should be logged without destroying a running training experiment.

# 4. The mathematical implementation: what we need to build and verify

This is the most important technical section of the project plan.

We need a complete forward and backward calculation through the network. Calling a ready-made attention or Transformer function would defeat the purpose of this particular exercise.

I would use a pre-normalisation Transformer block and keep its exact order fixed and documented:

```
Input character indices
          │
          ▼
Character embeddings + position embeddings
          │
          ▼
     Layer norm
          │
          ▼
   Causal self-attention
          │
          ▼
    Residual addition
          │
          ▼
     Layer norm
          │
          ▼
   Feed-forward network
          │
          ▼
    Residual addition
          │
          ▼
   Final layer norm
          │
          ▼
  Output projection + softmax
          │
          ▼
Next-character probabilities
```

This is a simplified decoder-only Transformer, not a literal reproduction of every detail in GPT-2.

The implementation can be developed in the following order.

### A. Embeddings and output probabilities

Start with character embeddings, learned absolute position embeddings and an output projection.

The first executable milestone should be a complete, simple next-character predictor without attention. It must produce correctly shaped probability distributions and a finite cross-entropy loss.

This establishes the input-to-target alignment before we introduce more complicated calculations.

### B. Causal attention

Implement the query, key and value projections:

Q=XWQ,K=XWK,V=XWVQ=XW_Q,\qquad K=XW_K,\qquad V=XW_VQ=XWQ,K=XWK,V=XWV

Then calculate the attention scores:

S=QKTdkS=\frac{QK^\mathsf{T}}{\sqrt{d_k}}S=dkQKT

Apply a causal mask and a numerically stable, row-wise softmax:

A=softmax⁡(S+M)A=\operatorname{softmax}(S+M)A=softmax(S+M)

Finally combine the value vectors:

Z=AVZ=AVZ=AV

The attention matrix AAA has one row and one column for every character position. At position ttt, only columns corresponding to positions 1,…,t1,\ldots,t1,…,t may receive non-zero attention weight.

The mask must allow a character to attend to itself. This is valid because the representation at position ttt predicts the character at position t+1t+1t+1, not the character already supplied at position ttt.

We should make this a named, directly tested function. The model must never be able to see its future targets through incorrectly aligned input sequences or an incorrectly constructed mask.

### C. The complete Transformer block

Add the attention output projection, residual connections, layer normalisation and the feed-forward network.

The feed-forward network can use two learned linear transformations and a ReLU activation. This is sufficient for our educational model.

Each component should be implemented with clear dimensional conventions. In particular, R's implicit vector recycling should not be allowed to determine whether adding a bias or normalisation parameter behaves correctly. Matrix shapes and broadcasting should be explicit and tested.

### D. The complete backward pass

Only after the forward pass is correct should we implement the gradients.

The backward pass must propagate through the output projection, feed-forward network, both residual paths, normalisation, attention output, masked softmax, query/key/value projections and input embeddings.

The attention gradients are the central new mathematical work. We should derive them in a separate technical document and implement them in the corresponding code section.

A backward pass that merely allows training loss to decline is not sufficient evidence of correctness. Every trainable parameter group needs numerical gradient checking.

# 5. Tests: the conditions for calling the model correct

I would separate the tests into four files so that debugging remains manageable.

|
Test file

|

What it establishes

|
| --- | --- |
|

`test_model.R`

|

Dimensions, probability normalisation, embeddings, attention masking, forward pass and sampling

|
|

`test_gradients.R`

|

Manual gradients agree with numerical finite differences

|
|

`test_training.R`

|

The complete model learns a tiny known sequence and preserves reproducibility, checkpoints and resume behaviour

|
|

`test_monitoring.R`

|

Metrics parsing, plotting and browser-monitor outputs behave correctly

|

The causal-mask test deserves particular attention. Construct two input sequences that share an identical prefix but have different later characters. The predictions for positions within the shared prefix must remain unchanged. We should also check that masked attention probabilities are zero and that every attention row sums to one.

For gradient checking, use a deliberately tiny model: a vocabulary of four characters, a context of three or four positions and an embedding width of four. Disable clipping and numerical randomness for the check, perturb parameters by a small amount, and compare the analytical gradients with central finite differences.

Check embeddings and layer-normalisation parameters as well as the attention and feed-forward matrices. Near-zero gradients require care when calculating relative error; report meaningful absolute and relative errors rather than relying on a single unstable ratio.

The training test should use a tiny repeated character pattern whose next-character relationships are learnable. The test should verify a substantial loss reduction under a controlled training configuration rather than require every randomly initialised run to achieve an arbitrary exact loss.

Finally, resume testing should establish that an uninterrupted run and a stopped-and-resumed run produce matching model parameters and optimiser state when the seed, input and total training schedule are identical. This is especially valuable for an experiment intended to run for an extended period on a laptop.

# 6. The experiment runner

`experiments/run.R` should follow the design of the previous runner closely.

It should remain possible to start training with a single command and understand the resulting log without reading the source code.

## The three entry points

Fast correctness check

Bash

```
Rscript tests/test_model.R
Rscript tests/test_gradients.R
Rscript experiments/run.R --smoke --no-plot
```

The smoke test should use a bundled short text, a very small embedding width and context, and a few dozen updates. It must not require a download or optional graphics packages.

Short real-data experiment

Bash

```
Rscript experiments/run.R --iterations=2000
```

This is a proposed initial pilot, not yet the demonstrated training budget for a successful published run. It should download Tiny Shakespeare if needed, produce learning metrics, generate samples and save a resumable checkpoint.

Longer experiment

Bash

```
Rscript experiments/run.R --iterations=20000
```

We should decide whether this is sensible only after measuring the short run's actual CPU throughput, training behaviour and validation performance.

The final README should use the verified commands and defaults from the completed implementation, not necessarily these preliminary numerical settings.

## Human-readable logs

Retain the RNN project's five-stage structure, adjusting the stage names for the Transformer:

```
2026-09-20 14:30:00 | START  | min-char-transformer | base R | CPU-only
2026-09-20 14:30:00 | STAGE  | [1/5] Preparing data
2026-09-20 14:30:00 | DATA   | characters=... | vocabulary=... | train=... | validation=... | test=...
2026-09-20 14:30:00 | STAGE  | [2/5] Initialising model
2026-09-20 14:30:00 | MODEL  | context=32 | embedding=32 | heads=1 | blocks=1 | parameters=...
2026-09-20 14:30:00 | STAGE  | [3/5] Training
2026-09-20 14:30:00 | TRAIN  | [======..................] 25.0% | 500/2000 | elapsed ... | ETA ...
2026-09-20 14:30:00 | VALID  | loss=... nats/char | best=... at update ...
2026-09-20 14:30:00 | SAVE   | resumable checkpoint written
2026-09-20 14:30:00 | STAGE  | [4/5] Saving final model and results
2026-09-20 14:30:00 | STAGE  | [5/5] Experiment complete
```

These are illustrative log formats, not measured results.

The runner should report the locations of `training.html`, `samples.txt` and the experiment directory at startup or when those files first become available.

## Reproducibility and clean resume

We should preserve the existing `STOP` marker interface:

Bash

```
touch output/YOUR_EXPERIMENT_DIRECTORY/STOP
```

And resume to a new total target with:

Bash

```
rm output/YOUR_EXPERIMENT_DIRECTORY/STOP

Rscript experiments/run.R \
  --resume=output/YOUR_EXPERIMENT_DIRECTORY \
  --iterations=10000
```

The resumable checkpoint must include the model parameters, optimiser accumulators, current training-window position or sampling state, RNG state, current update number, metrics and experiment configuration.

The runner should also record the dataset checksum and reject a resume attempt if the input data or model-defining configuration has changed.

Generated samples and plots should use independent, controlled randomness so that viewing a figure or producing a sample does not alter subsequent training.

# 7. A training and evaluation protocol we can defend

The RNN project showed us why it is useful to separate an apparent improvement in training loss from reliable improvement on held-out text.

For the new model, I would define the evaluation protocol before training and leave it unchanged throughout the reference experiment.

## Training

Begin with independent 32-character windows drawn from the training portion of the corpus. Their next-character targets must lie entirely within the same training split.

For the first implementation, we can use deterministic traversal of consecutive windows, optionally shuffling their order once per pass using the recorded RNG state. The exact sampling policy should be fixed before the reference run and recorded in metadata.

Unlike the RNN, this Transformer does not carry a recurrent hidden state from one training window to the next. Its context is explicitly the supplied character window.

We should log both the number of parameter updates and the number of character positions processed. If we later introduce multiple windows per parameter update, those two quantities must remain distinct.

## Validation

Rather than repeatedly measuring only the first 2,000 held-out character transitions, select several fixed, non-overlapping validation passages distributed across the validation split.

For example, eight passages of 256 characters each provide 2,048 evaluated character positions. Divide each passage into context-limited windows and use a consistent rule for predictions at window boundaries.

Record the loss for each passage as well as the aggregate loss. Evaluate with no gradients, no parameter updates and no random sampling.

This gives us a more informative learning curve while keeping validation computationally bounded.

## Test data

The test portion should remain untouched during architecture selection, learning-rate adjustments and checkpoint selection.

Once the model and training procedure are finalised, evaluate the selected checkpoint on the test portion using a predefined context-window policy.

If we subsequently inspect those test results and revise the model, we should not continue describing the same measurements as an untouched final test.

## Baselines

The previous RNN project used uniform-character and add-one-smoothed bigram reference models. We should retain both.

Evaluate all baselines on the same held-out character positions as the Transformer. A model that performs below uniform loss has learned something, but that alone does not show that attention is useful: a simple bigram model may explain much of the improvement.

For the public result, the central quantitative question is how the Transformer compares with these simple predictors and with the RNN under a clearly documented evaluation procedure.

# 8. The figures: what should readers actually see?

The RNN project already has a useful visual style. For Project 2, I would make the figures serve distinct explanatory purposes rather than produce many variants of the same learning curve.

I would plan four principal figures, plus generated-text examples.

## Figure 1. What changes from recurrence to attention?

Vector illustration · `docs/figures/transformer_model.svg`

## Two mechanisms for using previous characters

A. Recurrent model

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(255%2C%20255%2C%20255\)%22%20viewBox%3D%220%200%20160%20122%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Crect%20x%3D%224%22%20y%3D%2248%22%20width%3D%2238%22%20height%3D%2228%22%20rx%3D%225%22%20fill%3D%22%23d9edfc%22%20stroke%3D%22%233181c1%22%2F%3E%3Crect%20x%3D%2261%22%20y%3D%2248%22%20width%3D%2238%22%20height%3D%2228%22%20rx%3D%225%22%20fill%3D%22%23d9edfc%22%20stroke%3D%22%233181c1%22%2F%3E%3Crect%20x%3D%22118%22%20y%3D%2248%22%20width%3D%2238%22%20height%3D%2228%22%20rx%3D%225%22%20fill%3D%22%23d9edfc%22%20stroke%3D%22%233181c1%22%2F%3E%3Cg%20stroke%3D%22currentColor%22%20stroke-width%3D%221.4%22%20fill%3D%22none%22%3E%3Cpath%20d%3D%22M42%2062%20H61%20M99%2062%20H118%20M23%2048%20V25%20M80%2048%20V25%20M137%2048%20V25%20M23%2076%20V98%20M80%2076%20V98%20M137%2076%20V98%22%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%22currentColor%22%20font-size%3D%2211%22%20text-anchor%3D%22middle%22%3E%3Ctext%20x%3D%2223%22%20y%3D%2218%22%3Eh%3C%2Ftext%3E%3Ctext%20x%3D%2280%22%20y%3D%2218%22%3Eh%3C%2Ftext%3E%3Ctext%20x%3D%22137%22%20y%3D%2218%22%3Eh%3C%2Ftext%3E%3Ctext%20x%3D%2223%22%20y%3D%2266%22%3ERNN%3C%2Ftext%3E%3Ctext%20x%3D%2280%22%20y%3D%2266%22%3ERNN%3C%2Ftext%3E%3Ctext%20x%3D%22137%22%20y%3D%2266%22%3ERNN%3C%2Ftext%3E%3Ctext%20x%3D%2223%22%20y%3D%22113%22%3ET%3C%2Ftext%3E%3Ctext%20x%3D%2280%22%20y%3D%22113%22%3Eh%3C%2Ftext%3E%3Ctext%20x%3D%22137%22%20y%3D%22113%22%3Ee%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fsvg%3E)Information passes through successive hidden states.

B. Attention model

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(255%2C%20255%2C%20255\)%22%20viewBox%3D%220%200%20160%20122%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Crect%20x%3D%224%22%20y%3D%2222%22%20width%3D%2238%22%20height%3D%2227%22%20rx%3D%225%22%20fill%3D%22%23d5f2e5%22%20stroke%3D%22%2322966b%22%2F%3E%3Crect%20x%3D%2261%22%20y%3D%2222%22%20width%3D%2238%22%20height%3D%2227%22%20rx%3D%225%22%20fill%3D%22%23d5f2e5%22%20stroke%3D%22%2322966b%22%2F%3E%3Crect%20x%3D%22118%22%20y%3D%2222%22%20width%3D%2238%22%20height%3D%2227%22%20rx%3D%225%22%20fill%3D%22%23d5f2e5%22%20stroke%3D%22%2322966b%22%2F%3E%3Crect%20x%3D%2250%22%20y%3D%2272%22%20width%3D%2260%22%20height%3D%2228%22%20rx%3D%225%22%20fill%3D%22%23d9edfc%22%20stroke%3D%22%233181c1%22%2F%3E%3Cg%20stroke%3D%22currentColor%22%20stroke-width%3D%221.4%22%20fill%3D%22none%22%3E%3Cpath%20d%3D%22M23%2049%20L63%2072%20M80%2049%20V72%20M137%2049%20L97%2072%22%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%22currentColor%22%20font-size%3D%2211%22%20text-anchor%3D%22middle%22%3E%3Ctext%20x%3D%2223%22%20y%3D%2239%22%3ET%3C%2Ftext%3E%3Ctext%20x%3D%2280%22%20y%3D%2239%22%3Eh%3C%2Ftext%3E%3Ctext%20x%3D%22137%22%20y%3D%2239%22%3Ee%3C%2Ftext%3E%3Ctext%20x%3D%2280%22%20y%3D%2289%22%3EAttention%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fsvg%3E)A position combines information from permitted earlier positions.

The final figure should use the same example text in both panels and show where each model obtains its contextual information. It must not imply that the Transformer is allowed to inspect future target characters.

This is the main conceptual figure for the README and blog. We should create it as an editable SVG with actual vector shapes and text, not a PNG embedded inside an SVG.

## Figure 2. A causal attention matrix

Vector or high-resolution plot · `attention.png` / selected SVG

## Which preceding positions can influence a prediction?

Use a short fixed input such as `The king`, with rows corresponding to the current input position and columns corresponding to positions that can be attended to.

The figure should show a lower-triangular attention matrix. Future positions are masked, while the permitted positions receive weights whose row sum is one.

We can show the same input before and after training to demonstrate that the attention weights are learned rather than manually specified. The plotted weights must be extracted from the actual model and saved with the checkpoint or enough metadata to reproduce the figure.

Interpretation limit: attention weights describe the model's weighting of value representations; they are not, on their own, a complete causal explanation of its output.

A second, smaller annotation could show the actual predicted next-character distribution at one selected position. That connects the attention calculation to the language-modelling task without suggesting that the attention matrix itself is the final prediction.

## Figure 3. Training and validation performance

Experiment output · `training.png` and `training_mobile.png`

## Does the model improve on unseen text?

Retain the restrained style of the RNN learning curves: white background, explicit units, labelled training and validation lines, a uniform baseline and a marker for the lowest measured validation loss.

Add the bigram baseline when it is useful and does not make the figure crowded.

Do not plot training and validation as if they were identical measurements. The training series may use smoothing, while validation measures fixed held-out passages. State this in the caption.

For mobile screens, I would generate a separate compact plot, not simply shrink a 10-inch desktop figure. The mobile version should use fewer x-axis labels, larger text relative to the canvas, a concise subtitle and a legend that remains readable at roughly 360 CSS pixels.

The local `training.html` can select the appropriate image using responsive HTML. The README and blog can use the mobile figure where the publishing layout requires it.

## Figure 4. Validation across independent passages

Experiment output · `validation_detail.png`

## Is the improvement consistent across held-out text?

Show the aggregate validation loss and the variation across the predefined held-out passages. This could use a small number of passage-level points at selected checkpoints rather than eight overlapping lines across the entire training run.

The purpose is to distinguish a broad change in validation performance from an abrupt change driven by one particular passage.

This figure is an improvement carried forward from what we learned during the longer RNN experiment.

## Generated text should remain actual text

I would show generated samples as short, carefully labelled excerpts rather than turn them into a fifth complicated plot.

The samples should include an untrained model, an early checkpoint and the selected trained checkpoint, using the same seed prompt, generation length and sampling temperature. Preserve the exact generated output, including malformed words and punctuation.

We should not select only the most impressive sample and imply that it is representative. If we publish a representative passage, state how it was chosen and make the complete saved samples available in the local experiment output or a small curated results file.

# 9. What are the main results we should aim to obtain?

We cannot know the actual learning curves, runtime or generated-text quality until the Transformer is implemented and trained. The public project should therefore be organised around measured results, not an anticipated claim that it will outperform the RNN.

I would define the following outcomes before beginning the longer experiment.

|
Result

|

Evidence needed

|
| --- | --- |
|

The mathematical implementation is correct

|

Passing analytical-versus-numerical gradient tests and causal-mask tests

|
|

The model is trainable

|

A controlled small-data experiment with a reproducible reduction in loss

|
|

The model learns from real text

|

Recorded training and held-out losses on Tiny Shakespeare

|
|

The learned model is useful beyond a uniform guess

|

Validation loss compared with uniform and bigram baselines

|
|

Generation works autoregressively

|

Actual saved samples in which predicted characters become subsequent inputs

|
|

The run is reproducible

|

Saved configuration, corpus checksum, RNG state, model and optimiser state

|
|

The CPU cost is practical

|

Measured updates per second, wall-clock time and memory use

|
|

The attention mechanism is observable

|

A recorded causal attention matrix from an actual checkpoint

|

The public experiment should not proceed to its final long run merely because the smoke test passes.

First, a short pilot must establish that the optimiser, learning rate and model dimensions produce numerically stable training. Then we should verify that held-out loss improves enough to make the demonstration meaningful.

If the Transformer beats the uniform baseline but not the bigram baseline, that is still a valid observation. We should report it and investigate whether the limitation comes from model capacity, context length, training duration or optimisation. We should not replace the recorded result with a more favourable one without documenting the changed experiment.

# 10. The README layout

The final README should feel immediately familiar to readers of `min-char-rnn`.

I would use the following order.

## Proposed `README.md`

1. Title and introduction. One or two paragraphs explaining the relationship to `min-char-rnn`, the next-character prediction task and what is implemented directly in base R.

2. Architecture figure. The RNN-versus-Transformer SVG, close enough to the introduction to explain the central change visually.

3. Quick test and run. The shortest commands needed to verify the model, run the bundled smoke test and begin a real-data experiment. Readers should not need to scroll through the mathematical derivation before finding them.

4. How the model learns. Follow `The king` through character and position embeddings, causal attention, next-character probabilities, cross-entropy and parameter updates. Include the central attention equation and one matrix-shape example.

5. Requirements and configuration. Base R, optional `ggplot2`, dataset download, initial model dimensions and a compact custom-corpus command.

6. Training and monitoring. Human-readable logs, learning curves, generated samples and the browser monitor.

7. Understanding the results. Explain nats per character, the evaluation split, the fixed validation passages, the baseline predictors and the limits of interpreting attention weights.

8. Stop, resume and saved files. The `STOP` marker, resume command and meanings of `best_model.rds`, `model.rds` and `latest_checkpoint.rds`.

9. Repository structure. A short file-purpose table, followed by links to the more detailed documents.

10. RNN comparison and references. Link to the previous project, explain the limits of any comparison, and cite the original Transformer paper and the relevant decoder-only language-modelling work.

The README should be a complete introduction, but it does not need to contain the entire backward-pass derivation. That belongs in `docs/02_attention.md`, where we can explain each matrix and gradient without making the repository front page unwieldy.

The key editorial decision is to show how to run it before explaining every part of how it works. A reader can execute the small example immediately, then return to the mathematical explanation with an actual experiment in mind.

# 11. What belongs in the experiment output?

Each run should create a timestamped directory following the existing RNN pattern.

```
output/20260920_143000_seed42/
├── experiment.log
├── metadata.rds
├── metrics.csv
├── validation_passages.csv
├── samples.txt
├── vocab.rds
├── best_model.rds
├── model.rds
├── latest_checkpoint.rds
├── attention_snapshot.rds
├── training.html
├── training.png
├── training_mobile.png
├── validation_detail.png
├── attention.png
└── FINISHED
```

The precise list can be refined during implementation, but the distinction between file types should remain clear.

`metrics.csv` contains aggregate training and validation measurements. `validation_passages.csv` records the individual held-out passage losses so that the validation figure can be recreated without retraining. `attention_snapshot.rds` stores the input positions and attention weights underlying the example attention figure, alongside the model checkpoint used to produce them.

The latest model, best-validation model and resumable checkpoint must remain separate. The full checkpoint also needs the optimiser state, training position and RNG state, not just the model weights.

The browser monitor is an observer, not a training control. A separate plotting process should be able to reconstruct every figure from the recorded experiment files.

# 12. A staged development plan

I would divide implementation into six milestones. Each should have a concrete acceptance condition before we proceed.

1. Data, vocabulary and a minimal predictor

   Establish the repository, UTF-8 handling, train/validation/test split, deterministic vocabulary and a simple character-level output model. Run the smoke test and verify input-to-target alignment.

   Complete when: a short real sequence produces correctly shaped next-character probabilities and a finite loss.

2. Attention and causal masking

   Implement the query, key and value projections, score calculation, stable masked softmax and weighted value combination. Produce an attention matrix for a short input.

   Complete when: dimensions, row sums, masked probabilities and prefix-invariance tests pass.

3. Complete Transformer and manual differentiation

   Add the residual paths, layer normalisation, feed-forward network, output projection and backward pass. Check every parameter group against finite differences.

   Complete when: the entire model passes numerical gradient tests and learns a deliberately small repeated text pattern.

4. Training runner and experiment controls

   Implement the CLI, readable logs, deterministic sampling, validation, baselines, metadata, clean stopping, atomic checkpoint writing and resume behaviour.

   Complete when: smoke training, uninterrupted training and stopped-and-resumed training pass their integration tests.

5. Real-data pilot and visualisation

   Run a short Tiny Shakespeare experiment, measure actual CPU throughput, confirm numerical stability, and produce the learning curves and attention figure. Check their appearance at desktop and mobile sizes.

   Complete when: the plotted values match saved metrics, the attention figure corresponds to an actual saved model, and the training run has a documented, reproducible outcome.

6. Reference run, comparison and publication

   Freeze the training protocol, run the reference experiment, select its checkpoint using validation loss, evaluate the held-out test portion and document the results. Re-run or adapt the earlier RNN under the same evaluation protocol if a direct comparison is part of the article.

   Complete when: every published numerical claim and figure can be traced to a saved experiment, and the README commands work from a clean repository checkout.

The last milestone must be based on the measured results. We should not finalise the blog's performance claims or the default long-run command until the pilot has established what is practical on the reference CPU.

## The finished public demonstration

A reader will see two small language models solving the same recognisable task. The first, our RNN, carries information forward through a recurrent hidden state. The second, our Transformer, uses causal attention to combine information from the characters available in its current context. Both are built from the underlying calculations in R rather than assembled from pretrained models or deep-learning framework functions. The Transformer repository will let readers run the model, inspect its attention matrix, watch it learn from Shakespeare and see the text it generates.

That provides a concrete way to understand an important transition in AI history. The original Transformer architecture was introduced in 2017, and later decoder-only language models such as GPT-2 demonstrated how attention-based text generation could be developed at much greater scale. Our model will be deliberately small, but it will implement and test the central mechanism in a complete working experiment. The public value is not a claim that R produces a better language model; it is that the reader can see exactly what was built, how it learns, what it actually achieves and how the architecture differs from the RNN that came before it.

