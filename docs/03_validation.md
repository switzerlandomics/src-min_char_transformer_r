# Evaluation protocol

## Separation of data

The runner divides the exact UTF-8 corpus into contiguous 80% training, 10%
validation and 10% test regions. All training input positions and their next
character targets remain inside the training split. The deterministic
vocabulary is fitted using training text only. Unseen held-out characters
cause an explicit error.

A parameter update uses one non-overlapping context-sized training window.
Training shuffles the order of eligible window starts once per traversal. No
recurrent state carries information between windows. The checkpoint stores the
window order, current position and random-number state, allowing the next
update to resume consistently.

## Fixed evaluation passages

The runner chooses non-overlapping passages distributed across each held-out
region. By default there are eight passages of 256 next-character transitions
in each held-out split. Within a passage, evaluation divides the positions
into **disjoint 32-character input windows and their 32 following targets**.
The context resets at every window boundary. A first position within a window
therefore has less available history than a later position, and evaluation
does not claim to provide a sliding 32-character context for every target.

The same selection and window rule applies at each validation checkpoint. The
loss is a character-weighted average across passages; per-passage losses are
also saved. The runner evaluates the **test** passages only when explicitly
invoked with `--test`, after selecting a model by validation loss. It refuses
to overwrite an existing `test_results.rds` in the same experiment directory.
Repeated examination of test results for model selection invalidates their
interpretation as an untouched final test.

## Baselines and limitations

The uniform baseline is `log(V)` nats per character. The add-one-smoothed
bigram baseline is fitted on adjacent character pairs from **training text
only** and evaluated on exactly the transitions used for Transformer
validation. Bigram probabilities are `(count(previous,next)+1) /
(count(previous)+V)`.

The training curve is exponentially smoothed mean loss on successive training
windows; validation is an unsmoothed loss on fixed held-out passages. Their
numerical difference should not be presented as a precisely matched
train/test generalisation gap. Generated samples are qualitative and depend
on prompt, sampling temperature and random seed.

The earlier RNN's published run used another split and a carried hidden state.
For a controlled comparison, both architectures need a new run with the same
corpus bytes, split, target positions and evaluation protocol, with their
parameter counts and computation reported separately.
