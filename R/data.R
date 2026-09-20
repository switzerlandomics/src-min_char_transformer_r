# UTF-8 text, deterministic vocabularies, data splits and evaluation windows.
# This module does not depend on the neural-network implementation.

read_text_file <- function(path) {
  if (!file.exists(path)) stop("Input file does not exist: ", path)
  size <- file.info(path)$size
  if (!is.finite(size) || size < 2) stop("Input must contain at least two bytes.")
  text <- rawToChar(readBin(path, what = "raw", n = size))
  Encoding(text) <- "UTF-8"
  if (is.na(iconv(text, from = "UTF-8", to = "UTF-8", sub = NA))) {
    stop("Input must be valid UTF-8 text.")
  }
  text
}

text_to_characters <- function(text, minimum_size = 1L) {
  if (!is.character(text) || length(text) != 1L || is.na(text)) {
    stop("text must be one non-missing character string.")
  }
  characters <- strsplit(text, "", fixed = TRUE)[[1L]]
  if (length(characters) < minimum_size) stop("Text is shorter than the required character count.")
  characters
}

build_vocab <- function(text) {
  characters <- sort(unique(text_to_characters(text, minimum_size = 2L)), method = "radix")
  index <- seq_along(characters)
  names(index) <- characters
  list(chars = characters, char_to_ix = index, ix_to_char = characters,
       size = length(characters))
}

encode_text <- function(text, vocab) {
  characters <- text_to_characters(text)
  indices <- unname(vocab$char_to_ix[characters])
  if (anyNA(indices)) {
    stop("Characters absent from the training vocabulary: ",
         paste(unique(characters[is.na(indices)]), collapse = " "))
  }
  as.integer(indices)
}

decode_indices <- function(indices, vocab) {
  if (!length(indices)) return("")
  if (anyNA(indices) || any(indices < 1L | indices > vocab$size)) {
    stop("Character indices are outside the one-based vocabulary range.")
  }
  paste0(vocab$ix_to_char[indices], collapse = "")
}

split_characters <- function(text, train_fraction = 0.8, validation_fraction = 0.1,
                             minimum_size = 2L) {
  if (!is.finite(train_fraction) || !is.finite(validation_fraction) ||
      train_fraction <= 0 || validation_fraction <= 0 ||
      train_fraction + validation_fraction >= 1) {
    stop("Split fractions must be positive and leave a non-empty test split.")
  }
  characters <- text_to_characters(text, minimum_size = 2L)
  n <- length(characters)
  train_end <- floor(n * train_fraction)
  validation_end <- floor(n * (train_fraction + validation_fraction))
  sizes <- c(train_end, validation_end - train_end, n - validation_end)
  if (any(sizes < minimum_size)) {
    stop("Dataset too small for the requested split and context length.")
  }
  list(train = paste0(characters[seq_len(train_end)], collapse = ""),
       validation = paste0(characters[seq.int(train_end + 1L, validation_end)], collapse = ""),
       test = paste0(characters[seq.int(validation_end + 1L, n)], collapse = ""),
       sizes = setNames(as.integer(sizes), c("train", "validation", "test")))
}

# A target at start + context_length must remain within the same data split.
training_window_starts <- function(n_characters, context_length) {
  if (context_length < 1L || n_characters < context_length + 1L) {
    stop("Not enough characters for one training window and its targets.")
  }
  seq.int(1L, n_characters - context_length, by = context_length)
}

# A fixed, non-overlapping selection distributed across a held-out split.
# All passage lengths are multiples of context_length: evaluation resets context
# at each window, without using characters from another split or passage.
make_passage_plan <- function(n_characters, context_length, n_passages = 8L,
                              transitions_per_passage = 256L) {
  if (context_length < 1L || n_passages < 1L ||
      transitions_per_passage < context_length ||
      transitions_per_passage %% context_length != 0L) {
    stop("Invalid passage configuration: length must be a multiple of context.")
  }
  latest_start <- n_characters - transitions_per_passage
  if (latest_start < 1L ||
      (n_passages > 1L && latest_start < 1L +
       (n_passages - 1L) * (transitions_per_passage + 1L))) {
    stop("Held-out split is too short for these non-overlapping passages.")
  }
  starts <- if (n_passages == 1L) 1L else {
    as.integer(round(seq(1, latest_start, length.out = n_passages)))
  }
  data.frame(passage_id = seq_len(n_passages), start = starts,
             transitions = rep.int(as.integer(transitions_per_passage), n_passages))
}

# Construct inputs and the immediately following targets, with no split crossing.
sequence_window <- function(indices, start, context_length) {
  if (length(start) != 1L || start < 1L ||
      start + context_length > length(indices)) stop("Window exceeds its data split.")
  list(inputs = indices[seq.int(start, length.out = context_length)],
       targets = indices[seq.int(start + 1L, length.out = context_length)])
}

input_checksum <- function(path) unname(tools::md5sum(path))

save_vocab <- function(vocab, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(vocab, path)
  invisible(path)
}

load_vocab <- function(path) {
  vocab <- readRDS(path)
  if (!is.list(vocab) || !all(c("chars", "char_to_ix", "ix_to_char", "size") %in%
                                    names(vocab))) stop("Invalid vocabulary file.")
  vocab
}
