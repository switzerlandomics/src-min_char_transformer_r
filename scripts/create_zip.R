#!/usr/bin/env Rscript
# Archive the source, tests and documentation, excluding downloaded data and outputs.
args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script <- if (length(args)) sub("^--file=", "", args[1L]) else "scripts/create_zip.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
name <- basename(root)
parent <- dirname(root)
relative_files <- c(
  ".gitignore", "README.md", "LICENSE", "data/input.txt", "data/README.md",
  file.path("R", list.files(file.path(root, "R"), pattern = "[.]R$")),
  file.path("experiments", list.files(file.path(root, "experiments"), pattern = "[.]R$")),
  file.path("tests", list.files(file.path(root, "tests"), pattern = "[.]R$")),
  file.path("docs", list.files(file.path(root, "docs"), pattern = "[.]md$")),
  file.path("docs/figures", list.files(file.path(root, "docs", "figures"), pattern = "[.]svg$")),
  "scripts/create_zip.R")
missing <- relative_files[!file.exists(file.path(root, relative_files))]
if (length(missing)) stop("Missing project files: ", paste(missing, collapse = ", "))
if (!nzchar(Sys.which("zip"))) stop("A system 'zip' executable is required.")
zipfile <- file.path(parent, paste0(name, ".zip"))
if (file.exists(zipfile)) unlink(zipfile)
old <- setwd(parent)
on.exit(setwd(old), add = TRUE)
utils::zip(zipfile, file.path(name, relative_files), flags = "-9X")
cat("Created: ", normalizePath(zipfile, mustWork = TRUE), "\n", sep = "")
