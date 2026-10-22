#!/usr/bin/env Rscript
# Plot one PC-by-PC correlation heatmap per method from a combined CSV.
# Requires: install.packages("pheatmap")

suppressPackageStartupMessages(library(pheatmap))

cli <- commandArgs(trailingOnly = TRUE)
if (length(cli) %% 2 != 0 || any(!grepl("^--", cli[seq(1, length(cli), 2)]))) {
  stop("Provide arguments as --key value pairs.", call. = FALSE)
}
opts <- as.list(cli[seq(2, length(cli), 2)])
names(opts) <- sub("^--", "", cli[seq(1, length(cli), 2)])
opt <- function(key, default = NULL) if (!is.null(opts[[key]])) opts[[key]] else default
csv_list <- function(s) if (is.null(s) || !nzchar(s) || s == "none") character() else trimws(strsplit(s, ",", fixed = TRUE)[[1]])

input_csv <- opt("input_csv")
output_dir <- opt("output_dir")
if (is.null(input_csv) || is.null(output_dir)) {
  stop("Required: --input_csv FILE --output_dir DIR", call. = FALSE)
}
method_col <- opt("method_col", "method")
annotation_cols <- csv_list(opt("annotation_cols", "dataset,n_iter,n_oversamples"))
sort_cols <- csv_list(opt("sort_cols", "dataset,n_iter,n_oversamples,seed"))
pc_pattern <- opt("pc_pattern", "^PC[0-9]+_corr$")
width <- as.numeric(opt("width", "10"))
height <- as.numeric(opt("height", "12"))
fontsize <- as.numeric(opt("fontsize", "12"))
fontsize_col <- as.numeric(opt("fontsize_col", "10"))
if (!is.finite(width) || !is.finite(height) || min(width, height) <= 0) stop("Invalid width/height")

df <- read.csv(input_csv, check.names = FALSE)
needed <- unique(c(method_col, annotation_cols, sort_cols))
if (length(setdiff(needed, names(df)))) {
  stop("Missing CSV columns: ", paste(setdiff(needed, names(df)), collapse = ", "))
}
pc_cols <- grep(pc_pattern, names(df), value = TRUE)
if (!length(pc_cols)) stop("No PC correlation columns match --pc_pattern")
# Keep numeric PC order regardless of whether PC10 precedes PC2 in the input file.
pcs_nums <- as.integer(gsub("[^0-9]", "", pc_cols))
if (!anyNA(pcs_nums)) pc_cols <- pc_cols[order(pcs_nums)]

filter_col <- opt("filter_col")
if (!is.null(filter_col)) {
  if (!filter_col %in% names(df)) stop("Unknown filter column: ", filter_col)
  filter_values <- csv_list(opt("filter_values"))
  if (!length(filter_values)) stop("--filter_col requires --filter_values")
  df <- df[as.character(df[[filter_col]]) %in% filter_values, , drop = FALSE]
}
if (!nrow(df)) stop("No rows remain after filtering")

methods <- csv_list(opt("methods"))
if (!length(methods)) methods <- unique(as.character(df[[method_col]]))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Palettes depend on the number of observed categories, not on dataset/parameter names.
palettes <- list(
  function(n) grDevices::hcl(seq(15, 375, length.out = n + 1)[seq_len(n)], c = 75, l = 65),
  function(n) grDevices::colorRampPalette(c("#e5f5f9", "#238b45"))(n),
  function(n) grDevices::colorRampPalette(c("#efedf5", "#6a51a3"))(n)
)

for (method in methods) {
  d <- df[as.character(df[[method_col]]) == method, , drop = FALSE]
  if (!nrow(d)) { warning("No records for method: ", method); next }
  if (length(sort_cols)) d <- d[do.call(order, c(d[sort_cols], list(na.last = TRUE))), , drop = FALSE]
  mat <- as.matrix(data.frame(lapply(d[pc_cols], as.numeric), check.names = FALSE))
  if (anyNA(mat)) stop("Missing or nonnumeric PC correlations for method: ", method)
  rownames(mat) <- paste0("run", seq_len(nrow(d)))

  annotation <- NULL
  annotation_colors <- NULL
  if (length(annotation_cols)) {
    annotation <- d[annotation_cols]
    annotation_colors <- list()
    for (i in seq_along(annotation_cols)) {
      column <- annotation_cols[i]
      values <- d[[column]]
      levels <- if (is.numeric(values)) sort(unique(values)) else sort(unique(as.character(values)))
      annotation[[column]] <- factor(as.character(values), levels = as.character(levels))
      annotation_colors[[column]] <- setNames(palettes[[((i - 1) %% length(palettes)) + 1]](length(levels)), as.character(levels))
    }
    rownames(annotation) <- rownames(mat)
  }

  filename <- gsub("[^[:alnum:]_.-]+", "_", method)
  for (ext in c("png", "pdf")) {
    output <- file.path(output_dir, paste0(filename, "_pc_correlations.", ext))
    pheatmap(mat, cluster_rows = FALSE, cluster_cols = FALSE,
             show_rownames = FALSE, annotation_row = annotation,
             annotation_colors = annotation_colors,
             color = colorRampPalette(c("#4575b4", "#ffffbf", "#d73027"))(100),
             breaks = seq(0, 1, length.out = 101),
             main = paste(method, "PC-by-PC correlation"),
             fontsize = fontsize, fontsize_col = fontsize_col,
             filename = output, width = width, height = height)
    cat("Wrote:", output, "(", nrow(mat), "runs )\n")
  }
}
