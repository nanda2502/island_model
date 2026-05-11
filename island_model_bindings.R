island_model_bindings_loaded <- FALSE

load_island_model_bindings <- function(rebuild = FALSE, verbose = FALSE) {
  if (island_model_bindings_loaded && !rebuild) {
    return(invisible(TRUE))
  }

  old_flags <- Sys.getenv("PKG_CXXFLAGS", unset = "")
  old_libs <- Sys.getenv("PKG_LIBS", unset = "")
  on.exit(Sys.setenv(PKG_CXXFLAGS = old_flags, PKG_LIBS = old_libs), add = TRUE)
  Sys.setenv(
    PKG_CXXFLAGS = paste(trimws(old_flags), "-std=gnu++20 -fopenmp"),
    PKG_LIBS = paste(trimws(old_libs), "-fopenmp")
  )

  Rcpp::sourceCpp(file.path("src", "rcpp", "island_model_bindings.cpp"), rebuild = rebuild, verbose = verbose)
  island_model_bindings_loaded <<- TRUE
  invisible(TRUE)
}

lattice_structure <- function(columns,
                              layers,
                              cross_column_depth = layers,
                              rebuild = FALSE,
                              verbose = FALSE) {
  load_island_model_bindings(rebuild = rebuild, verbose = verbose)
  im_lattice_structure_cpp(
    as.integer(columns),
    as.integer(layers),
    as.integer(cross_column_depth)
  )
}

payoff_landscape <- function(columns,
                             layers,
                             island_count = 1,
                             delta,
                             sigma_b,
                             sigma_nu,
                             k,
                             seed,
                             cross_column_depth = layers,
                             rebuild = FALSE,
                             verbose = FALSE) {
  load_island_model_bindings(rebuild = rebuild, verbose = verbose)
  im_payoff_landscape_cpp(
    as.integer(columns),
    as.integer(layers),
    as.integer(island_count),
    as.numeric(delta),
    as.numeric(sigma_b),
    as.numeric(sigma_nu),
    as.numeric(k),
    as.numeric(seed),
    as.integer(cross_column_depth)
  )
}

reachable_states <- function(columns,
                             layers,
                             cross_column_depth = layers,
                             rebuild = FALSE,
                             verbose = FALSE) {
  load_island_model_bindings(rebuild = rebuild, verbose = verbose)
  im_reachable_states_cpp(
    as.integer(columns),
    as.integer(layers),
    as.integer(cross_column_depth)
  )
}

run_island_model <- function(runs,
                             tolerance = 1e-10,
                             max_steps = 50000L,
                             bookkeeping_interval = 50L,
                             frequency_threshold = 1e-6,
                             write_equilibrium_distribution = TRUE,
                             threads = NULL,
                             rebuild = FALSE,
                             verbose = FALSE) {
  load_island_model_bindings(rebuild = rebuild, verbose = verbose)

  required_cols <- c(
    "seed", "columns", "layers", "island_count",
    "m", "rho", "mu", "alpha", "beta", "gamma", "eta",
    "delta", "sigma_b", "sigma_nu", "k"
  )
  missing_cols <- setdiff(required_cols, names(runs))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  runs <- as.data.frame(runs, stringsAsFactors = FALSE)
  runs$seed <- as.numeric(runs$seed)
  runs$columns <- as.integer(runs$columns)
  runs$layers <- as.integer(runs$layers)
  if (!("cross_column_depth" %in% names(runs))) {
    runs$cross_column_depth <- runs$layers
  }
  runs$cross_column_depth <- as.integer(runs$cross_column_depth)
  runs$island_count <- as.integer(runs$island_count)

  numeric_cols <- c("m", "rho", "mu", "alpha", "beta", "gamma", "eta", "delta", "sigma_b", "sigma_nu", "k")
  for (col in numeric_cols) {
    runs[[col]] <- as.numeric(runs[[col]])
  }

  im_run_model_cpp(
    runs = runs,
    tolerance = as.numeric(tolerance),
    max_steps = as.integer(max_steps),
    bookkeeping_interval = as.integer(bookkeeping_interval),
    frequency_threshold = as.numeric(frequency_threshold),
    include_equilibrium_distribution = isTRUE(write_equilibrium_distribution),
    threads = if (is.null(threads)) NULL else as.integer(threads)
  )
}

lattice_payoff_graph <- function(columns,
                                 layers,
                                 island_count = 1,
                                 delta,
                                 sigma_b,
                                 sigma_nu,
                                 k,
                                 seed,
                                 island = 1,
                                 cross_column_depth = layers,
                                 rebuild = FALSE,
                                 verbose = FALSE) {
  payload <- payoff_landscape(
    columns = columns,
    layers = layers,
    cross_column_depth = cross_column_depth,
    island_count = island_count,
    delta = delta,
    sigma_b = sigma_b,
    sigma_nu = sigma_nu,
    k = k,
    seed = seed,
    rebuild = rebuild,
    verbose = verbose
  )

  vertices <- transform(payload$vertices, payoff = as.numeric(payload$payoffs[as.integer(island), ]))
  igraph::graph_from_data_frame(payload$edges, directed = TRUE, vertices = vertices)
}

lattice_layout <- function(graph) {
  vertices <- igraph::vertex_attr(graph)
  cbind(
    x = as.numeric(vertices$column),
    y = -as.numeric(vertices$layer)
  )
}

repertoire_list <- function(columns,
                            layers,
                            cross_column_depth = layers,
                            rebuild = FALSE,
                            verbose = FALSE) {
  payload <- reachable_states(
    columns,
    layers,
    cross_column_depth = cross_column_depth,
    rebuild = rebuild,
    verbose = verbose
  )
  raw_matrix <- payload$state_payload$repertoire_raw
  trait_count <- payload$trait_count

  lapply(seq_len(payload$state_count), function(state_idx) {
    bits <- rawToBits(raw_matrix[state_idx, ])
    which(as.logical(bits[seq_len(trait_count)])) - 1L
  })
}



plot_graph <- function(graph, attr_v = "name", ...) {
    old_mar <- par("mar")
    par(mar = c(1, 1, 1, 1))
    V(graph)$color <- rep("black", vcount(graph))
    plot(graph,
         layout = lattice_layout(graph),
         vertex.label = if (attr_v == "name") V(graph) else round(V(graph)$payoff, 2),
         vertex.label.cex = 0.7,   
         vertex.size = 14,
         vertex.label.color = "white",
         #edge.label = round(E(graph)$weight,1),
         edge.label.cex = 0.5,
         edge.arrow.size = 0.3,
         ...
    )
    par(mar = old_mar)
}

trait_equilibrium_payoff_landscape <- function(trait_df, rebuild = FALSE, verbose = FALSE) {
  required_cols <- c("seed", "columns", "layers", "island_count", "delta", "sigma_b", "sigma_nu", "k")
  missing_cols <- setdiff(required_cols, names(trait_df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns for trait payoffs: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  payoff_landscape(
    columns = as.integer(unique(trait_df$columns)),
    layers = as.integer(unique(trait_df$layers)),
    cross_column_depth = if ("cross_column_depth" %in% names(trait_df)) {
      as.integer(unique(trait_df$cross_column_depth))
    } else {
      as.integer(unique(trait_df$layers))
    },
    island_count = as.integer(unique(trait_df$island_count)),
    delta = as.numeric(unique(trait_df$delta)),
    sigma_b = as.numeric(unique(trait_df$sigma_b)),
    sigma_nu = as.numeric(unique(trait_df$sigma_nu)),
    k = as.numeric(unique(trait_df$k)),
    seed = as.numeric(unique(trait_df$seed)),
    rebuild = rebuild,
    verbose = verbose
  )
}

attach_trait_payoff_labels <- function(vertices, payoffs, island = NULL) {
  if (is.null(island)) {
    payoff_values <- colMeans(payoffs)
  } else {
    island_row <- as.integer(island) + 1L
    if (island_row < 1L || island_row > nrow(payoffs)) {
      stop("island is outside the payoff matrix", call. = FALSE)
    }
    payoff_values <- as.numeric(payoffs[island_row, ])
  }

  vertices$payoff <- payoff_values[as.integer(vertices$trait_id) + 1L]
  vertices$label <- format(round(vertices$payoff, 2), trim = TRUE)
  vertices
}

plot_trait_graph <- function(graph, main, ...) {
  layout <- lattice_layout(graph)
  label_layout <- layout
  label_layout[, "x"] <- label_layout[, "x"] - 1 / 3

  plot(
    graph,
    layout = layout,
    vertex.size = igraph::V(graph)$size,
    vertex.label = NA,
    edge.arrow.size = 0.3,
    main = main,
    rescale = FALSE,
    xlim = range(c(layout[, "x"], label_layout[, "x"])),
    ylim = range(layout[, "y"]),
    ...
  )

  text(
    label_layout[, "x"],
    label_layout[, "y"],
    labels = igraph::V(graph)$label,
    col = igraph::V(graph)$label.color,
    cex = igraph::V(graph)$label.cex
  )
}

plot_trait_equilibrium_graph <- function(trait_df,
                                         vertex_size_range = c(8, 28),
                                         rebuild = FALSE,
                                         verbose = FALSE,
                                         ...) {
  trait_df <- as.data.frame(trait_df, stringsAsFactors = FALSE)

  required_cols <- c(
    "run_id", "columns", "layers", "trait_id",
    "trait_column", "trait_layer", "frequency", "island"
  )
  missing_cols <- setdiff(required_cols, names(trait_df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  if (nrow(trait_df) == 0) {
    stop("trait_df must contain at least one row", call. = FALSE)
  }

  if (length(vertex_size_range) != 2 || any(!is.finite(vertex_size_range))) {
    stop("vertex_size_range must contain two finite values", call. = FALSE)
  }

  if (vertex_size_range[1] <= 0 || vertex_size_range[2] <= 0) {
    stop("vertex_size_range values must be positive", call. = FALSE)
  }

  if (vertex_size_range[1] > vertex_size_range[2]) {
    stop("vertex_size_range must be in increasing order", call. = FALSE)
  }

  single_run_cols <- c(
    "run_id", "seed", "columns", "layers", "cross_column_depth",
    "island_count", "delta", "sigma_b", "sigma_nu", "k"
  )
  varying_cols <- single_run_cols[vapply(single_run_cols, function(col) {
    col %in% names(trait_df) && length(unique(trait_df[[col]])) != 1L
  }, logical(1))]
  if (length(varying_cols) > 0) {
    stop(
      "trait_df must contain exactly one run; these columns vary: ",
      paste(varying_cols, collapse = ", "),
      call. = FALSE
    )
  }

  columns <- as.integer(unique(trait_df$columns))
  layers <- as.integer(unique(trait_df$layers))
  cross_column_depth <- if ("cross_column_depth" %in% names(trait_df)) {
    as.integer(unique(trait_df$cross_column_depth))
  } else {
    layers
  }

  payload <- lattice_structure(
    columns = columns,
    layers = layers,
    cross_column_depth = cross_column_depth,
    rebuild = rebuild,
    verbose = verbose
  )
  vertices <- payload$vertices
  edges <- payload$edges
  payoff_payload <- trait_equilibrium_payoff_landscape(trait_df, rebuild = rebuild, verbose = verbose)

  mean_frequency <- stats::aggregate(
    frequency ~ trait_id + trait_column + trait_layer,
    data = trait_df,
    FUN = mean
  )

  vertices <- merge(
    vertices,
    mean_frequency,
    by = "trait_id",
    all.x = TRUE,
    sort = FALSE
  )
  vertices$frequency[is.na(vertices$frequency)] <- 0

  vertices <- vertices[order(vertices$trait_id), ]
  vertices <- attach_trait_payoff_labels(vertices, payoff_payload$payoffs)

  freq_min <- min(vertices$frequency)
  freq_max <- max(vertices$frequency)
  if (isTRUE(all.equal(freq_min, freq_max))) {
    vertex_size <- rep(mean(vertex_size_range), nrow(vertices))
  } else {
    vertex_size <- vertex_size_range[1] +
      (vertices$frequency - freq_min) / (freq_max - freq_min) *
      diff(vertex_size_range)
  }

  graph <- igraph::graph_from_data_frame(edges, directed = TRUE, vertices = vertices)
  igraph::V(graph)$size <- vertex_size
  igraph::V(graph)$color <- "black"
  igraph::V(graph)$label.color <- "black"
  igraph::V(graph)$label.cex <- 0.7

  old_mar <- par("mar")
  on.exit(par(mar = old_mar), add = TRUE)
  par(mar = c(1, 1, 2, 1))

  plot_trait_graph(
    graph,
    main = sprintf(
      "Run %s: mean trait frequency across islands",
      as.character(unique(trait_df$run_id))
    ),
    ...
  )

  invisible(graph)
}

plot_trait_equilibrium_graph_by_island <- function(trait_df,
                                                   n = NULL,
                                                   vertex_size_range = c(8, 28),
                                                   rebuild = FALSE,
                                                   verbose = FALSE,
                                                   ...) {
  trait_df <- as.data.frame(trait_df, stringsAsFactors = FALSE)

  required_cols <- c(
    "run_id", "columns", "layers", "trait_id",
    "trait_column", "trait_layer", "frequency", "island"
  )
  missing_cols <- setdiff(required_cols, names(trait_df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  if (nrow(trait_df) == 0) {
    stop("trait_df must contain at least one row", call. = FALSE)
  }

  if (length(vertex_size_range) != 2 || any(!is.finite(vertex_size_range))) {
    stop("vertex_size_range must contain two finite values", call. = FALSE)
  }

  if (vertex_size_range[1] <= 0 || vertex_size_range[2] <= 0) {
    stop("vertex_size_range values must be positive", call. = FALSE)
  }

  if (vertex_size_range[1] > vertex_size_range[2]) {
    stop("vertex_size_range must be in increasing order", call. = FALSE)
  }

  single_run_cols <- c(
    "run_id", "seed", "columns", "layers", "cross_column_depth",
    "island_count", "delta", "sigma_b", "sigma_nu", "k"
  )
  varying_cols <- single_run_cols[vapply(single_run_cols, function(col) {
    col %in% names(trait_df) && length(unique(trait_df[[col]])) != 1L
  }, logical(1))]
  if (length(varying_cols) > 0) {
    stop(
      "trait_df must contain exactly one run; these columns vary: ",
      paste(varying_cols, collapse = ", "),
      call. = FALSE
    )
  }

  island_ids <- sort(unique(as.integer(trait_df$island)))
  if (length(island_ids) == 0) {
    stop("trait_df must contain at least one island", call. = FALSE)
  }

  if (is.null(n)) {
    n <- length(island_ids)
  }
  if (length(n) != 1 || !is.finite(n) || n <= 0) {
    stop("n must be a single positive number", call. = FALSE)
  }

  n <- min(as.integer(n), length(island_ids))
  island_ids <- island_ids[seq_len(n)]

  columns <- as.integer(unique(trait_df$columns))
  layers <- as.integer(unique(trait_df$layers))
  cross_column_depth <- if ("cross_column_depth" %in% names(trait_df)) {
    as.integer(unique(trait_df$cross_column_depth))
  } else {
    layers
  }
  payload <- lattice_structure(
    columns = columns,
    layers = layers,
    cross_column_depth = cross_column_depth,
    rebuild = rebuild,
    verbose = verbose
  )
  base_vertices <- payload$vertices
  edges <- payload$edges
  payoff_payload <- trait_equilibrium_payoff_landscape(trait_df, rebuild = rebuild, verbose = verbose)

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  panel_cols <- ceiling(sqrt(n))
  panel_rows <- ceiling(n / panel_cols)
  par(mfrow = c(panel_rows, panel_cols), mar = c(0, 0, 1.1, 0), oma = c(0, 0, 0, 0))

  graphs <- vector("list", length(island_ids))
  names(graphs) <- paste0("island_", island_ids)

  for (idx in seq_along(island_ids)) {
    island_id <- island_ids[idx]
    island_df <- trait_df[as.integer(trait_df$island) == island_id, , drop = FALSE]

    vertices <- merge(
      base_vertices,
      island_df[, c("trait_id", "frequency")],
      by = "trait_id",
      all.x = TRUE,
      sort = FALSE
    )
    vertices$frequency[is.na(vertices$frequency)] <- 0
    vertices <- vertices[order(vertices$trait_id), ]
    vertices <- attach_trait_payoff_labels(vertices, payoff_payload$payoffs, island = island_id)

    freq_min <- min(vertices$frequency)
    freq_max <- max(vertices$frequency)
    if (isTRUE(all.equal(freq_min, freq_max))) {
      vertex_size <- rep(mean(vertex_size_range), nrow(vertices))
    } else {
      vertex_size <- vertex_size_range[1] +
        (vertices$frequency - freq_min) / (freq_max - freq_min) *
        diff(vertex_size_range)
    }

    graph <- igraph::graph_from_data_frame(edges, directed = TRUE, vertices = vertices)
    igraph::V(graph)$size <- vertex_size
    igraph::V(graph)$color <- "black"
    igraph::V(graph)$label.color <- "black"
    igraph::V(graph)$label.cex <- 0.7

    plot_trait_graph(
      graph,
      main = sprintf("Run %s, island %s", as.character(unique(trait_df$run_id)), island_id),
      ...
    )

    graphs[[idx]] <- graph
  }

  invisible(graphs)
}



if (FALSE) {
  test_grid_repertoires <- repertoire_list(
    columns = 1,
    layers = 7
  )
  test_grid_graph <- lattice_payoff_graph(
    columns = 2,
    layers = 7,
    island_count = 1,
    delta = 0.2,
    sigma_b = 0.5,
    sigma_nu = 0.1,
    k = 0.0,
    seed = 12345,
    island = 1
  )

  plot_graph(test_grid_graph, "payoff")
}
 
