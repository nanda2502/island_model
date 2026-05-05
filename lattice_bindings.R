load_island_lattice_bindings <- local({
  loaded <- FALSE

  function(rebuild = FALSE, verbose = FALSE) {
    if (loaded && !rebuild) {
      return(invisible(TRUE))
    }

    old_flags <- Sys.getenv("PKG_CXXFLAGS", unset = "")
    on.exit(Sys.setenv(PKG_CXXFLAGS = old_flags), add = TRUE)
    Sys.setenv(PKG_CXXFLAGS = paste(trimws(old_flags), "-std=gnu++20"))

    Rcpp::sourceCpp(file.path("src", "rcpp", "lattice_bindings.cpp"), rebuild = rebuild, verbose = verbose)
    loaded <<- TRUE
    invisible(TRUE)
  }
})

lattice_structure <- function(columns, layers, rebuild = FALSE, verbose = FALSE) {
  load_island_lattice_bindings(rebuild = rebuild, verbose = verbose)
  im_lattice_structure_cpp(as.integer(columns), as.integer(layers))
}

payoff_landscape <- function(columns,
                             layers,
                             island_count = 1,
                             delta,
                             sigma_b,
                             sigma_nu,
                             k,
                             seed,
                             rebuild = FALSE,
                             verbose = FALSE) {
  load_island_lattice_bindings(rebuild = rebuild, verbose = verbose)
  im_payoff_landscape_cpp(
    as.integer(columns),
    as.integer(layers),
    as.integer(island_count),
    as.numeric(delta),
    as.numeric(sigma_b),
    as.numeric(sigma_nu),
    as.numeric(k),
    as.numeric(seed)
  )
}

reachable_states <- function(columns, layers, rebuild = FALSE, verbose = FALSE) {
  load_island_lattice_bindings(rebuild = rebuild, verbose = verbose)
  im_reachable_states_cpp(as.integer(columns), as.integer(layers))
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
                                 rebuild = FALSE,
                                 verbose = FALSE) {
  payload <- payoff_landscape(
    columns = columns,
    layers = layers,
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

repertoire_list <- function(columns, layers, rebuild = FALSE, verbose = FALSE) {
  payload <- reachable_states(columns, layers, rebuild = rebuild, verbose = verbose)
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
 
