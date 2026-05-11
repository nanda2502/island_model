if (rstudioapi::isAvailable()) setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)
library(igraph)
library(Rcpp)

source("island_model_bindings.R")
source("island_model_plotting.R")

##### run #####

runs <- expand.grid(
    "seed" = c(12345),                        
    "columns" = c(4),                  
    "layers" = c(5),
    "cross_column_depth" = c(1),       # max layer at which there should be incoming diagonal edges (1 for none) 
    "strictness" = c(0,1),             # 0 for unrestricted learning, 1 for restricted by structure 
    "island_count" = c(10),
    "m" = c(0.01, 0.1, 0.3),           # migration rate (0 for no migration, 1 for complete mixing)
    "rho" = c(0.01),                   # reset rate 
    "mu" = c(0.05),                    # innovation rate
    "alpha" = c(1),                    # frequency bias (1 for random copying, >1 for conformity, <1 for anti-conformity)
    "beta" = c(1),                     # payoff expression bias (0 for no payoff bias, >0 for more bias towards expressing higher payoff traits)
    "gamma" = c(0),                    # payoff bias (0 for no payoff bias, >0 for more bias towards copying higher payoff traits)
    "lambda" = c(0),                   # prestige bias (0 for no prestige bias, >0 for more bias towards copying demonstrators with higher total payoff)
    "eta" = c(1),                      # transparency (0 for opaque prerequisite structure, 1 for fully transparent; only accessible traits are attempted)
    "delta" = c(4),                    # payoff scaling by layer (0 for no scaling, >0 for higher layers having higher payoffs)
    "sigma_b" = c(0.25),               # payoff variability across columns (0 for no variability, >0 for more variability)
    "sigma_nu" = c(0.0),               # payoff variability across traits (0 for no variability, >0 for more variability)
    "k" = c(0.00)                      # payoff correlation across islands (0 for no correlation, 1 for perfect correlation
) 

res <- run_island_model(runs, threads = 15)

saveRDS(res, "res.rds")



#res <- readRDS("res.rds")

time <- read.csv("time_bookkeeping.csv", header = T)

##### plot #####

plot_trait_equilibrium_graph_by_island(subset(res$traits, run_id == 0), n = 10)

plot_dv_by_time_m(res$time, "cultural_divergence", "strictness", 1)

plot_dv_by_time_m(res$time, "adaptive_divergence")


df1 <- subset(res$equilibrium, strictness == 1)
df2 <- subset(res$equilibrium, strictness == 0)



tapply(df1$cultural_divergence_run, df1$m, max)
tapply(df2$cultural_divergence_run, df2$m, max)


p1 <- res$time %>%
    filter(
        columns == 1,
        !(alpha == 2 & gamma == 2)) %>%
    plot_dv_by_time(
        "mean_payoff",
        "Mean Payoff",
        "alpha",
        "Frequency bias",
        "gamma",
        "Payoff bias"
        )

p2 <- res$time %>%
    filter(
        layers == 1,
        !(alpha == 2 & gamma == 2)) %>%
    plot_dv_by_time(
        "mean_payoff",
        "Mean Payoff",
        "alpha",
        "Frequency bias",
        "gamma",
        "Payoff bias"
    )
traits <- res$traits

##### lattice #####
test_grid_graph <- lattice_payoff_graph(
    columns = 4,
    layers = 6,
    island_count = 10,
    delta = 8,
    sigma_b = 1.25,
    sigma_nu = 0.0,
    k = 0.0,
    seed = 12345,
    island = 1
)

plot_graph(test_grid_graph, "payoff")

tapply(V(test_grid_graph)$payoff, V(test_grid_graph)$column, mean)


##### load #####
df <- read_csv("equilibrium_distribution.csv", show_col_types = FALSE)

##### clean #####
df <- df %>%
  mutate(
    frequency = as.numeric(frequency),
    repertoire_size = as.numeric(repertoire_size),
    repertoire_payoff_sum = as.numeric(repertoire_payoff_sum),
    repertoire_payoff_mean = as.numeric(repertoire_payoff_mean),
    repertoire_payoff_max = as.numeric(repertoire_payoff_max),
    max_layer = as.numeric(max_layer)
  ) %>%
  filter(is.finite(frequency), frequency > 0)

##### helpers #####
weighted_var <- function(x, w) {
  m <- weighted.mean(x, w)
  sum(w * (x - m)^2) / sum(w)
}

effective_support <- function(p) {
  p <- p[p > 0]
  exp(-sum(p * log(p)))
}

top_mass <- function(p, k) {
  sum(sort(p, decreasing = TRUE)[seq_len(min(k, length(p)))])
}

safe_cor <- function(x, y, method = "pearson") {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  if (length(unique(x)) < 2 || length(unique(y)) < 2) return(NA_real_)
  cor(x, y, method = method)
}

column_distance <- function(a, b, columns) {
  direct <- abs(a - b)
  min(direct, columns - direct)
}

make_lattice_payoffs <- function(columns,
                                 layers,
                                 island_count = 1,
                                 delta,
                                 sigma_b,
                                 sigma_nu,
                                 k,
                                 seed = NULL) {
  stopifnot(columns > 0, layers > 0, island_count > 0)
  stopifnot(sigma_b >= 0, sigma_nu >= 0, k >= 0, k <= 1)

  if (!is.null(seed)) {
    old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (old_seed_exists) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    }
    on.exit({
      if (old_seed_exists) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  shared_scale <- sigma_b * sqrt(k)
  residual_scale <- sigma_b * sqrt(1 - k)
  payoffs <- matrix(0, nrow = island_count, ncol = columns * layers)
  column_effect <- matrix(0, nrow = island_count, ncol = columns)

  shared <- rnorm(columns)
  for (island_idx in seq_len(island_count)) {
    residual <- rnorm(columns)
    column_effect[island_idx, ] <- shared_scale * shared + residual_scale * residual
  }

  for (layer in seq_len(layers) - 1L) {
    for (island_idx in seq_len(island_count)) {
      for (column in seq_len(columns) - 1L) {
        trait_id <- layer * columns + column + 1L
        mean_value <- delta * (layer + 1)
        nu <- sigma_nu * rnorm(1)
        payoffs[island_idx, trait_id] <- exp(mean_value + column_effect[island_idx, column + 1L] + nu)
      }
    }
  }

  payoffs
}

make_lattice_payoff_graph <- function(columns,
                                      layers,
                                      island_count = 1,
                                      delta,
                                      sigma_b,
                                      sigma_nu,
                                      k,
                                      seed = NULL,
                                      island = 1,
                                      cross_column_depth = layers) {
  stopifnot(island >= 1, island <= island_count)

  payoffs <- make_lattice_payoffs(
    columns = columns,
    layers = layers,
    island_count = island_count,
    delta = delta,
    sigma_b = sigma_b,
    sigma_nu = sigma_nu,
    k = k,
    seed = seed
  )

  vertices <- tibble(
    trait_id = seq_len(columns * layers) - 1L,
    column = rep(seq_len(columns) - 1L, times = layers),
    layer = rep(seq_len(layers) - 1L, each = columns),
    payoff = as.numeric(payoffs[island, ])
  )

  edges <- map_dfr(seq_len(nrow(vertices)), function(idx) {
    v <- vertices[idx, ]

    if (v$layer == 0L) {
      return(tibble(from = integer(), to = integer()))
    }

    parent_layer <- v$layer - 1L
    parent_columns <- if (v$layer + 1L <= cross_column_depth) {
      c((v$column - 1L) %% columns, v$column, (v$column + 1L) %% columns)
    } else {
      v$column
    }

    tibble(
      from = parent_layer * columns + parent_columns,
      to = rep(v$trait_id, length(parent_columns))
    )
  })

  graph_from_data_frame(edges, directed = TRUE, vertices = vertices)
}

island_lattice_bindings_loaded <- FALSE

load_island_lattice_bindings <- function(rebuild = FALSE, verbose = FALSE) {
  if (island_lattice_bindings_loaded && !rebuild) {
    return(invisible(TRUE))
  }

  binding_file <- file.path("src", "rcpp", "lattice_bindings.cpp")
  old_flags <- Sys.getenv("PKG_CXXFLAGS", unset = "")
  on.exit(Sys.setenv(PKG_CXXFLAGS = old_flags), add = TRUE)
  Sys.setenv(PKG_CXXFLAGS = paste(old_flags, "-std=gnu++20"))

  Rcpp::sourceCpp(binding_file, rebuild = rebuild, verbose = verbose)
  island_lattice_bindings_loaded <<- TRUE
  invisible(TRUE)
}

make_lattice_payoff_graph_cpp <- function(columns,
                                          layers,
                                          island_count = 1,
                                          delta,
                                          sigma_b,
                                          sigma_nu,
                                          k,
                                          seed,
                                          island = 1,
                                          cross_column_depth = layers,
                                          rebuild_bindings = FALSE) {
  stopifnot(island >= 1, island <= island_count)

  load_island_lattice_bindings(rebuild = rebuild_bindings)

  payload <- im_payoff_landscape_cpp(
    columns = columns,
    layers = layers,
    island_count = island_count,
    delta = delta,
    sigma_b = sigma_b,
    sigma_nu = sigma_nu,
    k = k,
    seed = seed,
    cross_column_depth = cross_column_depth
  )

  vertices <- payload$vertices %>%
    mutate(payoff = as.numeric(payload$payoffs[island, ]))

  graph_from_data_frame(payload$edges, directed = TRUE, vertices = vertices)
}

##### per run/island summaries #####
run_island_summary <- df %>%
  group_by(run_id, island) %>%
  summarise(
    mass = sum(frequency),
    support_size = n(),
    effective_support = effective_support(frequency),
    entropy = -sum(frequency * log(frequency)),
    top1_mass = top_mass(frequency, 1),
    top5_mass = top_mass(frequency, 5),
    top10_mass = top_mass(frequency, 10),
    empty_mass = sum(frequency[repertoire_size == 0]),
    mean_repertoire_size = weighted.mean(repertoire_size, frequency),
    var_repertoire_size = weighted_var(repertoire_size, frequency),
    mean_max_layer = weighted.mean(max_layer, frequency),
    var_max_layer = weighted_var(max_layer, frequency),
    mean_payoff_sum = weighted.mean(repertoire_payoff_sum, frequency),
    mean_payoff_mean = weighted.mean(repertoire_payoff_mean, frequency),
    mean_payoff_max = weighted.mean(repertoire_payoff_max, frequency),
    .groups = "drop"
  )

print(run_island_summary, width=Inf)

##### filtered payoff-frequency correlations #####
corr_summary <- df %>%
  group_by(run_id, island) %>%
  group_modify(~{
    d <- .x
    d_filt <- d %>% filter(frequency > 1e-6)

    tibble(
      pearson_sum_all   = safe_cor(log10(d$frequency), d$repertoire_payoff_sum, "pearson"),
      spearman_sum_all  = safe_cor(d$frequency, d$repertoire_payoff_sum, "spearman"),
      pearson_mean_all  = safe_cor(log10(d$frequency), d$repertoire_payoff_mean, "pearson"),
      spearman_mean_all = safe_cor(d$frequency, d$repertoire_payoff_mean, "spearman"),
      pearson_max_all   = safe_cor(log10(d$frequency), d$repertoire_payoff_max, "pearson"),
      spearman_max_all  = safe_cor(d$frequency, d$repertoire_payoff_max, "spearman"),

      pearson_sum_filt   = safe_cor(log10(d_filt$frequency), d_filt$repertoire_payoff_sum, "pearson"),
      spearman_sum_filt  = safe_cor(d_filt$frequency, d_filt$repertoire_payoff_sum, "spearman"),
      pearson_mean_filt  = safe_cor(log10(d_filt$frequency), d_filt$repertoire_payoff_mean, "pearson"),
      spearman_mean_filt = safe_cor(d_filt$frequency, d_filt$repertoire_payoff_mean, "spearman"),
      pearson_max_filt   = safe_cor(log10(d_filt$frequency), d_filt$repertoire_payoff_max, "pearson"),
      spearman_max_filt  = safe_cor(d_filt$frequency, d_filt$repertoire_payoff_max, "spearman")
    )
  }) %>%
  ungroup()

print(corr_summary, width = Inf)

##### mass by repertoire size #####
mass_by_size <- df %>%
  group_by(run_id, island, repertoire_size) %>%
  summarise(mass = sum(frequency), .groups = "drop")

ggplot(mass_by_size, aes(repertoire_size, mass)) +
  geom_col() +
  facet_grid(run_id ~ island, scales = "free_y") +
  labs(title = "Equilibrium mass by repertoire size")

##### mass by max layer #####
mass_by_layer <- df %>%
  group_by(run_id, island, max_layer) %>%
  summarise(mass = sum(frequency), .groups = "drop")

ggplot(mass_by_layer, aes(max_layer, mass)) +
  geom_col() +
  facet_grid(run_id ~ island, scales = "free_y") +
  labs(title = "Equilibrium mass by max layer")

##### top repertoires #####
top_reps <- df %>%
  group_by(run_id, island) %>%
  slice_max(order_by = frequency, n = 25, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(run_id, island, desc(frequency))

ggplot(top_reps, aes(reorder(as.factor(repertoire_id), frequency), frequency)) +
  geom_col() +
  coord_flip() +
  facet_grid(run_id ~ island, scales = "free_y") +
  labs(title = "Top 25 repertoire frequencies", x = "repertoire_id")

##### payoff vs frequency #####
df_plot <- df %>%
  filter(frequency > 1e-6)

ggplot(df_plot, aes(repertoire_payoff_sum, log10(frequency), color = repertoire_size)) +
  geom_point(alpha = 0.5) +
  facet_grid(run_id ~ island) +
  labs(title = "log10(frequency) vs repertoire payoff sum")

ggplot(df_plot, aes(max_layer, log10(frequency), color = repertoire_size)) +
  geom_jitter(alpha = 0.4, width = 0.15, height = 0) +
  facet_grid(run_id ~ island) +
  labs(title = "log10(frequency) vs max layer")

##### simple regression diagnostic #####
regression_summary <- df %>%
  filter(frequency > 1e-6) %>%
  group_by(run_id, island) %>%
  group_modify(~{
    d <- .x
    fit <- lm(log10(frequency) ~ repertoire_payoff_sum + repertoire_size + max_layer, data = d)
    broom::tidy(fit)
  }) %>%
  ungroup()

print(regression_summary)

##### time bookkeeping #####
time_df <- read_csv("time_bookkeeping.csv", show_col_types = FALSE) %>%
  mutate(
    step = as.numeric(step),
    adaptive_divergence = as.numeric(adaptive_divergence),
    cultural_divergence = as.numeric(cultural_divergence),
    divergence_pair_count = as.numeric(divergence_pair_count),
    mean_payoff = as.numeric(mean_payoff),
    adj_payoff = as.numeric(adj_payoff),
    mean_max_depth = as.numeric(mean_max_depth),
    mean_depth = as.numeric(mean_depth),
    eff_column = as.numeric(eff_column),
    top_col_mass = as.numeric(top_col_mass),
    mean_rep_size = as.numeric(mean_rep_size),
    empty_rep_size = as.numeric(empty_rep_size)
  )

time_vars <- c(
  "adaptive_divergence",
  "cultural_divergence",
  "divergence_pair_count",
  "mean_payoff",
  "adj_payoff",
  "mean_max_depth",
  "mean_depth",
  "eff_column",
  "top_col_mass",
  "mean_rep_size",
  "empty_rep_size"
)

time_labels <- c(
  adaptive_divergence = "Adaptive Divergence",
  cultural_divergence = "Cultural Divergence",
  divergence_pair_count = "Island Pair Count",
  mean_payoff = "Mean Payoff",
  adj_payoff = "Adjusted Payoff",
  mean_max_depth = "Mean Max Depth",
  mean_depth = "Mean Depth",
  eff_column = "Effective Columns",
  top_col_mass = "Top Column Mass",
  mean_rep_size = "Mean Repertoire Size",
  empty_rep_size = "Empty Repertoire Mass"
)

time_long <- time_df %>%
  pivot_longer(
    cols = all_of(time_vars),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(metric = factor(metric, levels = time_vars, labels = time_labels[time_vars]))

ggplot(time_long, aes(step, value, group = interaction(run_id, seed))) +
  geom_line(linewidth = 0.7, alpha = 0.9, color = "#1b6ca8") +
  geom_point(size = 1.2, alpha = 0.9, color = "#1b6ca8") +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  labs(
    x = "Timestep",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )


library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(forcats)
library(rlang)

##### TIME ANALYSIS #####
df <- read_csv("time_bookkeeping.csv", show_col_types = FALSE)

parameter_cols <- c(
    "columns", "layers", "island_count", "m",
    "rho", "mu", "alpha", "beta", "gamma", "eta",
    "delta", "sigma_b", "sigma_nu", "k", "seed"
)

parameter_labels <- c(
    columns = "Columns",
    layers = "Layers",
    island_count = "Islands",
    m = "Migration",
    rho = "rho",
    mu = "mu",
    alpha = "alpha",
    beta = "beta",
    gamma = "gamma",
    eta = "eta",
    delta = "delta",
    sigma_b = "sigma_b",
    sigma_nu = "sigma_nu",
    k = "k",
    seed = "Seed"
)

present_parameter_cols <- intersect(parameter_cols, names(df))

label_parameter <- function(param) {
    if (param %in% names(parameter_labels)) parameter_labels[[param]] else param
}

format_parameter_values <- function(values) {
    values_chr <- as.character(sort(unique(values)))
    paste(values_chr, collapse = ", ")
}

varied_parameters <- function(data, exclude = character()) {
    candidates <- setdiff(intersect(parameter_cols, names(data)), exclude)
    candidates[vapply(candidates, function(param) dplyr::n_distinct(data[[param]], na.rm = TRUE) > 1, logical(1))]
}

fixed_parameters <- function(data, exclude = character()) {
    candidates <- setdiff(intersect(parameter_cols, names(data)), exclude)
    candidates[vapply(candidates, function(param) dplyr::n_distinct(data[[param]], na.rm = TRUE) <= 1, logical(1))]
}

build_plot_annotations <- function(data, shown_params = character()) {
    omitted_varied <- setdiff(varied_parameters(data), shown_params)
    fixed <- fixed_parameters(data, exclude = shown_params)

    subtitle <- if (length(omitted_varied) > 0) {
        paste("Also varied:", paste(vapply(omitted_varied, label_parameter, character(1)), collapse = ", "))
    } else {
        NULL
    }

    caption <- if (length(fixed) > 0) {
        paste(
            "Fixed:",
            paste(
                vapply(
                    fixed,
                    function(param) paste0(label_parameter(param), "=", format_parameter_values(data[[param]])),
                    character(1)
                ),
                collapse = " | "
            )
        )
    } else {
        NULL
    }

    list(subtitle = subtitle, caption = caption)
}

resolve_parameter_facet_params <- function(
    data,
    row_params = character(),
    col_params = character(),
    exclude = character()
) {
    extra_params <- varied_parameters(data, exclude = c(row_params, col_params, exclude))

    if (length(extra_params) > 0) {
        split_idx <- ceiling(length(extra_params) / 2)
        extra_rows <- extra_params[seq_len(split_idx)]
        extra_cols <- if (split_idx < length(extra_params)) extra_params[(split_idx + 1):length(extra_params)] else character()

        row_params <- c(row_params, extra_rows)
        col_params <- c(col_params, extra_cols)
    }

    list(
        row_params = row_params,
        col_params = col_params,
        facet_params = c(row_params, col_params)
    )
}

build_parameter_labeller <- function(data, params) {
    param_labellers <- setNames(
        lapply(
            params,
            function(param) {
                values <- as.character(sort(unique(data[[param]])))
                setNames(
                    paste0(label_parameter(param), " = ", values),
                    values
                )
            }
        ),
        params
    )

    do.call(labeller, param_labellers)
}

build_parameter_facet <- function(
    data,
    row_params = character(),
    col_params = character(),
    exclude = character()
) {
    facet_params <- resolve_parameter_facet_params(
        data,
        row_params = row_params,
        col_params = col_params,
        exclude = exclude
    )

    row_params <- facet_params$row_params
    col_params <- facet_params$col_params

    if (length(row_params) == 0 && length(col_params) == 0) {
        return(NULL)
    }

    facet_labeller <- build_parameter_labeller(data, c(row_params, col_params))

    if (length(row_params) == 0) {
        return(facet_grid(cols = vars(!!!syms(col_params)), labeller = facet_labeller))
    }

    if (length(col_params) == 0) {
        return(facet_grid(rows = vars(!!!syms(row_params)), labeller = facet_labeller))
    }

    facet_grid(
        rows = vars(!!!syms(row_params)),
        cols = vars(!!!syms(col_params)),
        labeller = facet_labeller
    )
}

##### basic cleanup #####
df <- df %>%
    mutate(
        run_id = as.integer(run_id),
        step = as.integer(step),
    adaptive_divergence = as.numeric(adaptive_divergence),
    cultural_divergence = as.numeric(cultural_divergence),
    divergence_pair_count = as.numeric(divergence_pair_count),
        across(any_of(present_parameter_cols), as.factor)
    )

# sanity check
mass_check <- df %>%
    group_by(run_id) %>%
    summarise(
        max_step = max(step),
        converged = first(converged),
        steps_to_equilibrium = first(steps_to_equilibrium),
        .groups = "drop"
    )

print(mass_check, n = 20)

##### final-row dataset #####
final_df <- df %>%
    group_by(run_id) %>%
    slice_max(step, n = 1, with_ties = FALSE) %>%
    ungroup()

##### helper: heatmap plot #####
plot_final_heatmap <- function(data, metric, title = metric) {
    facet_params <- resolve_parameter_facet_params(data, exclude = c("mu", "rho"))
    annotations <- build_plot_annotations(
        data,
        shown_params = c("mu", "rho", facet_params$facet_params)
    )

    base_plot <- ggplot(data, aes(x = mu, y = rho, fill = .data[[metric]])) +
        geom_tile() +
        build_parameter_facet(data = data, exclude = c("mu", "rho")) +
        labs(
            title = title,
            subtitle = annotations$subtitle,
            x = "mu",
            y = "rho",
            fill = metric,
            caption = annotations$caption
        ) +
        theme_minimal()

    base_plot
}

##### helper: trajectory plot #####
# filter to one beta at a time
plot_trajectory <- function(data, metric, rho_value) {
    plot_data <- data %>% filter(rho == rho_value)
    facet_params <- resolve_parameter_facet_params(
        plot_data,
        row_params = "beta",
        col_params = "mu",
        exclude = c("step", "rho", "alpha", "gamma", metric, "run_id")
    )
    annotations <- build_plot_annotations(
        plot_data,
        shown_params = c("step", "beta", "alpha", "gamma", facet_params$facet_params)
    )

    base_plot <- ggplot(plot_data, aes(
        x = step,
        y = .data[[metric]],
        group = run_id,
        color = alpha,
        linetype = gamma
    )) +
        geom_line(alpha = 0.85) +
        build_parameter_facet(
            data = plot_data,
            row_params = "rho",
            col_params = "mu",
            exclude = c("step", "beta", "alpha", "gamma", metric, "run_id")
        ) +
        labs(
            title = paste(metric, "over time | beta =", rho_value),
            subtitle = annotations$subtitle,
            x = "step",
            y = metric,
            color = "alpha",
            linetype = "gamma",
            caption = annotations$caption
        ) +
        theme_minimal()

    base_plot
}

##### helper: endpoint scatter #####
plot_endpoint_scatter <- function(data, xvar, yvar) {
    annotations <- build_plot_annotations(
        data,
        shown_params = c(xvar, yvar, "rho", "gamma", "alpha", "beta")
    )

    ggplot(data, aes(
        x = .data[[xvar]],
        y = .data[[yvar]],
        color = rho,
        shape = gamma
    )) +
        geom_point(size = 2.2, alpha = 0.9) +
        facet_grid(alpha ~ beta) +
        labs(
            title = paste(yvar, "vs", xvar),
            subtitle = annotations$subtitle,
            x = xvar,
            y = yvar,
            color = "rho",
            shape = "gamma",
            caption = annotations$caption
        ) +
        theme_minimal()
}

##### final-state heatmaps #####
p_final_payoff <- plot_final_heatmap(final_df, "mean_payoff", "Final mean payoff")
p_final_depth <- plot_final_heatmap(final_df, "mean_depth", "Final mean depth")
p_final_effcol <- plot_final_heatmap(final_df, "eff_column", "Final effective number of columns")
p_final_steps <- plot_final_heatmap(final_df, "steps_to_equilibrium", "Steps to equilibrium")
p_final_empty <- plot_final_heatmap(final_df, "empty_rep_size", "Final empty-repertoire mass")
p_final_topcol <- plot_final_heatmap(final_df, "top_col_mass", "Final top-column mass")

print(p_final_payoff)
print(p_final_depth)
print(p_final_effcol)
print(p_final_steps)
print(p_final_empty)
print(p_final_topcol)

##### trajectory plots #####
# do one beta at a time
for (b in levels(df$rho)) {
    print(plot_trajectory(df, "mean_payoff", b))

}

##### endpoint tradeoff plots #####
p_scatter_1 <- plot_endpoint_scatter(final_df, "mean_depth", "mean_payoff")
p_scatter_2 <- plot_endpoint_scatter(final_df, "eff_column", "mean_payoff")
p_scatter_3 <- plot_endpoint_scatter(final_df, "top_col_mass", "mean_payoff")
p_scatter_4 <- plot_endpoint_scatter(final_df, "mean_rep_size", "mean_payoff")

print(p_scatter_1)
print(p_scatter_2)
print(p_scatter_3)
print(p_scatter_4)

##### condensed summary table #####
summary_tbl <- final_df %>%
    select(
        run_id, all_of(present_parameter_cols),
        mean_payoff, adj_payoff, mean_max_depth, mean_depth,
        eff_column, top_col_mass, mean_rep_size, empty_rep_size,
        steps_to_equilibrium
    ) %>%
    arrange(across(any_of(present_parameter_cols)))

print(summary_tbl, n = Inf, width = Inf)

##### grouped summaries by parameter setting #####
# useful if you later add multiple seeds
group_summary <- final_df %>%
    group_by(across(any_of(setdiff(present_parameter_cols, "seed")))) %>%
    summarise(
        mean_payoff = mean(mean_payoff),
        mean_depth = mean(mean_depth),
        eff_column = mean(eff_column),
        top_col_mass = mean(top_col_mass),
        empty_rep_size = mean(empty_rep_size),
        steps_to_equilibrium = mean(steps_to_equilibrium),
        .groups = "drop"
    )

print(group_summary, n = Inf, width = Inf)

##### optional: save plots #####
# ggsave("final_mean_payoff_heatmap.png", p_final_payoff, width = 10, height = 6, dpi = 300)
# ggsave("final_mean_depth_heatmap.png", p_final_depth, width = 10, height = 6, dpi = 300)
# ggsave("final_eff_column_heatmap.png", p_final_effcol, width = 10, height = 6, dpi = 300)

##### optional: quick ranking of runs #####
best_payoff <- final_df %>%
    arrange(desc(mean_payoff)) %>%
    select(run_id, all_of(present_parameter_cols), mean_payoff, mean_depth, eff_column, top_col_mass, steps_to_equilibrium)

print(best_payoff, n = 20, width = Inf)
