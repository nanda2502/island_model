plot_dv_by_time <- function(data, DV, DV_label, color_var, color_lab, linetype_var, linetype_lab) {
    p <- ggplot(data, aes_string(x = "step", y = DV, color = sprintf("factor(%s)", color_var), linetype = sprintf("factor(%s)", linetype_var))) +
        geom_point() + 
        geom_line() + 
        labs(x = "Time",
             y = DV_label,
             color = color_lab,
             linetype = linetype_lab)
    print(p)
    return(p)
}

plot_dv_by_time <- function(df, dv = "mean_payoff") {
    stopifnot(requireNamespace("ggplot2", quietly = TRUE))
    
    if (!dv %in% names(df)) {
        stop(sprintf("Column '%s' not found in df.", var))
    }
    
    ggplot2::ggplot(df, ggplot2::aes(x = step, y = .data[[dv]])) +
        ggplot2::geom_line() +
        ggplot2::geom_point() +
        ggplot2::labs(
            x = "Step",
            y = dv
        ) +
        ggplot2::theme_minimal()
}


plot_dv_by_time_m <- function(df, dv = "mean_payoff", subset_var = NULL, subset_val = NULL) {
    stopifnot(requireNamespace("ggplot2", quietly = TRUE))
    
    if (!dv %in% names(df)) {
        stop(sprintf("Column '%s' not found in df.", dv))
    }
    if (!"m" %in% names(df)) {
        stop("Column 'm' not found in df.")
    }
    if (!"step" %in% names(df)) {
        stop("Column 'step' not found in df.")
    }
    
    if (!is.null(subset_var) & !is.null(subset_val)) {
        df <- df[df[[subset_var]] == subset_val,]
    }
    
    ggplot2::ggplot(
        df,
        ggplot2::aes(
            x = step,
            y = .data[[dv]],
            group = factor(m),
            linetype = factor(m)
        )
    ) +
        ggplot2::geom_line() +
        ggplot2::geom_point(ggplot2::aes(color = factor(m))) +
        ggplot2::labs(
            x = "Step",
            y = dv,
            linetype = "m",
            color = "m"
        ) +
        ggplot2::theme_minimal()
}
