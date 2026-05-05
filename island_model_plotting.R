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