library(tidyverse)
library(reshape2)

upper_corr_matrix_to_df <- function(draws) {
    # Compute correlation matrix
    cor_mat <- cor(draws)
    
    # Mask lower triangle
    cor_mat[lower.tri(cor_mat, diag = TRUE)] <- NA
    
    # Convert to long format
    melted <- melt(cor_mat, varnames = c("Var1", "Var2"), value.name = "Correlation")
    melted <- na.omit(melted)
    return(melted)
}

plot_icar_corr_matrix <- function(draws, title) {
    corr_df <- upper_corr_matrix_to_df(draws)
    ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
        geom_tile() +
        scale_fill_gradient2(low = "darkblue", mid = "white", high = "darkorange", midpoint = 0) +
        theme_minimal() +
        theme(
            axis.text.x = element_blank(),
            axis.text.y = element_blank()
        ) +
        labs(x = "", y = "", title = title)
}

ppc_y_yrep_overlay <- function(y_rep, y, title) {
    # Calculate summaries
    y_rep_median <- apply(y_rep, 2, median)
    y_rep_lower <- apply(y_rep, 2, quantile, probs = 0.025)
    y_rep_upper <- apply(y_rep, 2, quantile, probs = 0.975)
    
    # Create dataframe
    df_plot <- data.frame(
        obs_id = seq_along(y),
        y = y,
        y_rep_median = y_rep_median,
        y_rep_lower = y_rep_lower,
        y_rep_upper = y_rep_upper
    )
    
    # Sort by y
    df_plot_sorted <- df_plot %>%
        arrange(y) %>%
        mutate(sorted_index = row_number())
    
    ggplot(df_plot_sorted, aes(x = sorted_index)) +
        geom_point(aes(y = y), color = "darkblue", size = 0.25) +
        geom_line(aes(y = y_rep_median), color = "orange", alpha = 0.8) +
        geom_ribbon(aes(ymin = y_rep_lower, ymax = y_rep_upper),
                   fill = "grey", alpha = 0.2) +
        theme_minimal() +
        theme(
            plot.title = element_text(size = 11),
            axis.text.x = element_blank()
        ) +
        labs(y = "y_rep", x = "y", title = title)
}

ppc_central_interval <- function(y_rep, y) {
    # Compute percentiles
    q25 <- apply(y_rep, 2, quantile, probs = 0.25)
    q75 <- apply(y_rep, 2, quantile, probs = 0.75)
    
    # Count observations within interval
    within_50 <- sum(y >= q25 & y <= q75)
    
    # Return formatted string
    sprintf("y total: %d, ct y is within y_rep central 50%% interval: %d, pct: %.2f",
            length(y),
            within_50,
            100 * within_50 / length(y))
}


plot_heatmap <- function(nyc_gdf, data, title, subtitle, scale_name) {
  p <- ggplot(nyc_gdf) +
      geom_sf(aes(fill = data), color = "darkblue", size = 0.1) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "orange", midpoint = 0, name=scale_name) +
      labs(title=title, subtitle=subtitle) +
      theme_minimal() +
      theme(plot.title = element_text(size = 32),
            plot.subtitle = element_text(size = 24),
            legend.position = "left",
            legend.title = element_text(size = 20),
            legend.text = element_text(size = 16),
            legend.key.size = unit(24, "pt"),
            plot.margin = margin(20, 20, 20, 20, "pt")
            )
  return(p)
}
