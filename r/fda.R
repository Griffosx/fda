library(fda)
library(ggplot2)
library(dplyr)
library(reshape2)


#######################################################################################################################
# Data pre processing
#######################################################################################################################

# Edit this
setwd("~/Projects/fda/classified_data")

options(max.print = 10000)

# Getting values
bear_dt <- read.csv("bear.csv")
bear_center_dt <- read.csv("bear_center.csv")
bull_dt <- read.csv("bull.csv")
bull_center_dt <- read.csv("bull_center.csv")

# Datasets list
datasets <- list(bull_dt, bull_center_dt, bear_dt, bear_center_dt)
names(datasets) <- c("bull_dt", "bull_center_dt", "bear_dt", "bear_center_dt")

# Removing NA values
for (dataset in names(datasets)) {
  num_na_rows <- sum(rowSums(is.na(datasets[[dataset]])) > 0)  # Count rows with at least one NA
  perc_of_all_rows <- round(100.0 * num_na_rows / nrow(datasets[[dataset]]), 2)
  cat("Dataset:", dataset, "- Rows with NA:", num_na_rows, "- Percent of all Rows:", perc_of_all_rows, "%\n")
}

bull_dt <- bull_dt[complete.cases(bull_dt), ]
bull_center_dt <- bull_center_dt[complete.cases(bull_center_dt), ]
bear_dt <- bear_dt[complete.cases(bear_dt), ]
bear_center_dt <- bear_center_dt[complete.cases(bear_center_dt), ]

cols_for_smoothing <- c("norm_0930",
                        "norm_0945", "norm_1000",
                        "norm_1015", "norm_1030",
                        "norm_1045", "norm_1100",
                        "norm_1115", "norm_1130",
                        "norm_1145")

# Creating a time_grid (x-axis) that corresponds to trading time
# = 9.50 9.75 10.00 10.25 10.50 10.75 11.00 11.25 11.50 11.75
time_grid <- as.numeric(gsub("norm_", "", cols_for_smoothing))
time_grid <- floor(time_grid / 100) + (time_grid %% 100) / 60


# Only selecting columns with times
bull_fda <- bull_dt[, cols_for_smoothing]
bull_center_fda <- bull_center_dt[, cols_for_smoothing]
bear_fda <- bear_dt[, cols_for_smoothing]
bear_center_fda <- bear_center_dt[, cols_for_smoothing]


# Graphs of unsmoothed data
matplot(time_grid, t(bull_fda), type = "l", lty = 1, col = 1:nrow(bull_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bull Stock Days Over Time")

matplot(time_grid, t(bull_center_fda), type = "l", lty = 1, col = 1:nrow(bull_center_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bull Center Stock Days Over Time")

matplot(time_grid, t(bear_fda), type = "l", lty = 1, col = 1:nrow(bear_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bear Stock Days Over Time")

matplot(time_grid, t(bear_center_fda), type = "l", lty = 1, col = 1:nrow(bear_center_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bear Center Stock Days Over Time")


#######################################################################################################################
# Smoothing and FPCA
#######################################################################################################################


# Function for B-spline smoothing with optimal parameter selection
bspline_smoothing <- function(dataset, time_grid, norder = 4, nbasis = 5,
                              lambda_range = 10^seq(-4, -1, length.out = 10)) {
  # Ensure dataset is a matrix with numeric storage mode
  dataset <- as.matrix(dataset)
  storage.mode(dataset) <- "numeric"

  # Create B-spline basis
  basis_bspline <- create.bspline.basis(rangeval = range(time_grid), norder = norder, nbasis = nbasis)

  # Compute GCV for each lambda value
  gcv_scores <- sapply(lambda_range, function(lambda) {
    fdParObj <- fdPar(basis_bspline, lambda = lambda)
    gcv_result <- smooth.basis(time_grid, t(dataset), fdParObj)
    return(mean(gcv_result$gcv))
  })

  # Find optimal lambda
  optimal_lambda <- lambda_range[which.min(gcv_scores)]
  print(paste("Optimal lambda (via GCV):", optimal_lambda))

  # Final smoothing with optimal lambda
  fdParObj_optimized <- fdPar(basis_bspline, lambda = optimal_lambda)
  fdObj_optimized <- smooth.basis(time_grid, t(dataset), fdParObj_optimized)$fd

  # Evaluate smoothed function at time points
  smoothed_values <- eval.fd(time_grid, fdObj_optimized)
  mean_values <- rowMeans(smoothed_values)

  # Return results
  return(list(
    fdObj = fdObj_optimized,
    smoothed_values = smoothed_values,
    mean_values = mean_values,
    optimal_lambda = optimal_lambda,
    gcv_scores = gcv_scores,
    lambda_range = lambda_range
  ))
}

plot_smoothed_values <- function(smooth_result, time_grid, title = "Smoothed Stock Data",
                                 line_color = "lightblue", mean_color = "red",
                                 show_individual = TRUE, ylim = NULL) {

  # Extract data from the smooth_result
  smoothed_values <- smooth_result$smoothed_values
  mean_values <- smooth_result$mean_values
  optimal_lambda <- smooth_result$optimal_lambda

  # Set up the plot area
  if (is.null(ylim)) {
    ylim_range <- range(smoothed_values)
    ylim <- c(ylim_range[1] - 0.05, ylim_range[2] + 0.05)
  }

  # Create empty plot with appropriate dimensions
  plot(time_grid, mean_values, type = "n",
       ylim = ylim,
       xlab = "Time (Hour)",
       ylab = "Normalized Stock Change",
       main = paste0(title, "\n(λ = ", format(optimal_lambda, scientific = TRUE, digits = 3), ")"))

  # Add individual curves if requested
  if (show_individual) {
    matlines(time_grid, smoothed_values, col = adjustcolor(line_color, alpha.f = 0.3),
             lty = 1, type = "l")
  }

  # Add mean curve with thicker line
  lines(time_grid, mean_values, col = mean_color, lwd = 3)

  # Add grid for better readability
  grid()

  # Add legend
  legend("topright",
         legend = c("Individual Curves", "Mean Curve"),
         col = c(line_color, mean_color),
         lty = 1,
         lwd = c(1, 3),
         bg = "white")
}

# Function to perform FPCA with rotation
perform_fpca <- function(fdobj, n_components = 4, apply_rotation = TRUE) {
  # Perform FPCA
  pca_result <- pca.fd(fdobj, nharm = n_components)

  # Apply varimax rotation if requested
  if (apply_rotation) {
    pca_result <- varmx.pca.fd(pca_result)
  }

  # Return the results
  return(pca_result)
}

plot_fpca_components <- function(fpca_result, time_grid, title = "FPCA Components",
                                 line_colors = c("red", "blue", "green", "purple", "orange", "brown"),
                                 variance_threshold = 1.0) {
  # Extract harmonics and the percentage of variation explained
  harmonics <- fpca_result$harmonics
  values <- fpca_result$values
  
  # Calculate the percentage of variation explained by each component
  total_var <- sum(values)
  perc_var <- values / total_var
  cum_var <- cumsum(perc_var)
  
  # Determine how many components to display based on variance threshold
  n_components <- min(which(cum_var >= variance_threshold), length(values))
  if(is.infinite(n_components)) n_components <- length(values)
  
  # Calculate grid layout
  n_row <- ceiling(n_components / 2)
  
  # Set up the plotting area
  par(mfrow = c(n_row, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
  
  # Evaluate harmonics at the time points
  harmonic_values <- eval.fd(time_grid, harmonics)
  
  # Determine y-axis limits for consistent scales across all plots
  y_min <- min(harmonic_values[, 1:n_components])
  y_max <- max(harmonic_values[, 1:n_components])
  y_range <- y_max - y_min
  y_limits <- c(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
  
  # Plot each component
  for (i in 1:n_components) {
    # Plot component
    plot(time_grid, harmonic_values[, i], type = "l", 
         col = line_colors[1 + (i-1) %% length(line_colors)],
         lwd = 2,
         ylim = y_limits,
         xlab = "Time (Hour)",
         ylab = "Component Loading",
         main = sprintf("PC%d (%.1f%% of variance)", 
                        i, 100 * perc_var[i]))
    
    # Add horizontal line at y = 0
    abline(h = 0, lty = 2, col = "gray")
    
    # Add grid lines
    grid(lty = 3, col = "lightgray")
    
    # Add effect visualization: +/- the component
    if(i <= n_components) {
      # Mean function (if available)
      if(!is.null(fpca_result$meanfd)) {
        mean_values <- eval.fd(time_grid, fpca_result$meanfd)
        lines(time_grid, mean_values, lty = 2, col = "black")
        
        # Show effect of adding/subtracting the component
        effect_scale <- 2 * sqrt(values[i])  # Scale by 2 std deviations
        lines(time_grid, mean_values + effect_scale * harmonic_values[, i], 
              lty = 3, col = adjustcolor(line_colors[1 + (i-1) %% length(line_colors)], 0.7))
        lines(time_grid, mean_values - effect_scale * harmonic_values[, i], 
              lty = 3, col = adjustcolor(line_colors[1 + (i-1) %% length(line_colors)], 0.7))
      }
    }
  }
  
  # If odd number of components, handle the empty plot in the grid
  if (n_components %% 2 == 1 && n_components > 1) {
    plot.new()
  }
  
  # Add overall title
  mtext(paste0(title, " (", 
               ifelse(fpca_result$rmat, "with", "without"), 
               " rotation)"), 
        outer = TRUE, cex = 1.2)
  
  # Return the number of components displayed and variance explained
  return(list(
    n_components = n_components,
    perc_var = perc_var[1:n_components],
    cum_var = cum_var[n_components]
  ))
}

# BULL
bull_smooth <- bspline_smoothing(bull_fda, time_grid)
plot_smoothed_values(bull_smooth, time_grid, title = "Smoothed Bull Market Data")
bull_fpca <- perform_fpca(bull_smooth$fdObj)
plot_fpca_components(bull_fpca, time_grid, title = "Bull FPCA")

# BEAR
bear_smooth <- bspline_smoothing(bear_fda, time_grid)
plot_smoothed_values(bear_smooth, time_grid, title = "Smoothed Bear Market Data")
bear_fpca <- perform_fpca(bear_smooth$fdObj)
plot_fpca_components(bear_fpca, time_grid, title = "Bear FPCA")

# BULL CENTER
bull_center_smooth <- bspline_smoothing(bull_center_fda, time_grid)
plot_smoothed_values(bull_center_smooth, time_grid, title = "Smoothed Bull Center Market Data")
bull_center_fpca <- perform_fpca(bull_center_smooth$fdObj)
plot_fpca_components(bull_center_fpca, time_grid, title = "Bull center FPCA")

# BEAR CENTER
bear_center_smooth <- bspline_smoothing(bear_center_fda, time_grid)
plot_smoothed_values(bear_center_smooth, time_grid, title = "Smoothed Bear Center Market Data")
bear_center_fpca <- perform_fpca(bear_center_smooth$fdObj)
plot_fpca_components(bear_center_fpca, time_grid, title = "Bear center FPCA")




