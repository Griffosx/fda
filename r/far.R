library(fda)
library(fda.usc)
library(far)
library(dplyr)
library(ggplot2)
library(reshape2)

#######################################################################################################################
# Data pre processing for FAR modeling
#######################################################################################################################

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

bull_dt <- bull_dt[complete.cases(bull_dt), ]
bull_center_dt <- bull_center_dt[complete.cases(bull_center_dt), ]
bear_dt <- bear_dt[complete.cases(bear_dt), ]
bear_center_dt <- bear_center_dt[complete.cases(bear_center_dt), ]

# Define predictor columns (X) and response column (Y)
predictor_cols <- c("norm_0930", "norm_0945", "norm_1000", "norm_1015", "norm_1030", 
                    "norm_1045", "norm_1100", "norm_1115", "norm_1130")
response_col <- "norm_1145"

# Creating time grids
predictor_time_grid <- as.numeric(gsub("norm_", "", predictor_cols))
predictor_time_grid <- floor(predictor_time_grid / 100) + (predictor_time_grid %% 100) / 60

# Extract data for each dataset
extract_far_data <- function(dataset, predictor_cols, response_col) {
  predictor_data <- as.matrix(dataset[, predictor_cols])
  response_data <- dataset[, response_col]
  return(list(predictors = predictor_data, response = response_data))
}

bull_far_data <- extract_far_data(bull_dt, predictor_cols, response_col)
bull_center_far_data <- extract_far_data(bull_center_dt, predictor_cols, response_col)
bear_far_data <- extract_far_data(bear_dt, predictor_cols, response_col)
bear_center_far_data <- extract_far_data(bear_center_dt, predictor_cols, response_col)


#######################################################################################################################
# FAR Model Functions using dedicated packages
#######################################################################################################################

# Function to prepare data for FAR package
prepare_far_data <- function(predictor_matrix, response_vector, time_grid) {
  # Create fdata object for predictors
  predictor_fdata <- fdata(predictor_matrix, argvals = time_grid)
  
  # For FAR package, we need to create a proper data structure
  # The 'far' package expects data in a specific format
  far_data <- list(
    X = t(predictor_matrix),  # Transpose so each column is an observation
    Y = response_vector
  )
  
  return(list(
    fdata = predictor_fdata,
    far_format = far_data,
    time_grid = time_grid
  ))
}

# Function to fit FAR model using fda.usc (simplified - no CV)
fit_far_fda_usc <- function(predictor_matrix, response_vector, time_grid) {
  cat("Creating functional data object...\n")
  # Create fdata object - converts discrete observations to functional form
  predictor_fdata <- fdata(predictor_matrix, argvals = time_grid)
  
  cat("Fitting PC model...\n")
  # PC model: Fixed number of components (no automatic selection)
  # Use a reasonable fixed number of components to avoid CV overhead
  n_components <- min(5, nrow(predictor_matrix) - 1)  # Fixed to 5 or fewer
  far_model_pc <- fregre.pc(predictor_fdata, response_vector, 
                            l = n_components,  # Fixed number of components
                            lambda = 0)  # No penalization
  
  cat("Fitting basis model...\n")
  # Basis model: Uses default parameters (no CV for lambda selection)
  far_model_basis <- fregre.basis(predictor_fdata, response_vector)
  
  cat("Model fitting completed.\n")
  
  return(list(
    pc_model = far_model_pc,
    basis_model = far_model_basis,
    fdata = predictor_fdata
  ))
}

# Function to evaluate model performance
evaluate_model <- function(actual, predicted, model_name) {
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  r_squared <- cor(actual, predicted, use = "complete.obs")^2
  
  cat("\n", model_name, "Performance:\n")
  cat("MAE:", round(mae, 4), "\n")
  cat("RMSE:", round(rmse, 4), "\n")
  cat("R-squared:", round(r_squared, 4), "\n")
  
  return(list(mae = mae, rmse = rmse, r_squared = r_squared))
}

# Function to plot model results
plot_model_results <- function(actual, predicted, dataset_name, model_type = "FAR") {
  par(mfrow = c(2, 2))
  
  # 1. Actual vs Predicted
  plot(actual, predicted, 
       main = paste(dataset_name, "-", model_type, "Actual vs Predicted"),
       xlab = "Actual norm_1145", ylab = "Predicted norm_1145",
       pch = 19, col = "darkblue")
  abline(0, 1, col = "red", lwd = 2, lty = 2)
  abline(lm(predicted ~ actual), col = "darkgreen", lwd = 2)
  grid(lty = "dotted")
  
  # Add correlation
  r_val <- cor(actual, predicted, use = "complete.obs")
  text(min(actual, na.rm = TRUE), max(predicted, na.rm = TRUE), 
       paste("R =", round(r_val, 3)), adj = c(0, 1), cex = 1.2)
  
  # 2. Residuals vs Fitted
  residuals <- actual - predicted
  plot(predicted, residuals,
       main = paste(dataset_name, "- Residuals vs Fitted"),
       xlab = "Fitted Values", ylab = "Residuals",
       pch = 19, col = "darkblue")
  abline(h = 0, col = "red", lwd = 2, lty = 2)
  grid(lty = "dotted")
  
  # 3. Q-Q plot
  qqnorm(residuals, main = paste(dataset_name, "- Q-Q Plot"))
  qqline(residuals, col = "red", lwd = 2)
  
  # 4. Residual histogram
  hist(residuals, main = paste(dataset_name, "- Residuals"), 
       col = "lightblue", xlab = "Residuals")
  
  par(mfrow = c(1, 1))
}


#######################################################################################################################
# Process each dataset
#######################################################################################################################

process_dataset <- function(far_data, dataset_name, time_grid) {
  cat("\n=== Processing", dataset_name, "===\n")
  cat("Dataset size:", nrow(far_data$predictors), "observations with", 
      ncol(far_data$predictors), "time points\n")
  
  # Prepare data
  cat("Preparing data structures...\n")
  prepared_data <- prepare_far_data(far_data$predictors, far_data$response, time_grid)
  
  # Fit models using fda.usc (no CV)
  models <- fit_far_fda_usc(far_data$predictors, far_data$response, time_grid)
  
  # Get predictions from fitted models
  cat("Generating predictions...\n")
  predictions_pc <- predict(models$pc_model, prepared_data$fdata)
  predictions_basis <- predict(models$basis_model, prepared_data$fdata)
  
  # Evaluate models
  pc_performance <- evaluate_model(far_data$response, predictions_pc, 
                                   paste(dataset_name, "PC Model"))
  basis_performance <- evaluate_model(far_data$response, predictions_basis, 
                                      paste(dataset_name, "Basis Model"))
  
  # Plot results
  plot_model_results(far_data$response, predictions_pc, dataset_name, "PC")
  plot_model_results(far_data$response, predictions_basis, dataset_name, "Basis")
  
  return(list(
    models = models,
    predictions = list(pc = predictions_pc, basis = predictions_basis),
    performance = list(pc = pc_performance, basis = basis_performance),
    data = prepared_data
  ))
}

# Process all datasets
cat("Starting FAR model analysis...\n")
bull_results <- process_dataset(bull_far_data, "Bull Market", predictor_time_grid)
bear_results <- process_dataset(bear_far_data, "Bear Market", predictor_time_grid)
bull_center_results <- process_dataset(bull_center_far_data, "Bull Center Market", predictor_time_grid)
bear_center_results <- process_dataset(bear_center_far_data, "Bear Center Market", predictor_time_grid)


#######################################################################################################################
# Model comparison and summary
#######################################################################################################################

compare_all_models <- function(results_list, dataset_names) {
  cat("\n=== Model Comparison Summary ===\n")
  
  # Create comparison dataframe
  comparison_df <- data.frame(
    Dataset = rep(dataset_names, each = 2),
    Model = rep(c("PC", "Basis"), length(dataset_names)),
    MAE = numeric(length(dataset_names) * 2),
    RMSE = numeric(length(dataset_names) * 2),
    R_squared = numeric(length(dataset_names) * 2)
  )
  
  # Fill in the data
  for (i in 1:length(results_list)) {
    row_pc <- (i - 1) * 2 + 1
    row_basis <- (i - 1) * 2 + 2
    
    # Training performance
    comparison_df[row_pc, 3:5] <- c(results_list[[i]]$performance$pc$mae,
                                    results_list[[i]]$performance$pc$rmse,
                                    results_list[[i]]$performance$pc$r_squared)
    
    comparison_df[row_basis, 3:5] <- c(results_list[[i]]$performance$basis$mae,
                                       results_list[[i]]$performance$basis$rmse,
                                       results_list[[i]]$performance$basis$r_squared)
  }
  
  print(comparison_df)
  
  # Find best models
  best_r2 <- which.max(comparison_df$R_squared)
  best_rmse <- which.min(comparison_df$RMSE)
  
  cat("\nBest model by R²:", comparison_df[best_r2, 1], "-", comparison_df[best_r2, 2], 
      "(R² =", round(comparison_df[best_r2, 5], 4), ")\n")
  cat("Best model by RMSE:", comparison_df[best_rmse, 1], "-", comparison_df[best_rmse, 2], 
      "(RMSE =", round(comparison_df[best_rmse, 4], 4), ")\n")
  
  return(comparison_df)
}

# Compare all results
all_results <- list(bull_results, bear_results, bull_center_results, bear_center_results)
dataset_names <- c("Bull", "Bear", "Bull Center", "Bear Center")
final_comparison <- compare_all_models(all_results, dataset_names)


#######################################################################################################################
# Summary and conclusions
#######################################################################################################################

cat("\n=== FAR Model Analysis Summary ===\n")
cat("This analysis used fda.usc package to fit Functional Auto-Regression models\n")
cat("to predict norm_1145 based on functional data from norm_0930 to norm_1130.\n\n")

cat("Two model types were compared:\n")
cat("1. Principal Components (PC) model - uses fregre.pc() with fixed 5 components\n")
cat("2. Basis representation model - uses fregre.basis() with default parameters\n\n")

cat("Key findings:\n")
best_overall <- final_comparison[which.max(final_comparison$R_squared), ]
cat("- Best performing model:", best_overall$Dataset, "-", best_overall$Model, "\n")
cat("- Training R²:", round(best_overall$R_squared, 4), "\n")
cat("- Training RMSE:", round(best_overall$RMSE, 4), "\n")

cat("\nNote: Cross-validation was removed to improve computation speed.\n")
cat("Models use fixed parameters: PC model with 5 components, Basis model with defaults.\n")