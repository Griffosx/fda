# Libraries
library(fda)
library(ggplot2)
library(dplyr)
library(reshape2)

setwd("C:/Users/KasparasRutkauskas/OneDrive - Valyuz UAB/Desktop/Private/VU/Functional Data Analysis/classified_data")

options(max.print = 10000)

# Getting values
bear_dt <- read.csv("bear.csv")
bear_center_dt <- read.csv("bear_center.csv")
bull_dt <- read.csv("bull.csv")
bull_center_dt <- read.csv("bull_center.csv")
unknown_dt <- read.csv("unknown.csv")
weird_dt <- read.csv("weird.csv")


head(bear_dt)

# Datasets list
datasets <- list(bull_dt, bull_center_dt, bear_dt, bear_center_dt, unknown_dt, weird_dt)
names(datasets) <- c("bull_dt", "bull_center_dt", "bear_dt", "bear_center_dt", "unknown_dt", "weird_dt")

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
unknown_dt <- unknown_dt[complete.cases(unknown_dt), ]
weird_dt <- weird_dt[complete.cases(weird_dt), ]

cols_for_smoothing <- c("norm_0930",
                        "norm_0945", "norm_1000",
                        "norm_1015", "norm_1030",
                        "norm_1045", "norm_1100",
                        "norm_1115", "norm_1130",
                        "norm_1145")

# Creating a time_grid (x-axis) that corresponds to trading time
time_grid2 <- as.numeric(gsub("norm_", "", cols_for_smoothing)) 

# Only selecting columns with times
bull_fda <- bull_dt[, cols_for_smoothing]
bull_center_fda <- bull_center_dt[, cols_for_smoothing]
bear_fda <- bear_dt[, cols_for_smoothing]
bear_center_fda <- bear_center_dt[, cols_for_smoothing]
weird_fda <- weird_dt[, cols_for_smoothing]
unknown_fda <- unknown_dt[, cols_for_smoothing]


# Original Data

time_grid2 <- as.numeric(gsub("norm_", "", colnames(bull_fda)))

time_grid2 <- floor(time_grid2 / 100) + (time_grid2 %% 100) / 60

# Creating another grid with less intervals for b-splines (5 intervals)
time_grid3 <- seq(min(range(time_grid2)), max(range(time_grid2)), length.out = 5)


# Graphs of unsmoothed data

matplot(time_grid2, t(bull_fda), type = "l", lty = 1, col = 1:nrow(bull_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bull Stock Days Over Time")

matplot(time_grid2, t(bull_center_fda), type = "l", lty = 1, col = 1:nrow(bull_center_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bull Center Stock Days Over Time")

matplot(time_grid2, t(bear_fda), type = "l", lty = 1, col = 1:nrow(bear_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bear Stock Days Over Time")

matplot(time_grid2, t(bear_center_fda), type = "l", lty = 1, col = 1:nrow(bear_center_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Bear Center Stock Days Over Time")

matplot(time_grid2, t(weird_fda), type = "l", lty = 1, col = 1:nrow(weird_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Weird Stock Days Over Time")

matplot(time_grid2, t(unknown_fda), type = "l", lty = 1, col = 1:nrow(unknown_fda),
        xlab = "Time (HHMM)", ylab = "Stock Change", main = "Not smoothed Unknown Stock Days Over Time")


# Setting up a testing smoothing plot function

smoothing_plot <- function(dataset, smooth_values, title) {
  rows_to_check <- sample(nrow(dataset), 3)
  selected_data <- dataset[rows_to_check,]
  
  colors <- rainbow(nrow(selected_data)) 
  
  mainn = paste("Stock Price Action Smoothed vs Non-Smoothed of ", title, "days")
  matplot(time_grid2, t(selected_data), type = "l", lty = 1, col = colors, 
          main = mainn, xlab = "Time (Hours)", ylab = "Normalized Value", lwd = 1)
  
  # Overlay the smoothed data using matlines()
  matlines(time_grid2, smooth_values[,rows_to_check], lty = 1, col = colors, lwd = 2)
  
  legend("topright", legend = c("Raw Data", "Smoothed Data"), lty = c(2,1), lwd = 2, xpd = TRUE)
}

#  From here on out, the code is separated into sections representing the four types of stock days - bull, bear, bull center, bear center

#  BULL FDA

bull_fda <- as.matrix(bull_fda)  # Ensure it's a matrix
storage.mode(bull_fda) <- "numeric"
  

# B-spline Smoothing

norder = 4
basis_bspline <- create.bspline.basis(rangeval = range(time_grid2), norder = norder, nbasis = 5)

fdParObj <- fdPar(basis_bspline, lambda = 0.01)  # Smoothing parameter lambda
fdObj <- smooth.basis(time_grid2, t(bull_fda), fdParObj)$fd

# Step 1: Plotting smoothing with 4 norder and 0.01 lambda

plot(fdObj, main = "Bull Days' Smoothed Stock Changes with B-splines")

# Step 2: Define a Range of Lambda Values (Log Scale)
lambda_values <- 10^seq(-4, -1, length.out = 10)  # Test values from 10^-6 to 10^2

# Step 3: Compute GCV for Each Lambda
gcv_scores <- sapply(lambda_values, function(lambda) {
  fdParObj <- fdPar(basis_bspline, lambda = lambda)  # Create fdParObj for each lambda
  gcv_result <- smooth.basis(time_grid2, t(bull_fda), fdParObj)  # Smooth data
  return(mean(gcv_result$gcv))  # Extract GCV score
})

# Step 4: Find the Optimal Lambda
optimal_lambda_bsplines <- lambda_values[which.min(gcv_scores)]
print(paste("Optimal lambda (via GCV):", optimal_lambda_bsplines))

# Step 5: splining
fdParObj_optimized <- fdPar(basis_bspline, lambda = optimal_lambda_bsplines)  # Create fdParObj for each lambda
fdObj_bspline <- smooth.basis(time_grid2, t(bull_fda), fdParObj_optimized)$fd

sv_bsplines_bull <- eval.fd(time_grid2, fdObj_bspline)

mean_bsplines_bull <- rowMeans(sv_bsplines_bull)

# Prepping for FPCA

fpca_bsplines_bull <- pca.fd(fdObj_bspline, nharm=4)

var_explained_bsplines_bull <- round(fpca_bsplines_bull$varprop * 100, 1)

# Creating Perturbation factpr
multiplier_bsplines_bull <- 2 * sqrt(fpca_bsplines_bull$values[1:4])  # Scaling factor for visualization

# Plot of smoother values

mainn = paste("Bullish stock days smoothed B-splines with norder ",
              norder,
              ' and lambda',
              optimal_lambda_bsplines)
plot(fdObj_bspline, main = mainn, lwd=0.5)

lines(time_grid2, mean_bsplines_bull, col="red", lwd=2)

# Plot the GCV values

plot(lambda_values, gcv_scores, type = "b", log = "x", 
     xlab = "Lambda (log scale)", ylab = "GCV Score", 
     main = "GCV Scores for Different Lambda Values of the Bullish days")
points(optimal_lambda_bsplines, min(gcv_scores), col = "red", pch = 19)


# Testing 3 random days smoothed v not smoothed

set.seed(123)

smoothing_plot(bull_fda, sv_bsplines_bull, 'Bull (B-splines)')

# FPCA Graph

# Define multi-panel layout for plotting
par(mfrow=c(2,2), mar=c(4,4,2,1)) 

for (i in 1:4) {
  print(i)
  plot(fpca_bsplines_bull$harmonics[i],
       xlab = 'Time',
       ylab = 'Value of PC Curve',
       ylim=c(-1, 1),
       main=paste0("PC ", i, " (", var_explained_bsplines_bull[i], "%)"),
       col="black", lwd=1
       )
}
mtext("4 PC Functions for Bullish Days (B-splines)",
      outer=TRUE,
      cex=1,
      font=1,
      line=-33.5)


# BEAR DAYS


# B-splines basis smoothing

norder = 4
basis_bspline <- create.bspline.basis(rangeval = range(time_grid2), norder = norder, nbasis = 5)


# Step 2: Define a Range of Lambda Values (Log Scale)
lambda_values <- 10^seq(-4, -1, length.out = 10)  # Test values from 10^-6 to 10^2

# Step 3: Compute GCV for Each Lambda
gcv_scores <- sapply(lambda_values, function(lambda) {
  fdParObj <- fdPar(basis_bspline, lambda = lambda)  # Create fdParObj for each lambda
  gcv_result <- smooth.basis(time_grid2, t(bear_fda), fdParObj)  # Smooth data
  return(mean(gcv_result$gcv))  # Extract GCV score
})

# Step 5: Find the Optimal Lambda
optimal_lambda_bsplines <- lambda_values[which.min(gcv_scores)]
print(paste("Optimal lambda (via GCV):", optimal_lambda_bsplines))

fdParObj_optimized <- fdPar(basis_bspline, lambda = optimal_lambda_bsplines)  # Create fdParObj for each lambda
fdObj_bspline <- smooth.basis(time_grid2, t(bear_fda), fdParObj_optimized)$fd

sv_bsplines_bear <- eval.fd(time_grid2, fdObj_bspline)

mean_bsplines_bear <- rowMeans(sv_bsplines_bear)

# Prepping for FPCA

fpca_bsplines_bear <- pca.fd(fdObj_bspline, nharm=4)

var_explained_bsplines_bear <- round(fpca_bsplines_bear$varprop * 100, 1)


# Plot of smoother values

mainn = paste("Bearish days B-splines with norder ",
              norder,
              ' and lambda',
              optimal_lambda_bsplines)
plot(fdObj_bspline, main = mainn)

lines(time_grid2, mean_bsplines_bear, col="red", lwd=2)

# GCV Scores over different lambdas

plot(lambda_values, gcv_scores, type = "b", log = "x", 
     xlab = "Lambda (log scale)", ylab = "GCV Score", 
     main = "GCV Scores for Different Lambda Values of the Bearish days")
points(optimal_lambda_bsplines, min(gcv_scores), col = "red", pch = 19)


# Testing 3 random days smoothed v not smoothed

smoothing_plot(bear_fda, sv_bsplines_bear, 'Bear (B-splines)')


# FPCA Graph

# Define multi-panel layout for plotting
par(mfrow=c(2,2), mar=c(4,4,2,1)) 

for (i in 1:4) {
  print(i)
  plot(fpca_bsplines_bear$harmonics[i],
       xlab = 'Time',
       ylab = 'Value of PC Curve',
       ylim=c(-1, 1),
       main=paste0("PC ", i, " (", var_explained_bsplines_bear[i], "%)"),
       col="black", lwd=1
  )
}
mtext("4 PC Functions for Bearish Days (B-splines)",
      outer=TRUE,
      cex=1,
      font=1,
      line=-33.5)


#  BULL CENTER FDA

bull_center_fda <- as.matrix(bull_center_fda)  # Ensure it's a matrix
storage.mode(bull_center_fda) <- "numeric"


# B-splines basis smoothing

norder = 4
basis_bspline <- create.bspline.basis(rangeval = range(time_grid2), norder = norder, nbasis = 5)

# Step 3: Define a Range of Lambda Values (Log Scale)
lambda_values <- 10^seq(-4, -1, length.out = 10)  # Test values from 10^-6 to 10^2

# Step 4: Compute GCV for Each Lambda
gcv_scores <- sapply(lambda_values, function(lambda) {
  fdParObj <- fdPar(basis_bspline, lambda = lambda)  # Create fdParObj for each lambda
  gcv_result <- smooth.basis(time_grid2, t(bull_center_fda), fdParObj)  # Smooth data
  return(mean(gcv_result$gcv))  # Extract GCV score
})

# Step 5: Find the Optimal Lambda
optimal_lambda_bsplines <- lambda_values[which.min(gcv_scores)]
print(paste("Optimal lambda (via GCV):", optimal_lambda_bsplines))

fdParObj_optimized <- fdPar(basis_bspline, lambda = optimal_lambda_bsplines)  # Create fdParObj for each lambda
fdObj_bspline <- smooth.basis(time_grid2, t(bull_center_fda), fdParObj_optimized)$fd

sv_bsplines_bull_center <- eval.fd(time_grid2, fdObj_bspline)

mean_bsplines_bull_center <- rowMeans(sv_bsplines_bull_center)

# Prepping for FPCA

fpca_bsplines_bull_center <- pca.fd(fdObj_bspline, nharm=4)

var_explained_bsplines_bull_center <- round(fpca_bsplines_bull_center$varprop * 100, 1)


# Plot of smoother values

mainn = paste("Bull Center days B-splines with norder ", norder,
              ', nbasis ', nbasis,
              ' and lambda ', optimal_lambda_bsplines)
plot(fdObj_bspline, main = mainn)

lines(time_grid2, mean_bsplines_bull_center, col="red", lwd=2)


# GCV Scores over different lambdas

plot(lambda_values, gcv_scores, type = "b", log = "x", 
     xlab = "Lambda (log scale)", ylab = "GCV Score", 
     main = "GCV Scores for Different Lambda Values for Bullish Center days")
points(optimal_lambda_bsplines, min(gcv_scores), col = "red", pch = 19)

# Testing 3 random days smoothed v not smoothed

smoothing_plot(bull_center_fda, sv_bsplines_bull_center, 'Bull Center (B-splines)')

# FPCA Graph

# Define multi-panel layout for plotting
par(mfrow=c(2,2), mar=c(4,4,2,1)) 

for (i in 1:4) {
  print(i)
  plot(fpca_bsplines_bull_center$harmonics[i],
       xlab = 'Time',
       ylab = 'Value of PC Curve',
       ylim=c(-1, 1),
       main=paste0("PC ", i, " (", var_explained_bsplines_bull_center[i], "%)"),
       col="black", lwd=1
  )
}
mtext("4 PC Functions for Bull Center Days (B-splines)",
      outer=TRUE,
      cex=1,
      font=1,
      line=-33.5)


# BEAR Center FDA

bear_center_fda <- as.matrix(bear_center_fda)  # Ensure it's a matrix
storage.mode(bear_center_fda) <- "numeric"

# B-splines basis smoothing

norder = 4
basis_bspline <- create.bspline.basis(rangeval = range(time_grid2), norder = norder, nbasis = 5)

# Step 3: Define a Range of Lambda Values (Log Scale)
lambda_values <- 10^seq(-4, -1, length.out = 10)  # Test values from 10^-6 to 10^2

# Step 4: Compute GCV for Each Lambda
gcv_scores <- sapply(lambda_values, function(lambda) {
  fdParObj <- fdPar(basis_bspline, lambda = lambda)  # Create fdParObj for each lambda
  gcv_result <- smooth.basis(time_grid2, t(bear_center_fda), fdParObj)  # Smooth data
  return(mean(gcv_result$gcv))  # Extract GCV score
})

# Step 5: Find the Optimal Lambda
optimal_lambda_bsplines <- lambda_values[which.min(gcv_scores)]
print(paste("Optimal lambda (via GCV):", optimal_lambda_bsplines))

fdParObj_optimized <- fdPar(basis_bspline, lambda = optimal_lambda_bsplines)  # Create fdParObj for each lambda
fdObj_bspline <- smooth.basis(time_grid2, t(bear_center_fda), fdParObj_optimized)$fd

sv_bsplines_bear_center <- eval.fd(time_grid2, fdObj_bspline)

mean_bsplines_bear_center <- rowMeans(sv_bsplines_bear_center)


# Prepping for FPCA

fpca_bsplines_bear_center <- pca.fd(fdObj_bspline, nharm=4)

var_explained_bsplines_bear_center <- round(fpca_bsplines_bear_center$varprop * 100, 1)

# Plot of smoother values

mainn = paste("Bear Center days B-splines with norder ", norder,
              ', nbasis ', nbasis,
              ' and lambda ', optimal_lambda_bsplines)
plot(fdObj_bspline, main = mainn)

lines(time_grid2, mean_bsplines_bear_center, col="red", lwd=2)


# GCV Scores over different lambdas

plot(lambda_values, gcv_scores, type = "b", log = "x", 
     xlab = "Lambda (log scale)", ylab = "GCV Score", 
     main = "GCV Scores for Different Lambda Values")
points(optimal_lambda_bsplines, min(gcv_scores), col = "red", pch = 19)


# Testing 3 random days smoothed v not smoothed

smoothing_plot(bear_center_fda, sv_bsplines_bear_center, 'Bear Center (B-splines)')

# FPCA Graph

# Define multi-panel layout for plotting
par(mfrow=c(2,2), mar=c(4,4,2,1)) 

for (i in 1:4) {
  plot(fpca_bsplines_bear_center$harmonics[i],
       xlab = 'Time',
       ylab = 'Value of PC Curve',
       ylim=c(-1, 1),
       main=paste0("PC ", i, " (", var_explained_bsplines_bear_center[i], "%)"),
       col="black", lwd=1
  )
}
mtext("4 PC Functions for Bear Center Days (B-splines)",
      outer=TRUE,
      cex=1,
      font=1,
      line=-33.5)
