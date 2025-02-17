library(sf)
library(spdep) |> suppressPackageStartupMessages()
library(ggplot2)
library(tidyverse) |> suppressPackageStartupMessages()
library(igraph)



#  create edgelist for ICAR component
nbs_to_adjlist <- function(nb) {
    adj_matrix = nb2mat(nb,style="B")
    t(as_edgelist(graph_from_adjacency_matrix(adj_matrix, mode="undirected")))
}

# Compute the inverse of a sparse precision matrix
q_inv_dense <- function(Q, A = NULL) {
  Sigma <- Matrix::solve(Q)
  if (is.null(A))
    return(Sigma)
  else {
    A <- matrix(1,1, nrow(Sigma))
    W <- Sigma %*% t(A)
    Sigma_const <- Sigma - W %*% solve(A %*% W) %*% t(W)
    return(Sigma_const)
  }
}

# Compute the geometric mean of the spatial covariance matrix
get_scaling_factor = function(nbs) {
    N = length(nbs)
    # Create ICAR precision matrix  (diag - adjacency): this is singular
    adj_matrix = nb2mat(nbs,style="B")
    Q =  Diagonal(N, rowSums(adj_matrix)) - adj_matrix
    # Add a small jitter to the diagonal for numerical stability (optional but recommended)
    Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
    # Compute the diagonal elements of the covariance matrix
    Q_inv = q_inv_dense(Q_pert, adj_matrix)
    # Compute the geometric mean of the variances, which are on the diagonal of Q_inv
    return(exp(mean(log(diag(Q_inv)))))
}

# Retry unstable function (needed to run Pathfinder to get inits)
# usage: retry_function(your_function, max_attempts = 5, arg1 = value1, arg2 = value2)
retry_function <- function(fun, max_attempts = 10, ...) {
    attempt <- 1
    success <- FALSE
    while (!success && attempt <= max_attempts) {
        result <- tryCatch(
            {
                result <- fun(...)
                success <- TRUE
                result
            },
            error = function(e) {
                message(sprintf("Attempt %d failed: %s", attempt, e$message))
                NULL
            }
        )
        if (!success) {
            attempt <- attempt + 1
            if (attempt <= max_attempts) {
                message(sprintf("Retrying... (attempt %d/%d)", attempt, max_attempts))
            }
        }
    }
    if (!success) {
        stop(sprintf("Failed after %d attempts", max_attempts))
    } else {
        message(sprintf("Succeeded on attempt %d", attempt))
        return(result)
    }
}
