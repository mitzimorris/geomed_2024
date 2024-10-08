# computes the inverse of a sparse precision matrix
# sub-optimal implementation - better to use INLA
q_inv_dense <- function(Q, A = NULL) {
  Sigma <- Matrix::solve(Q)   ## need sparse matrix solver
  if (is.null(A))
    return(Sigma)
  else {
    A <- matrix(1,1, nrow(Sigma))
    W <- Sigma %*% t(A)
    Sigma_const <- Sigma - W %*% solve(A %*% W) %*% t(W)
    return(Sigma_const)
  }
}

get_scaling_factor = function(nbs) {
    N = length(nbs)
    adj_matrix = nb2mat(nbs,style="B")
    # Build the adjacency matrix using edgelist
    Q = Diagonal(counts[1], rowSums(adj_matrix)) - adj_matrix

    # Create ICAR precision matrix  (diag - adjacency): this is singular
    Q =  Diagonal(N, rowSums(adj_matrix)) - adj_matrix
    # Add a small jitter to the diagonal for numerical stability (optional but recommended)
    Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)

    # Compute the diagonal elements of the covariance matrix
    Q_inv = q_inv_dense(Q_pert, adj_matrix)

    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    return(exp(mean(log(diag(Q_inv)))))
}
