get_scaling_factor_INLA = function(nbs) {
    N = length(nbs)
    # Create ICAR precision matrix  (diag - adjacency): this is singular
    adj_matrix = nb2mat(nbs,style="B")
    Q =  Diagonal(N, rowSums(adj_matrix)) - adj_matrix
    # Add a small jitter to the diagonal for numerical stability (optional but recommended)
    Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
    # Compute the diagonal elements of the covariance matrix
    Q_inv = inla.qinv(Q_pert, constr=list(A = matrix(1,1,N),e=0))
    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    return(exp(mean(log(diag(Q_inv)))))
}
