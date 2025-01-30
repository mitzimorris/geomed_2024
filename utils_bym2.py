"""
BYM2 model helper functions for computing scaling factors.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from libpysal.weights import W
from scipy import stats
from typing import Optional, Union

def nbs_to_adjlist(nbs: W) -> np.ndarray:
    nbs_adj =  nbs.to_adjlist(remove_symmetric=True)
    j1 = nbs_adj['focal'] + 1
    j2 = nbs_adj['neighbor'] + 1
    edge_pairs = np.vstack([j1, j2])
    return (edge_pairs)

def q_inv_dense(Q: sp.spmatrix, A: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the inverse of a sparse precision matrix Q.
    - param Q: Sparse precision matrix
    - param A: Optional constraint matrix (default: None)
    - return: Dense matrix representing either Q^(-1) or the constrained inverse
    """
    Sigma = splinalg.inv(Q).todense()
    if A is None:
        return Sigma
    
    A = np.ones((1, Sigma.shape[0])) if A is None else A
    W = Sigma @ A.T
    Sigma_const = Sigma - W @ np.linalg.inv(A @ W) @ W.T
    return Sigma_const

def _compute_precision_matrix(nbs: W) -> sp.spmatrix:
    """
    Helper function to compute the precision matrix with numerical stability.
    - param nbs: Neighborhood structure (W object from libpysal)
    - return: Sparse precision matrix with jitter added for stability
    """
    adj_matrix = sp.csr_matrix(nbs.full()[0])
    Q = sp.diags(np.ravel(adj_matrix.sum(axis=1))) - adj_matrix
    jitter = np.max(Q.diagonal()) * np.sqrt(np.finfo(float).eps)
    return Q + sp.eye(nbs.n) * jitter

def get_scaling_factor(nbs: W) -> np.float64:
    """
    Compute the geometric mean of the spatial covariance matrix.
    - param nbs: Neighborhood structure (W object from libpysal)
    - return: Geometric mean of the variances
    """
    Q_pert = _compute_precision_matrix(nbs)
    Q_inv = q_inv_dense(Q_pert, None)
    return np.exp(np.mean(np.log(np.diag(Q_inv))))

