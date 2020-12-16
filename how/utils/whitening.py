"""Functions for training and applying whitening"""

import numpy as np


def l2_normalize_vec(X):
    """L2-normalize given descriptors"""
    return X / (np.linalg.norm(X, ord=2, axis=1, keepdims=True) + 1e-6)


def whitenapply(X, m, P, dimensions=None):
    """Apply whitening (m, P) on descriptors X. If dimensions not None, perform dim reduction."""
    if not dimensions:
        dimensions = P.shape[1]

    X = np.dot(X-m, P[:, :dimensions])
    return l2_normalize_vec(X)


def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T
