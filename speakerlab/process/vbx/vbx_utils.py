# Copyright (c) 2023 Shuai Wang, 2024 VBx Contributors
# Utilities for VBx diarization system
# Licensed under the Apache License, Version 2.0

import math
import numpy as np
import h5py
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.special import logsumexp


def l2_norm(vec_or_matrix):
    """L2 normalization of vector array."""
    if len(vec_or_matrix.shape) == 1:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def norm_embeddings(embeddings, kaldi_style=True):
    """Norm embeddings to unit length."""
    scale = math.sqrt(embeddings.shape[-1]) if kaldi_style else 1.
    if len(embeddings.shape) == 2:
        return (scale * embeddings.transpose() / np.linalg.norm(embeddings, axis=1)).transpose()
    elif len(embeddings.shape) == 1:
        return scale * embeddings / np.linalg.norm(embeddings)


def get_class_means_between_and_within_covs(samples, classids, bias=True):
    """Return class means and the between and shared within class covariance matrix."""
    nsamples, dim = samples.shape
    sample2class = classids_to_posteriors(classids)
    counts = np.array(sample2class.sum(1))
    means = np.array(sample2class.dot(samples)) / counts
    between_cov = (means - samples.mean(0)) * np.sqrt(counts)
    between_cov = between_cov.T.dot(between_cov) / nsamples
    within_cov = np.cov(samples.T, bias=bias) - between_cov
    return means, between_cov, within_cov


def classids_to_posteriors(classids, dtype='f4'):
    """Transform classids into a sparse matrix of 1 and 0."""
    nsamples = np.array(classids).squeeze().shape[0]
    class2post = coo_matrix((np.ones(nsamples, dtype), (classids, list(range(nsamples))))).tocsr()
    return class2post


def compute_normalizing_transform(covar):
    """Compute normalizing transform from covariance matrix."""
    try:
        c = np.linalg.cholesky(covar)
    except np.linalg.LinAlgError:
        c = np.linalg.cholesky(covar + np.eye(covar.shape[0]) * 1e-6)
    c = np.linalg.inv(c)
    return c


def sort_svd(s, d):
    """Sort SVD results in descending order."""
    idx = np.argsort(-s)
    s1 = s[idx]
    d1 = d.T
    d1 = d1[idx].T
    return s1, d1


def train_lda_transform(embeddings, labels, lda_dim, whiten=False):
    """
    Train LDA transform on embeddings with labels.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of cluster labels
        lda_dim: target dimensionality
        whiten: whether to apply whitening
    
    Returns:
        mean1, lda, mean2: transformation parameters
    """
    n_vectors, vector_dim = embeddings.shape
    
    unique_idxs = list(set(labels))
    idxs = np.array([unique_idxs.index(x) for x in labels])
    
    # Subtract mean
    mean1 = np.mean(embeddings, axis=0)
    embeddings = embeddings - mean1
    
    # L2-norm
    embeddings = l2_norm(embeddings)
    
    # Train LDA
    means, between_cov, within_cov = get_class_means_between_and_within_covs(embeddings, idxs)
    
    # Regularization
    within_cov = within_cov + 0.01 * np.diag(np.ones(within_cov.shape[0]))
    
    w, v = eigh(between_cov, within_cov)
    index = np.argsort(w)[::-1]
    v = v[:, index]
    
    lda = v[:, 0:lda_dim]
    if whiten:
        lda = lda.dot(np.diag(1. / np.sqrt(np.diag(lda.T.dot(between_cov + within_cov).dot(lda)))))
    
    embeddings = embeddings.dot(lda)
    
    # Subtract mean again
    mean2 = np.mean(embeddings, axis=0)
    
    return mean1, lda, mean2


def save_transform_h5(output_h5, mean1, lda, mean2):
    """Save LDA transform to h5 file."""
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('mean1', data=mean1)
        f.create_dataset('mean2', data=mean2)
        f.create_dataset('lda', data=lda)


def load_transform_h5(transform_h5):
    """Load LDA transform from h5 file."""
    with h5py.File(transform_h5, 'r') as f:
        mean1 = f['mean1'][:]
        mean2 = f['mean2'][:]
        lda = f['lda'][:]
    return mean1, lda, mean2


def apply_transform(embeddings, mean1, lda, mean2):
    """Apply LDA transform to embeddings."""
    embeddings = embeddings - mean1
    embeddings = l2_norm(embeddings)
    embeddings = embeddings.dot(lda)
    embeddings = embeddings - mean2
    embeddings = l2_norm(embeddings)
    return embeddings


def VBx(X, Phi, loopProb=0.9, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10,
        epsilon=1e-4, alphaQInit=1.0):
    """
    VBx: Variational Bayes HMM over x-vectors for speaker diarization.
    
    Args:
        X: T x D array of feature vectors (e.g. x-vectors)
        Phi: D array with across-class covariance matrix diagonal
        loopProb: Probability of not switching speakers between frames
        Fa: Scale sufficient statistics
        Fb: Speaker regularization coefficient
        pi: Maximum number of speakers or initialization for speaker priors
        gamma: Initialization for responsibilities matrix
        maxIters: Maximum number of VB iterations
        epsilon: Convergence threshold
        alphaQInit: Dirichlet concentration parameter for initializing gamma
    
    Returns:
        gamma: S x T matrix of responsibilities
        pi: S dimensional speaker priors
        Li: Values of auxiliary function over iterations
    """
    D = X.shape[1]
    
    if type(pi) is int:
        pi = np.ones(pi) / pi
    
    if gamma is None:
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)
    
    assert(gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0])
    
    G = -0.5 * (np.sum(X**2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(Phi)
    rho = X * V
    Li = []
    
    for ii in range(maxIters):
        invL = 1.0 / (1 + Fa/Fb * gamma.sum(axis=0, keepdims=True).T * Phi)
        alpha = Fa/Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(Phi) + G)
        tr = np.eye(len(pi)) * loopProb + (1 - loopProb) * pi
        gamma, log_pX_, logA, logB = forward_backward(log_p_, tr, pi)
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)
        pi = gamma[0] + (1 - loopProb) * pi * np.sum(np.exp(logsumexp(
            logA[:-1], axis=1, keepdims=True) + log_p_[1:] + logB[1:] - log_pX_
        ), axis=0)
        pi = pi / pi.sum()
        Li.append(ELBO)
        
        if ii > 0 and ELBO - Li[-2] < epsilon:
            if ELBO - Li[-2] < 0:
                print('[WARNING] VBx: Value of auxiliary function has decreased!')
            break
    
    return gamma, pi, Li


def forward_backward(lls, tr, ip):
    """Forward-backward algorithm for HMM."""
    eps = 1e-8
    ltr = np.log(tr + eps)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + eps)
    lbw[-1] = 0.0
    
    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)
    
    for ii in reversed(range(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)
    
    tll = logsumexp(lfw[-1], axis=0)
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw
