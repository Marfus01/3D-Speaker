# Copyright (c) 2022 Shuai Wang, 2024 VBx Contributors
# Two-cov PLDA implementation for VBx
# Licensed under the Apache License, Version 2.0

import os, sys
import collections
import numpy as np
import h5py
from numpy.linalg import inv

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vbx_utils import compute_normalizing_transform, sort_svd

ClassInfo = collections.namedtuple('ClassInfo', ['weight', 'num_example', 'mu'])


class PldaStats(object):
    """Statistics for PLDA training."""
    
    def __init__(self, dim):
        self.dim = dim
        self.num_example, self.num_classes = 0, 0
        self.class_weight, self.example_weight = 0, 0
        self.sum_ = np.zeros(dim)
        self.offset_scatter = np.zeros((dim, dim))
        self.classinfo = []
    
    def add_samples(self, weight, spk_embeddings):
        """Add samples of a certain speaker to the PLDA stats."""
        n = spk_embeddings.shape[0]
        mean = np.mean(spk_embeddings, axis=0)
        tmp = spk_embeddings - mean
        self.offset_scatter += weight * np.matmul(tmp.T, tmp)
        self.classinfo.append(ClassInfo(weight, n, mean))
        self.num_example += n
        self.num_classes += 1
        self.class_weight += weight
        self.example_weight += weight * n
        self.sum_ += weight * mean


class TwoCovPLDA:
    """Two-covariance PLDA model."""
    
    def __init__(self, embed_dim=256):
        self.dim = embed_dim
        self.mu = np.zeros(self.dim)
        self.transform = np.zeros((self.dim, self.dim))
        self.psi = np.zeros(self.dim)
        self.offset = np.zeros(self.dim)
        self.stats = PldaStats(self.dim)
        self.B = np.eye(self.dim)
        self.B_stats = np.zeros((self.dim, self.dim))
        self.B_count = 0
        self.W = np.eye(self.dim)
        self.W_stats = np.zeros((self.dim, self.dim))
        self.W_count = 0
    
    def add_training_data(self, embeddings, labels):
        """
        Add training data from embeddings and labels.
        
        Args:
            embeddings: (N, D) array of embeddings
            labels: (N,) array of cluster labels
        """
        # Group embeddings by label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            spk_embeddings = embeddings[mask]
            self.stats.add_samples(1.0, spk_embeddings)
    
    def train(self, num_em_iters=5):
        """Train PLDA model with EM iterations."""
        print(f"[INFO] Training PLDA with {num_em_iters} EM iterations...")
        for i in range(num_em_iters):
            print(f"  PLDA EM iteration {i+1}/{num_em_iters}")
            self.em_one_iter()
        self.get_output()
        print("[INFO] PLDA training completed")
    
    def em_one_iter(self):
        """One EM iteration."""
        self.B_stats = np.zeros((self.stats.dim, self.stats.dim))
        self.B_count = 0
        self.W_stats = np.zeros((self.stats.dim, self.stats.dim))
        self.W_count = 0
        self.W_stats += self.stats.offset_scatter
        self.W_count += self.stats.example_weight - self.stats.class_weight
        B_inv = inv(self.B)
        W_inv = inv(self.W)
        
        for i in range(self.stats.num_classes):
            info = self.stats.classinfo[i]
            m = info.mu - self.stats.sum_ / self.stats.class_weight
            weight = info.weight
            n = info.num_example
            mix_var = inv(B_inv + n * W_inv)
            w = np.matmul(mix_var, n * np.matmul(W_inv, m))
            m_w = m - w
            self.B_stats += weight * (mix_var + np.outer(w, w))
            self.B_count += weight
            self.W_stats += weight * n * (mix_var + np.outer(m_w, m_w))
            self.W_count += weight
        
        self.W = self.W_stats / self.W_count
        self.B = self.B_stats / self.B_count
        self.W = 0.5 * (self.W + self.W.T)
        self.B = 0.5 * (self.B + self.B.T)
        
        print(f"    W_count: {self.W_count:.2f}, Trace of W: {np.trace(self.W):.4f}")
        print(f"    B_count: {self.B_count:.2f}, Trace of B: {np.trace(self.B):.4f}")
    
    def get_output(self):
        """Extract output parameters from trained model."""
        self.mu = self.stats.sum_ / self.stats.class_weight
        transform1 = compute_normalizing_transform(self.W)
        B_proj = np.matmul(transform1, self.B)
        B_proj = np.matmul(B_proj, transform1.T)
        s, U = np.linalg.eigh(B_proj)
        s = np.where(s > 0.0, s, 0.0)
        s, U = sort_svd(s, U)
        
        self.transform = np.matmul(U.T, transform1)
        self.psi = s
        self.offset = np.zeros(self.dim)
        self.offset = -1.0 * np.matmul(self.transform, self.mu)
    
    def save_model(self, output_file):
        """Save PLDA model to h5 file."""
        print(f"[INFO] Saving PLDA model to {output_file}")
        with h5py.File(output_file, "w") as f:
            f.create_dataset("mu", data=self.mu, compression="gzip", fletcher32=True)
            f.create_dataset("transform", data=self.transform, compression="gzip", fletcher32=True)
            f.create_dataset("psi", data=self.psi, compression="gzip", fletcher32=True)
            f.create_dataset("offset", data=self.offset, compression="gzip", fletcher32=True)
    
    @staticmethod
    def load_model(model_file):
        """Load PLDA model from h5 file."""
        print(f"[INFO] Loading PLDA model from {model_file}")
        with h5py.File(model_file, "r") as f:
            mu = f["mu"][:]
            transform = f["transform"][:]
            psi = f["psi"][:]
            offset = f["offset"][:]
        
        plda = TwoCovPLDA(embed_dim=mu.shape[0])
        plda.mu = mu
        plda.transform = transform
        plda.psi = psi
        plda.offset = offset
        return plda
