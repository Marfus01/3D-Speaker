# Copyright (c) 2024 VBx Integration for 3D-Speaker
# VBx wrapper class for speaker diarization enhancement
# Licensed under the Apache License, Version 2.0

import os
import numpy as np
from datetime import datetime
from scipy.linalg import eigh

from vbx_utils import (
    train_lda_transform, 
    save_transform_h5, 
    load_transform_h5,
    apply_transform,
    VBx as vbx_inference,
    l2_norm
)
from vbx_plda import TwoCovPLDA


class VBxEnhancer:
    """
    VBx-based clustering enhancement wrapper.
    
    This class provides a simple interface to enhance clustering results using VBx.
    It trains LDA transform and PLDA model on initial clustering results, then applies
    VBx inference to smooth the cluster labels.
    
    Args:
        lda_dim: Dimensionality for LDA projection (default: 128)
        Fa: VBx parameter for scaling sufficient statistics (default: 1.0)
        Fb: VBx parameter for speaker regularization (default: 1.0)
        loopP: VBx parameter for speaker transition probability (default: 0.9)
        num_em_iters: Number of EM iterations for PLDA training (default: 5)
        init_smoothing: Smoothing parameter for initialization (default: 5.0)
        max_iters: Maximum VBx iterations (default: 10)
    """
    
    def __init__(self, 
                 lda_dim=128,
                 Fa=1.0,
                 Fb=1.0,
                 loopP=0.9,
                 num_em_iters=5,
                 init_smoothing=5.0,
                 max_iters=10):
        self.lda_dim = lda_dim
        self.Fa = Fa
        self.Fb = Fb
        self.loopP = loopP
        self.num_em_iters = num_em_iters
        self.init_smoothing = init_smoothing
        self.max_iters = max_iters
        
        self.mean1 = None
        self.lda = None
        self.mean2 = None
        self.plda = None
        self.plda_mu = None
        self.plda_tr = None
        self.plda_psi = None
    
    def fit(self, embeddings, labels):
        """
        Train LDA transform and PLDA model on embeddings with initial cluster labels.
        
        Args:
            embeddings: (N, D) numpy array of embeddings
            labels: (N,) numpy array of cluster labels from initial clustering
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {current_time} VBx training started with {len(embeddings)} embeddings, "
              f"{len(np.unique(labels))} initial clusters")
        
        # Step 1: Train LDA transform
        print(f"[INFO] Training LDA transform to {self.lda_dim} dimensions...")
        self.mean1, self.lda, self.mean2 = train_lda_transform(
            embeddings.copy(), labels, self.lda_dim, whiten=False
        )
        
        # Apply LDA transform to embeddings
        transformed_embeddings = apply_transform(embeddings.copy(), self.mean1, self.lda, self.mean2)
        
        # Step 2: Train PLDA model
        print(f"[INFO] Training PLDA model...")
        self.plda = TwoCovPLDA(embed_dim=self.lda_dim)
        self.plda.add_training_data(transformed_embeddings, labels)
        self.plda.train(num_em_iters=self.num_em_iters)
        
        # Step 3: Extract PLDA parameters for VBx
        # Following vbhmm.py:
        # W = inv(plda_tr.T @ plda_tr)  =>  Sigma_w (within-class covariance)
        # B = inv((plda_tr.T / plda_psi) @ plda_tr)  =>  Sigma_b (between-class covariance)
        # Then solve generalized eigenvalue problem: eigh(B, W)
        self.plda_mu = self.plda.mu
        self.plda_tr = self.plda.transform
        self.plda_psi = self.plda.psi
        
        # Re-compute plda_tr and plda_psi for VBx (following vbhmm.py lines 119-124)
        W = np.linalg.inv(self.plda_tr.T.dot(self.plda_tr))
        B = np.linalg.inv((self.plda_tr.T / self.plda_psi).dot(self.plda_tr))
        acvar, wccn = eigh(B, W)
        self.plda_psi = acvar[::-1]  # Diagonal of Phi (descending order)
        self.plda_tr = wccn.T[::-1]  # E matrix
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {current_time} VBx training completed")
    
    def predict(self, embeddings, init_labels):
        """
        Apply VBx inference to smooth cluster labels.
        
        Args:
            embeddings: (N, D) numpy array of embeddings
            init_labels: (N,) numpy array of initial cluster labels
        
        Returns:
            smoothed_labels: (N,) numpy array of smoothed cluster labels
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {current_time} VBx inference started")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Step 1: Apply LDA transform
        x = apply_transform(embeddings.copy(), self.mean1, self.lda, self.mean2)
        
        # Step 2: Apply PLDA transform (following vbhmm.py lines 135-136)
        x = (x - self.plda_mu).dot(self.plda_tr.T)
        
        # Step 3: Initialize gamma from init_labels with smoothing
        num_speakers = len(np.unique(init_labels))
        gamma = np.zeros((len(init_labels), num_speakers))
        for i, label in enumerate(init_labels):
            gamma[i, label] = 1.0
        
        # Apply smoothing (following vbhmm.py lines 146-148)
        gamma = softmax(gamma * self.init_smoothing, axis=1)
        
        # Step 4: Run VBx inference
        print(f"[INFO] Running VBx inference with {num_speakers} speakers...")
        gamma, pi, Li = vbx_inference(
            x, 
            self.plda_psi, 
            loopProb=self.loopP,
            Fa=self.Fa, 
            Fb=self.Fb,
            pi=num_speakers,
            gamma=gamma,
            maxIters=self.max_iters
        )
        
        # Step 5: Extract smoothed labels
        smoothed_labels = np.argmax(gamma, axis=1)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {current_time} VBx inference completed, "
              f"final speakers: {len(np.unique(smoothed_labels))}")
        
        return smoothed_labels
    
    def fit_predict(self, embeddings, init_labels):
        """
        Train and predict in one call.
        
        Args:
            embeddings: (N, D) numpy array of embeddings
            init_labels: (N,) numpy array of initial cluster labels
        
        Returns:
            smoothed_labels: (N,) numpy array of smoothed cluster labels
        """
        self.fit(embeddings, init_labels)
        return self.predict(embeddings, init_labels)
    
    def save_models(self, transform_path, plda_path):
        """
        Save trained models to disk.
        
        Args:
            transform_path: Path to save LDA transform (.h5)
            plda_path: Path to save PLDA model (.h5)
        """
        if self.mean1 is not None and self.lda is not None and self.mean2 is not None:
            save_transform_h5(transform_path, self.mean1, self.lda, self.mean2)
            print(f"[INFO] LDA transform saved to {transform_path}")
        
        if self.plda is not None:
            self.plda.save_model(plda_path)
            print(f"[INFO] PLDA model saved to {plda_path}")
    
    def load_models(self, transform_path, plda_path):
        """
        Load trained models from disk.
        
        Args:
            transform_path: Path to LDA transform (.h5)
            plda_path: Path to PLDA model (.h5)
        """
        self.mean1, self.lda, self.mean2 = load_transform_h5(transform_path)
        print(f"[INFO] LDA transform loaded from {transform_path}")
        
        self.plda = TwoCovPLDA.load_model(plda_path)
        self.plda_mu = self.plda.mu
        self.plda_tr = self.plda.transform
        self.plda_psi = self.plda.psi
        
        # Re-compute for VBx
        W = np.linalg.inv(self.plda_tr.T.dot(self.plda_tr))
        B = np.linalg.inv((self.plda_tr.T / self.plda_psi).dot(self.plda_tr))
        acvar, wccn = eigh(B, W)
        self.plda_psi = acvar[::-1]
        self.plda_tr = wccn.T[::-1]
        
        print(f"[INFO] PLDA model loaded from {plda_path}")


def softmax(x, axis=None):
    """Softmax function for numerical stability."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
