import numpy as np
from numpy.linalg import eig, inv
from numpy import sqrt, trace, diag, argsort
import torch
from torch import nn

class MultiCSP:
    def __init__(self):
        self._Ws = []
        self._ds = []
        self._classes = []
    
    @staticmethod
    def single_csp(x1, x2):
        """
        Args
        ----
            x1 (ndarray) : size of (trials, channels, time). EEG signal matrix
            x2 (ndarray) : size of (trials, channels, time). EEG signal matrix
        Returns
        -------
            W (ndarray)  : size of (channels, channels). Spatial filter matrix that each row vector is a spatial filter.
            d2 (ndarray) : size of (channels,). Eigen values corresponding to CSP spatial filters are sorted in descending order.

        """
        # Maks Covariance matrices
        C1 = []
        for x in x1:
            C1.append((x @ x.T) / trace(x @ x.T))
        M1 = sum(C1)/len(C1)

        C2 = []
        for x in x2:
            C2.append((x @ x.T) / trace(x @ x.T))
        M2 = sum(C2)/len(C2)

        Sigma, U = eig(M1+M2)

        P = inv(diag(sqrt(Sigma))) @ U.T # Whitening matrix

        d1, psi1 = eig(P @ M1 @ P.T)
        d2, psi2 = eig(P @ M2 @ P.T)

        # Order according to eigenvalue
        sort1 = argsort(d1)
        sort2 = argsort(d2)

        d1   = d1[sort1] # Ascending order
        psi1 = psi1[:, sort1]

        d2   = d2[sort2[::-1]] # Descending order
        psi2 = psi2[:, sort2[::-1]]

        # Sanity check
        assert np.all(np.isclose(d1 + d2, np.ones_like(d1)))

        # filters (row vector)
        W = psi2.T @ P

        return W, d2
    
    def fit(self, X, y, classes=None):
        """
        Args
        ----
            X (ndarray) : size of (trials, channels, time). EEG signal matrix.
            y (ndarray) : size of (trials, 1). Labels.
            classes (list) : classes. If None, unique values of y are used.
        """
        assert len(X) == len(y)
        classes = classes if classes else np.unique(y) 
        
        for c in classes:
            mask = y.squeeze() == c
            Wc, dc = self.single_csp(X[mask], X[~mask])
            self._Ws.append(Wc)
            self._ds.append(dc)
            self._classes.append(c)
        
        return self
    
    def transform(self, X):
        """        
        Args
        ----
            X (ndarray) : size of (trials, channels, time). EEG signal matrix.
        Returns
        -------
            Z (ndarray) : size of (trials, C*4, time). C indicates the number of unique class in y. 
                          For each class (one-versus-rest), CSP filters are generated and first and last 2 filters are chosen (total 4).
                          All spatial filtered signal are concatenated in channel axis.
        """
        assert self._Ws, "not yet fitted."
        
        Z = []
        for Wc in self._Ws:
            Z.append(Wc[[0,1,-2,-1],:] @ X) # first and last 2 filters
        
        return np.concatenate(Z, axis=1)


class SEBlock(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=2):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
    