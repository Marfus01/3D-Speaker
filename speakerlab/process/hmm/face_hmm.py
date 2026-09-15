"""Independent binary HMMs for observed face-presence channels."""

import numpy as np
from hmmlearn.hmm import CategoricalHMM


class FaceHMM:
    """Fit one HMM per column of F_hat; lengths separates episodes."""

    def __init__(self, n_iter=100, tol=1e-3, random_state=100, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _validate(F_hat, lengths):
        F_hat = np.asarray(F_hat)
        if F_hat.ndim != 2 or 0 in F_hat.shape or not np.isin(F_hat, [0, 1]).all():
            raise ValueError("F_hat must be a nonempty binary matrix.")
        lengths = np.asarray([len(F_hat)] if lengths is None else lengths)
        if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths <= 0) or lengths.sum() != len(F_hat)):
            raise ValueError("lengths must contain positive episode lengths summing to N.")
        return F_hat.astype(int), lengths

    def fit(self, F_hat, lengths=None):
        F_hat, lengths = self._validate(F_hat, lengths)
        self.models_ = []
        for channel in F_hat.T:
            model = CategoricalHMM(
                n_components=2, n_iter=self.n_iter, tol=self.tol,
                random_state=self.random_state, verbose=self.verbose,
                init_params="", params="ste",
                # Tiny pseudocounts keep unvisited transition/emission rows valid.
                startprob_prior=1.000001, transmat_prior=1.000001,
                emissionprob_prior=1.000001,
            )
            model.n_features = 2
            alpha = np.clip(channel.mean(), 0.01, 0.99)
            model.startprob_ = np.array([1 - alpha, alpha])
            model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
            model.emissionprob_ = np.array([[0.9, 0.1], [0.1, 0.9]])
            if np.all(channel == channel[0]):
                # A constant channel cannot identify two states; preserve it.
                model.startprob_ = np.eye(2)[channel[0]]
                model.transmat_ = np.eye(2)
            else:
                model.fit(channel[:, None], lengths)
                # Hidden-state IDs are arbitrary: state 1 must mean face presence.
                order = np.argsort(model.emissionprob_[:, 1])
                model.startprob_ = model.startprob_[order]
                model.transmat_ = model.transmat_[np.ix_(order, order)]
                model.emissionprob_ = model.emissionprob_[order]
            self.models_.append(model)
        self.alpha_ = np.array([model.startprob_[1] for model in self.models_])
        self.A_F_ = np.array([model.transmat_ for model in self.models_])
        self.B_F_ = np.array([model.emissionprob_ for model in self.models_])
        return self

    def predict(self, F_hat, lengths=None):
        F_hat, lengths = self._validate(F_hat, lengths)
        if not hasattr(self, "models_") or F_hat.shape[1] != len(self.models_):
            raise ValueError("Fit the model first, using the same number of channels.")
        return np.column_stack([
            model.predict(F_hat[:, j:j + 1], lengths)
            for j, model in enumerate(self.models_)
        ])
