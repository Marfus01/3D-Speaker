import logging
import sys
from collections import deque

import numpy as np


_log = logging.getLogger(__name__)
#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class ConvergenceMonitor:
    """
    Monitor and report convergence to :data:`sys.stderr`.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.
    iter : int
        Number of iterations performed while training the model.

    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.

    >>> from hmmlearn.base import ConvergenceMonitor
    >>> from hmmlearn import hmm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = hmm.GaussianHMM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """

    _template = "{iter:>10d} {log_prob:>16.8f} {delta:>+16.8f}"

    def __init__(self, tol, n_iter, verbose):
        """
        Parameters
        ----------
        tol : double
            Convergence threshold.  EM has converged either if the maximum
            number of iterations is reached or the log probability improvement
            between the two consecutive iterations is less than threshold.
        n_iter : int
            Maximum number of iterations to perform.
        verbose : bool
            Whether per-iteration convergence reports are printed.
        """
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, log_prob):
        """
        Report convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        log_prob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = log_prob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, log_prob=log_prob, delta=delta)
            print(message, file=sys.stderr)

        # Allow for some wiggleroom based on precision.
        precision = np.finfo(float).eps ** (1/2)
        if self.history and (log_prob - self.history[-1]) < -precision:
            delta = log_prob - self.history[-1]
            _log.warning(f"Model is not converging.  Current: {log_prob}"
                         f" is not greater than {self.history[-1]}."
                         f" Delta is {delta}")
        self.history.append(log_prob)
        self.iter += 1

    @property
    def converged(self):
        """Whether the EM algorithm converged."""
        # XXX we might want to check that ``log_prob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))

