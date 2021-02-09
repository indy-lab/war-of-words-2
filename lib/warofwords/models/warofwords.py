import warnings

import numpy as np
# Remove Numba warnings.
from numba.errors import NumbaPendingDeprecationWarning
from numba.typed import List
from scipy.optimize import minimize

from ..features import ParameterVector
from .base import Model, TrainedModel
from .warofwords_jit import gradient_jit, log_likelihood_jit, probabilities_jit

warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class WarOfWords(Model):

    """Class for managing a WarOfWords model."""

    def __init__(self, data, features, hyperparameters, verbose=None):
        """Initialize the model.

        :data: List of tuples of feature matrix and label index.
        :features: Instance of helpers.Features.
        :hyperparameters: Hyperparameters of the model.

        The hyperparameters are given in a dict {group: value}, where the group
        is a feature group and the value is the value of the regularizer.
        """
        super().__init__()
        # Format data for Numba.
        self._data = List()
        self._data.extend(data)
        # Set features and hyperparameters.
        self._features = features
        self._hyperparameters = hyperparameters
        # Check that hyperparameter groups exist as feature groups.
        groups = set(features.groups())
        for group in hyperparameters:
            if group not in groups:
                raise ValueError(f'Group "{group}" not in features')
        self._verbose = verbose

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """Compute the gradient of the *negative* log-likelihood with
        regularization, i.e., the objective function to minimize.
        """
        theta = params.copy()
        for group, reg in self._hyperparameters.items():
            indices = self._features.get_group(group)
            theta[indices] *= 2 * reg
        return theta - gradient_jit(self._data, params)

    def _objective(self, params: np.ndarray) -> float:
        """Compute the regularized negative log-likelihood of the parameters"""
        val = 0
        for group, reg in self._hyperparameters.items():
            indices = self._features.get_group(group)
            val += reg * np.sum(params[indices] ** 2)
        return val - self.log_likelihood(params)

    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute the log-likelihood of the parameters given the data."""
        return log_likelihood_jit(self._data, params)

    def fit(self, *, maxiter=15000, tol=1e-5):
        x0 = self._features.new_parameters().as_array()
        options = {'gtol': tol, 'maxiter': maxiter, 'disp': self._verbose}
        res = minimize(
            fun=self._objective,
            x0=x0,
            method='L-BFGS-B',
            jac=self._gradient,
            options=options,
        )
        self.converged = res.success
        params = ParameterVector(self._features, base=res.x)
        return {'params': params}, res.fun


class TrainedWarOfWords(TrainedModel):
    def __init__(self, features, hyperparams, params):
        super().__init__()
        self.features = features
        self.hyperparameters = hyperparams
        self.parameters = params
        # Numpy array of params for faster computations.
        self._params = params.as_array()

    def probabilities(self, X):
        return probabilities_jit(X, self._params)

    def accuracy(self, data):
        acc = 0
        for X, y in data:
            prob = self.probabilities(X)
            if np.argmax(prob) == y:
                acc += 1
        return acc / len(data)

    def log_loss(self, data):
        loss = 0
        for X, y in data:
            prob = self.probabilities(X)
            loss -= np.log(prob[y])
        return loss / len(data)
