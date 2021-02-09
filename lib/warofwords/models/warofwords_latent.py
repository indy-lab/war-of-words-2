import numpy as np
import torch as pt

from ..features import ParameterVector
from .base import Model, TrainedModel
from .utils import BatchHelper


class WarOfWordsLatent(Model):

    # Default hyperparams. Doesn't include default value for explicit features.
    DEFAULT_HYPERPARAMS = {
        'reg_vec': 0.1,
        'n_dims': 10,
        'lr': 0.1,
        'lr_vec': 0.1,
        'gamma': 0.9,
        'n_epochs': 5,
        'batch_size': 1000,
        'tol': 1e-5,
    }

    def __init__(self, data, features, hyperparameters, verbose=False):
        """Initialize the model.

        :data: List of tuples of feature matrix and label index.
        :features: Instance of helpers.Features.
        :hyperparameters: Hyperparameters of the model.
        :bias_key: Optional. Key of bias in feautres.
        """
        super().__init__()
        self._data = data
        self._features = features
        # Set hyperparameters.
        self._hyperparameters = self.DEFAULT_HYPERPARAMS
        self._hyperparameters.update(hyperparameters)
        # Get index of bias.
        self._bias_idx = features.get_idx('bias')
        self._verbose = verbose
        # Get MEPs and dossier indices.
        self._meps = set(self._features.get_group('mep'))
        self._doss = set(self._features.get_group('dossier'))

    def _init_params(self):
        hp = self._hyperparameters
        meps = self._features.get_group('mep')
        doss = self._features.get_group('dossier')
        D = hp['n_dims']
        M = len(meps)
        N = len(doss)
        # Initialize parameters.
        params = pt.zeros(len(self._features))
        params.requires_grad_()  # In-place.
        # Initialize latent vectors.
        vec = pt.zeros((len(self._features), D))
        lo, hi = -1e-3, 1e-3
        pt.random.manual_seed(42)
        vec[meps] = (lo - hi) * pt.rand((M, D)) + hi
        pt.random.manual_seed(43)
        vec[doss] = (lo - hi) * pt.rand((N, D)) + hi
        vec.requires_grad_()  # In-place.

        return params, vec

    def _get_dossier_latent_vector(self, X, vec):
        K = len(X)
        # Go through all sparse entries.
        sparse_idx = X.coalesce().indices().T
        for i, j in sparse_idx:
            # Skip if not dossier feature vector.
            if int(i) != K - 1:
                continue
            # Return latent vector of dossier for this data point.
            j = int(j)
            if j in self._doss:
                return vec[j, :]

    def _get_meps_latent_vector(self, X, vec):
        K = len(X)
        # Initialize MEP latent features.
        vec_x = pt.zeros(K - 1, self._hyperparameters['n_dims'])
        # Go through all sparse entries.
        sparse_idx = X.coalesce().indices().T
        for i, j in sparse_idx:
            # Skip if dossier feature vector.
            if int(i) == K - 1:
                continue
            # Construct MEP latent vector for each edit.
            j = int(j)
            if j in self._meps:
                vec_x[i, :] += vec[j, :]
        return vec_x

    def _logits(self, X, params, vec):
        # X = pt.from_numpy(X).float()
        # Compute logits.
        logits = X.mm(params.unsqueeze(1))
        # Get dossier latent features.
        # last = X[-1, :]
        # vec_y = vec[doss][last[doss] > 0].transpose(0, 1)
        vec_y = self._get_dossier_latent_vector(X, vec)
        # Get MEPs latent features.
        # Xm = X[:-1, :][:, self._meps]
        # vec_x = Xm.matmul(vec[self._meps].unsqueeze(1))
        vec_x = self._get_meps_latent_vector(X, vec)
        # Compute latent features.
        latent_feat = vec_x.mm(vec_y.unsqueeze(1))
        # Compute logits with latent features.
        logits[:-1] += latent_feat
        return logits.squeeze()

    def _probabilities(self, X, params, vec):
        # Compute logits.
        logits = self._logits(X, params, vec)
        # Compute probabilities.
        num = pt.exp(logits)
        norm = pt.sum(num)
        return num / norm

    def accuracy(self, params, vec, data=None):
        if data is None:
            data = self._data
        acc = 0
        for X, y in data:
            prob = self._probabilities(X, params, vec)
            pred = np.argmax(prob)
            if pred == y:
                acc += 1
        return acc / len(data)

    def log_loss(self, params, vec, data=None):
        return -self.log_likelihood(params, vec, data) / len(data)

    def log_likelihood(self, params, vec, data=None):
        """Compute the log-likelihood of the parameters given the data."""
        if data is None:
            data = self._data
        llh = 0
        for X, y in data:
            # Compute logits.
            logits = self._logits(X, params, vec)
            # Compute log(softmax( y | X )) in a stable way.
            llh += logits[y]  # Log of numerator.
            llh -= pt.logsumexp(logits, dim=0)  # Log of denominator.
        return llh

    def _objective(self, params, vec, data, n_batches=1):
        hp = self._hyperparameters
        # Regularize each feature group separately.
        val = 0
        for group, reg in hp.items():
            indices = self._features.get_group(group)
            if len(indices) > 0:
                val += reg * pt.sum(params[indices] ** 2)
        # Regularize latent vectors.
        val += hp['reg_vec'] * pt.sum(vec ** 2)
        # Normalize by the number of batches as we compute the loss for each
        # batch separately.
        return val / n_batches - self.log_likelihood(params, vec, data)

    def fit(self, validation_data=None):
        verbose = self._verbose
        """Fit the model."""
        hp = self._hyperparameters
        # Initialize parameters.
        params, vec = self._init_params()
        # Set optimizer.
        conf_params = [
            {'params': params, 'lr': hp['lr']},
            {'params': vec, 'lr': hp['lr_vec']},
        ]
        # optimizer = pt.optim.SGD(conf_params)
        # optimizer = pt.optim.Adam(conf_params)
        optimizer = pt.optim.Adagrad(conf_params)
        scheduler = pt.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hp['gamma']
        )
        # Set minibatches.
        batches = BatchHelper(
            self._data,
            batch_size=hp['batch_size'],
            shuffle=True,
            drop_last=True,
            seed=10,
        )
        n_batches = len(batches)
        batch_losses = list()
        vlos = -1
        if validation_data is not None:
            # Compute "empty" validation loss
            valid_losses = list()
            with pt.no_grad():
                vlos = float(self.log_loss(params, vec, validation_data))
                valid_losses.append(vlos)
        for e in range(hp['n_epochs']):
            for i, batch in enumerate(batches):
                # Gradient step.
                optimizer.zero_grad()
                loss = self._objective(params, vec, batch, n_batches)
                loss.backward()
                optimizer.step()
                # Check for NaNs.
                if pt.isnan(params).any():
                    raise ValueError('NaN in parameters! Fitting stopped.')
                if pt.isnan(vec).any():
                    raise ValueError('NaN in latent vectors! Fitting stopped.')
                # Compute batch log-loss.
                with pt.no_grad():
                    tlos = self.log_loss(params, vec, batch)
                    if pt.isnan(tlos).any():
                        raise ValueError('NaN in log loss! Fitting stopped.')
                    batch_losses.append(float(tlos))
                if verbose:
                    self._print_progress(
                        i, e, n_batches, loss, tlos, vlos, params, vec
                    )
            # Update learning rate.
            scheduler.step()
            # Compute validation log-loss for the current epoch.
            if validation_data is not None:
                with pt.no_grad():
                    vlos = float(self.log_loss(params, vec, validation_data))
                    valid_losses.append(vlos)
        # Create return value of parameters.
        base = params.detach().numpy()
        params_return = {
            'params': ParameterVector(self._features, base=base),
            'vec': vec.detach().numpy(),
        }
        if validation_data is None:
            with pt.no_grad():
                # Compute cost on the whole training data (similar to
                # scipy.optimize)
                cost = self._objective(params, vec, self._data)
            return params_return, cost
        else:
            return params_return, batch_losses, valid_losses

    def _print_progress(self, i, e, n_batches, loss, tlos, vlos, params, vec):
        s = f'Epoch {e+1} ({(i+1)/n_batches*100:.2f}%): '
        s += f'cost={loss.item():.3f} '
        s += f'train-loss={tlos:.4f} '
        if vlos > 0:
            s += f'valid-loss={vlos:.4f} '
        s += f'params-norm={params.norm().item():.2f} '
        textidx = self._features.get_group('text-embedding')
        if len(textidx) > 0:
            textnorm = params[textidx].norm().item()
            s += f'txt-embd-norm={textnorm:.2f} '
        s += f'vec-norm={vec.norm().item():.2f} '
        s += f'bias={params[self._bias_idx]:.4f} '
        # print(s, end='\r')
        print(s)


class TrainedWarOfWordsLatent(TrainedModel):
    def __init__(self, features, hyperparams, params, vec):
        super().__init__()
        self.features = features
        self.hyperparameters = hyperparams
        self.parameters = params
        # Numpy array of params for faster computations.
        self._params = params.as_array()
        self._vec = vec

    def probabilities(self, X):
        # Compute logits.
        logits = X.dot(self._params)
        # Get MEPs and dossiers indices.
        meps = self.features.get_group('mep')
        doss = self.features.get_group('dossier')
        # Get dossier latent feature.
        last = X[-1, :]
        vec_y = self._vec[doss][last[doss] > 0].squeeze()
        # Get MEPs latent features.
        Xm = X[:-1, :][:, meps]
        vec_x = Xm.dot(self._vec[meps]).squeeze()
        # Compute latent features.
        latent_feat = vec_x.dot(vec_y)
        # Compute logits with latent features.
        logits[:-1] += latent_feat
        # Compute probabilities.
        num = np.exp(logits)
        norm = np.sum(num)
        return num / norm

    def accuracy(self, data):
        acc = 0
        for X, y in data:
            prob = self.probabilities(X)
            pred = np.argmax(prob)
            if pred == y:
                acc += 1
        return float(acc / len(data))

    def log_loss(self, data):
        loss = 0
        for X, y in data:
            prob = self.probabilities(X)
            loss -= np.log(prob[y])
        return float(loss / len(data))
