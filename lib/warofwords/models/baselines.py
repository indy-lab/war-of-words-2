import numpy as np

from collections import Counter, defaultdict
from dataclasses import dataclass
from ..features import ParameterVector, Features
from .base import Model, TrainedModel


class Random(Model):

    @dataclass
    class Hyperparameters:
        pass

    def __init__(self, data, features, hyperparameters, verbose):
        super().__init__()

    def log_likelihood(self, params):
        return 0

    def fit(self):
        return {'params': ParameterVector([])}, -1


class TrainedRandom(TrainedModel):

    def __init__(self, features, hyperparams, params):
        super().__init__()

    def probabilities(self, X):
        N = len(X)
        return [1 / N for _ in range(N)]

    def accuracy(self, data):
        acc = 0
        for X, y in data:
            pred = np.random.randint(len(X))
            if pred == y:
                acc += 1
        return acc / len(data)

    def log_loss(self, data):
        loss = 0
        for X, y in data:
            prob = self.probabilities(X)
            loss -= np.log(prob[y])
        return loss / len(data)


class StrongRandom(Model):

    @dataclass
    class Hyperparameters:
        pass

    def __init__(self, data, features, hyperparameters, verbose):
        self.data = data
        super().__init__()

    def log_likelihood(self, params):
        return 0

    def fit(self):
        # Compute proportion of time the dossier wins.
        count = 0
        for featmat, label in self.data:
            if label == featmat.shape[0] - 1:
                count += 1
        p = count / len(self.data)
        # Store this probability p that the dossier wins.
        features = Features()
        features.add('p')
        params = features.new_parameters()
        params['p'] = p
        return {'params': params}, -1


class TrainedStrongRandom(TrainedModel):

    def __init__(self, features, hyperparams, params):
        self.parameters = params
        super().__init__()

    def probabilities(self, X):
        # Number of conlficts between MEPs.
        n_conflicts = len(X) - 1
        p = self.parameters['p']
        # Probability vector tuned to the probability that the dossier wins.
        return [(1-p)/n_conflicts] * n_conflicts + [p]

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


class StrongestRandom(Model):

    @dataclass
    class Hyperparameters:
        pass

    def __init__(self, data, features, hyperparameters, verbose):
        self.data = data
        super().__init__()

    def log_likelihood(self, params):
        return 0

    def fit(self):
        # Compute proportion of time the dossier wins per conflict size.
        countwins = defaultdict(int)
        countconflicts = list()
        for featmat, label in self.data:
            size = len(featmat)
            if label == size - 1:
                countwins[size] += 1
            countconflicts.append(size)
        countconflicts = Counter(countconflicts)
        # Compute prior probability of dossier winning.
        p = (sum([v for v in countwins.values()])
             / sum([v for v in countconflicts.values()]))
        # Compute prior per dossier.
        ps = dict()
        features = Features()
        features.add('default', group='conflict-size')
        for size in countconflicts:
            ps[size] = countwins[size] / countconflicts[size]
            features.add('p-' + str(size), group='conflict-size')
        # Store this probability p that the dossier wins.
        params = features.new_parameters()
        params['default'] = p
        for size, p in ps.items():
            params['p-' + str(size)] = p
        return {'params': params}, -1


class TrainedStrongestRandom(TrainedModel):

    def __init__(self, features, hyperparams, params):
        self.parameters = params
        super().__init__()

    def probabilities(self, X):
        # Number of conlficts between MEPs.
        size = len(X)
        n_conflicts = size - 1
        key = 'p-' + str(size)
        p = self.parameters[key]
        # Probability vector tuned to the probability that the dossier wins.
        return [(1-p)/n_conflicts] * n_conflicts + [p]

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
