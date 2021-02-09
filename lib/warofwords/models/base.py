import pickle

import numpy as np


class Model:

    """Base class to define a model."""

    def __init__(self):
        pass

    @staticmethod
    def load_data(path, sparse=False):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if sparse:
            return data['features'], data['feature_matrices'], data['labels']
        else:
            numfeat = len(data['features'])
            feature_matrices = list()
            for matrix in data['feature_matrices']:
                mat = list()
                for feat in matrix:
                    vec = np.zeros(numfeat)
                    for idx, val in feat:
                        vec[idx] = val
                    mat.append(vec)
                feature_matrices.append(np.array(mat))
            features = data['features']
            labels = data['labels']

        return features, feature_matrices, labels


class TrainedModel(Model):

    """Base class to define a trained model."""

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
