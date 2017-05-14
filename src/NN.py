import numpy as np
from sklearn.neighbors import NearestNeighbors as NearestNeighbors

class NN(object):
    def __init__(self, k=1.0, n_jobs=1):
        self._k = k
        self._n_jobs = n_jobs

    def fit(self, x, y):
      self._neigh = NearestNeighbors(self._k, self._n_jobs, algorithm='kd_tree')
      self._y = y
      self._neigh.fit(x)

    def predict(self, x):
      distances, indices = self._neigh.kneighbors(x)

      with np.errstate(divide='ignore'):
          weights = np.divide(1.0, distances)

      infinity_mask = np.isinf(weights)
      infinity_row_mask = np.any(infinity_mask, axis=1)
      weights[infinity_row_mask] = infinity_mask[infinity_row_mask]

      denominator = np.repeat(np.sum(weights, axis=1), self._k).reshape((-1, self._k))
      weights /= denominator

      prediction = 0
      for i in range(self._k):
          prediction += np.multiply(weights[:, i, np.newaxis], self._y[indices[:, i]])

      return prediction
