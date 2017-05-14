import numpy as np

class RBF(object):
    def __init__(self, sigma=5.0, basisQuota = 0.05):
        self._sigma = sigma
        self._basisQuota = basisQuota

    def rbf(xi, yi, sigmam):
        return np.exp(-np.sum((xi-yi)**2)/(2*sigmam**2))

    def rbf_vec(xi, yi, sigmam):
        xi = np.tile(xi, (len(yi),1))
        return np.exp(-np.sum((xi-yi)**2, axis=1)/(2*sigmam**2))

    def fit(self, x, y):
        self._nprime = y.shape[1]

        n = len(x)

        #m is the numberOfSamplingPoints
        m = int(self._basisQuota*len(x))

        #according to http://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        createEqualSpacedIndices = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]


        self._samplingPoints = x[createEqualSpacedIndices(m, len(x))]
        self._sigmam = np.ones(m)*self._sigma

        #construct matrices
        A = np.empty((n, m))
        F = np.empty((n, self._nprime))
        L = None

        #construct target matrix F
        #F = flat_u_data_train.copy()

        #construct training matrix A
        for i in range(n):
          A[i] = RBF.rbf_vec(x[i], self._samplingPoints, self._sigmam)

        #calculate the resulting weights L
        APINV = np.linalg.pinv(A)
        L = np.dot(APINV, y) #F

        self._Lt = L.T

    def predict(self, x):
        prediction = np.zeros((len(x), self._nprime))

        for i in range(0, prediction.shape[0]):
            prediction[i] = np.dot(self._Lt, RBF.rbf_vec(x[i], self._samplingPoints, self._sigmam))

        return prediction
