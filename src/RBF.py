import numpy as np

class RBF(object):
    def __init__(self, basisPoints, sigma=5.0):
        self._sigma = sigma
        self._basisPoints = basisPoints

    def rbf(xi, yi, sigmam):
        return np.exp(-np.sum((xi-yi)**2)/(2*sigmam**2))

    def rbf_vec(xi, yi, sigmam):
        xi = np.tile(xi, (len(yi),1))
        return np.exp(-np.sum((xi-yi)**2, axis=1)/(2*sigmam**2))

    def fit(self, x, y):
        self._nprime = y.shape[1]

        n = len(x)

        #m is the numberOfSamplingPoints
        m = self._basisPoints

        #according to http://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        createEqualSpacedIndices = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]


        self._samplingPoints = x[createEqualSpacedIndices(m, len(x))]


        self._samplingPoints = x[createEqualSpacedIndices(m, len(x))]

        """
        print(self._samplingPoints[i].shape)

        distances = np.empty((len(self._samplingPoints)-1, len(self._samplingPoints)-1))
        for i in range(len(self._samplingPoints)):
            for j in range(i, len(self._samplingPoints)):
                distances[i, j] = np.mean((self._samplingPoints[i]-self._samplingPoints[j])**2)
                distances[j, i] = distances[i, j]
        """

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
        from numpy.linalg.linalg import LinAlgError
        try:
            APINV = np.linalg.pinv(A)
            L = np.dot(APINV, y) #F

            self._Lt = L.T
        except LinAlgError as err:
            print("SVD did NOT converge")
            print("#of NaNs: " + str(np.count_nonzero(np.isnan(A))))
            print("#of Infs: " + str(np.count_nonzero(np.isinf(A))))
            raise err

    def predict(self, x):
        prediction = np.zeros((len(x), self._nprime))

        for i in range(0, prediction.shape[0]):
            prediction[i] = np.dot(self._Lt, RBF.rbf_vec(x[i], self._samplingPoints, self._sigmam))

        return prediction
