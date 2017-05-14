class RBF(object):
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def rbf(xi, yi, sigmam):
        return np.exp(-np.sum((xi-yi)**2)/(2*sigmam**2))

    def rbf_vec(xi, yi, sigmam):
        xi = np.tile(xi, (len(yi),1))
        return np.exp(-np.sum((xi-yi)**2, axis=1)/(2*sigmam**2))

    def fit(self, x, y, basisQuota = 0.05)
        numberOfSamplingPoints = int(basisQuota*len(x))

        #according to http://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        createEqualSpacedIndices = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]


        self._samplingPoints = x[createEqualSpacedIndices(numberOfSamplingPoints, len(x))]
        self._sigmam = np.ones(numberOfSamplingPoints)*self.sigma

        #construct matrices
        A = np.empty((n, m))
        F = np.empty((n, nprime))
        L = None

        #construct target matrix F
        #F = flat_u_data_train.copy()

        #construct training matrix A
        for i in range(n):
          A[i] = rbf_vec(x[i], samplingPoints, sigmam)

        #calculate the resulting weights L
        APINV = np.linalg.pinv(A)
        L = np.dot(APINV, y) #F

        self._Lt = L.T

    def predict(self, x):
        flat_prediction = np.zeros((testLength, self._nprime))

        for i in range(0, flat_prediction.shape[0]):
            flat_prediction[i] = np.dot(self._Lt, rbf_vec(x[i], self._samplingPoints, self._sigmam))

        pred = flat_prediction.ravel()

        return pred
