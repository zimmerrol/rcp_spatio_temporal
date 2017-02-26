import numpy as np
import numpy.random as rnd
import pickle

class ESN:
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01,
                leak_rate=1.0, sparity=0.2,
                random_seed=None):

                self.n_input = n_input
                self.n_reservoir = n_reservoir
                self.n_output = n_output

                self.spectral_radius = spectral_radius
                self.noise_level = noise_level
                self.sparity = sparity
                self.leak_rate = leak_rate

                if (random_seed is not None):
                    rnd.seed(random_seed)

                self._create_reservoir()

    def _create_reservoir(self):
        #random weight matrix from -0.5 to 0.5
        self._W = rnd.rand(self.n_reservoir, self.n_reservoir) - 0.5

        #set sparity% to zero
        mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparity
        self._W[mask] = 0.0

        _W_eigenvalues = np.abs(np.linalg.eig(self._W)[0])
        self._W *= self.spectral_radius / np.max(_W_eigenvalues)

        #random weight matrix for the input from -0.5 to 0.5
        self._W_input = np.random.rand(self.n_reservoir, 1+self.n_input)-0.5

    def fit(self, inputData, outputData, transient_quota=0.05):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        trainLength = inputData.shape[0]-1
        skipLength = int(trainLength*transient_quota)

        #define states' matrix
        self._X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        for t in range(trainLength):
            u = inputData[t].reshape(self.n_input, 1)
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level
            if (t >= skipLength):
                #add valueset to the states' matrix
                self._X[:,t-skipLength] = np.vstack((1,u, self._x))[:,0]

        #define the target values
        #                                  +1
        Y_target = outputData.T[:,skipLength+1:trainLength+1]

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))

    def generate(self, n, initial_input, continuation=True):
        if (not continuation):
            self._x = np.zeros(self._x.shape)

        predLength = n

        Y = np.zeros((self.n_output,predLength))
        u = initial_input.reshape(self.n_input,1)
        for t in range(predLength):
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x))
            y = np.dot(self._W_out, np.vstack((1,u,self._x)))
            Y[:,t] = y[:,0]
            u = y

        return Y

    def predict(self, inputData, initial_input, continuation=True):
        if (not continuation):
            self._x = np.zeros(self._x.shape)

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))
        u = initial_input.reshape(self.n_input,1)
        for t in range(predLength):
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x))
            y = np.dot(self._W_out, np.vstack((1,u,self._x)))
            Y[:,t] = y[:,0]

            u = inputData[t].reshape(self.n_input,1)

        return Y

    def save(self, path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def load(path):
        f = open(path, "rb")
        result = pickle.load(f)
        f.close()
        return result
