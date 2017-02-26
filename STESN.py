import numpy as np
import numpy.random as rnd
import pickle

class STESN:
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

    def fit(self, inputList, outputList):
        if (len(inputList) != len(outputList)):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(len(inputList), len(outputList)))

        trainLength = 0
        for item in inputList:
            trainLength += item.shape[0]
        skipLength = 0

        #define states' matrix
        self._X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        t = 0
        for item in inputList:
            self._x = np.zeros((self.n_reservoir,1))

            for i in range(item.shape[0]):
                u = item[i].reshape(self.n_input, 1)
                self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level
                self._X[:,t] = np.vstack((1,u, self._x))[:,0]
                t+=1

        #define the target values
        #                                  +1
        Y_target = np.empty((0, self.n_output))
        for item in outputList:
            Y_target = np.append(Y_target, item, axis=0)
        Y_target = Y_target.T

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))

    def predict(self, inputData):
        self._x = np.zeros(self._x.shape)

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))
        for t in range(predLength):
            u = inputData[t].reshape(self.n_input,1)
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x))
            y = np.dot(self._W_out, np.vstack((1,u,self._x)))
            Y[:,t] = y[:,0]

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
