import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

class CESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2,
                random_seed=None, weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0):

        super(CESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, lambda x:x,
                lambda x:x, weight_generation, bias, output_bias, output_input_scaling)


    def fit(self, inputList, outputList, regression_parameter=None):
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
                u = super(CESN, self).update(item[i])
                self._X[:,t] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]
                t+=1

        #define the target values
        #                                  +1
        Y_target = np.empty((0, self.n_output))
        for item in outputList:
            Y_target = np.append(Y_target, item, axis=0)
        Y_target = Y_target.T

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        if (regression_parameter is None):
            self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))
        else:
            self._W_out = np.dot(np.dot(Y_target, self._X.T),np.linalg.inv(np.dot(self._X,self._X.T) + regression_parameter*np.identity(1+self.n_input+self.n_reservoir)))


    def predict(self, inputData):
        self._x = np.zeros(self._x.shape)

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))
        for t in range(predLength):
            u = super(CESN, self).update(inputData[t])
            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            Y[:,t] = y[:,0]

        return Y
