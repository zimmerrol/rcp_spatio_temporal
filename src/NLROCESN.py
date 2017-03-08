import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

class NLROCESN(BaseESN):
    def __init__(self, n_input, n_reservoir,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2,
                random_seed=None, weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0):

        super(NLROCESN, self).__init__(n_input, n_reservoir, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, lambda x:x,
                lambda x:x, weight_generation, bias, output_bias, output_input_scaling)

    def fit(self, inputList, outputList, mode="SVC", readout_parameters={}):
        #mode can be SVC, SVR or LR

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
                u = super(NLROCESN, self).update(item[i])
                self._X[:,t] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]
                t+=1

        #define the target values
        Y_target = np.empty((0, 1))
        for item in outputList:
            Y_target = np.append(Y_target, item, axis=0)

        from sklearn.svm import SVC
        from sklearn.svm import SVR
        from sklearn.linear_model import LogisticRegression

        if (mode == "LR"):
            self._clf = LogisticRegression(**readout_parameters)
        elif (mode == "SVR"):
            self._clf = SVR(**readout_parameters)
        else:
            self._clf = SVC(**readout_parameters)
        self._clf.fit(self._X.T, Y_target[:,0])

    def predict(self, inputData):
        self._x = np.zeros(self._x.shape)

        predLength = inputData.shape[0]

        Y = np.zeros((1,predLength))

        X = np.zeros((predLength, self._x.shape[0]+1+self.n_input))

        for t in range(predLength):
            u = super( NLROCESN, self).update(inputData[t])
            X[t,:] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]

        Y = self._clf.predict_proba(X) #_clf.predict(X)

        return Y
