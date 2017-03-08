import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

#uses the last STATE for the predictions
class LSESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.0, input_scaling=None,
                leak_rate=1.0, sparseness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0):

        super(LSESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, out_activation,
                out_inverse_activation, weight_generation, bias, output_bias, output_input_scaling)


    def fit(self, inputDataList, outputData, regression_parameter=None):
        if (inputDataList.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputDataList.shape[0], outputData.shape[0]))

        trialLength = len(inputDataList)
        if (trialLength == 0):
            return

        trainLength = inputDataList[0].shape[0]-1

        #define states' matrix
        self._X = np.zeros((1 + self.n_input + self.n_reservoir, trialLength*1))


        for j in range(trialLength):
            if (np.mod(j,100) == 0):
                print(j)

            self._x = np.zeros((self.n_reservoir,1))
            for t in range(trainLength):
                u = super(LSESN, self).update(inputDataList[j][t])

                #add valueset to the states' matrix
            self._X[:,j*1] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]

        #define the target values
        #                                  +1
        Y_target = self.out_inverse_activation(outputData).T

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        if (regression_parameter is None):
            self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))
        else:
            self._W_out = np.dot(np.dot(Y_target, self._X.T),np.linalg.inv(np.dot(self._X,self._X.T) + regression_parameter*np.identity(1+self.n_input+self.n_reservoir)))


    def predict(self, inputData, update_processor=lambda x:x):
        self._x = np.zeros(self._x.shape)

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output, 1))
        for t in range(predLength):
            u = super(LSESN, self).update(inputData[t])

        y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

        Y[:, 0] = update_processor(self.out_activation(y)).reshape(self.n_output)

        return Y
