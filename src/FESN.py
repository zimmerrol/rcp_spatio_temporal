import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

#allows a feedback connection to allow autonomous usage of the reservoir
class FESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0):

        super(FESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, out_activation,
                out_inverse_activation, weight_generation, bias, output_bias, output_input_scaling, feedback=True)


    def fit(self, inputData, outputData, transient_quota=0.05, regression_parameter=None):
        if (self.n_input != 0):
            if (inputData.shape[0] != outputData.shape[0]):
                raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))
        else:
            inputData = np.empty((len(outputData), 0))

        trainLength = outputData.shape[0]
        skipLength = int(trainLength*transient_quota)

        #define states' matrix
        self._X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        oldOutput = np.zeros((self.n_output,1))
        for t in range(trainLength):
            u = super(FESN, self).update_feedback(inputData[t], oldOutput)

            if (t >= skipLength):
                #add valueset to the states' matrix
                self._X[:,t-skipLength] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]
            oldOutput = outputData[t]

        #define the target values
        #                                                               +1
        Y_target = self.out_inverse_activation(outputData).T[:,skipLength:trainLength]

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        if (regression_parameter is None):
            self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))
        else:
            self._W_out = np.dot(np.dot(Y_target, self._X.T),np.linalg.inv(np.dot(self._X,self._X.T) + regression_parameter*np.identity(1+self.n_input+self.n_reservoir)))

        #calculate the training error now
        train_prediction = np.dot(self._W_out, self._X).T
        training_error = np.sqrt(np.mean((train_prediction - outputData[skipLength:])**2))

        return training_error

    def predict(self, inputData, n=None, continuation=True, initial_input_data=None, initial_output_data=None, start_output = None, update_processor=lambda x:x):
        if (self.n_input == 0):
            if (n is None):
                raise ValueError("Either n or n_input have to be greater than zero.")

            inputData = np.empty((n ,0))

        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_input_data is not None):
                for t in range(initial_data.shape[0]):
                    super(FESN, self).update_feedback(initial_data[t], initial_output_data[t])

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))

        if (start_output is None):
            start_output = np.zeros((self.n_output,1))
        oldOutput = start_output
        for t in range(predLength):
            u = super(FESN, self).update_feedback(inputData[t], oldOutput)
            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            Y[:,t] = update_processor(self.out_activation(y[:,0]))
            oldOutput = Y[:,t]

        return Y
