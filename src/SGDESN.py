import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import progressbar


class SGDESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2, random_seed=None,
                weight_generation='naive', bias=1.0, output_bias=0.0,
                learning_rate=1e-3, optimization_rate=1e-3, sin=1.0, regression_parameters=[2e-4]):

        super(SGDESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, lambda x:x,
                lambda x:x, weight_generation, bias, output_bias, output_input_scaling=1.0)

        self.learning_rate = learning_rate
        self.optimization_rate = optimization_rate
        self._regression_parameters = regression_parameters

        self.sin = sin

    def fit(self, inputData, outputData, transient_quota=0.05, verbose=0):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        total_trainLength = inputData.shape[0]
        skipLength = int(total_trainLength*transient_quota)
        pre_trainLength = int(np.ceil(total_trainLength/5)+skipLength)
        trainLength = total_trainLength-pre_trainLength
        print(skipLength)
        print(total_trainLength)
        print(pre_trainLength)
        print(trainLength)

        #define states' matrix
        X = np.zeros((1+self.n_input+self.n_reservoir,pre_trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        for t in range(pre_trainLength):
            u = super(SGDESN, self).update(inputData[t])
            if (t >= skipLength):
                #add valueset to the states' matrix
                X[:,t-skipLength] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]


        Y_target = self.out_inverse_activation(outputData).T[:,skipLength:pre_trainLength]
        self._W_out = np.dot(np.dot(Y_target, X.T),np.linalg.inv(np.dot(X,X.T) + self._regression_parameters[0]*np.identity(1+self.n_input+self.n_reservoir)))

        state_history = np.zeros((self.n_reservoir, trainLength, 1))
        print(state_history.shape)
        print("pre train finished. ")
        pre_train_prediction = self.out_activation(np.dot(self._W_out, X).T)
        pre_training_error = np.sqrt(np.mean((pre_train_prediction - outputData[skipLength:pre_trainLength])**2))
        print("pre train error: {0}".format(pre_training_error))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(trainLength):
            u = super(SGDESN, self).update(inputData[t+pre_trainLength])

            #add valueset to the states' matrix
            state_history[:,t] = self._x

            #update W_out
            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            error = outputData[t+pre_trainLength]-y
            u = np.vstack((self.bias , u))

            self._W_out = self._W_out + self.learning_rate*np.dot(error, np.vstack((state_history[:,t-1], u)).T)

            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        return 0.0

    def _optimize_W_out(u, output_d, x_prev):
        #following: http://212.201.49.24/sites/default/files/uploads/papers/leakyESN.pdf
        y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
        error = output-y
        self._W_out = self._W_out + self.learning_rate*np.dot(error, np.vstack((x_prev, u).T))

    def _optimize_parameters(u, output_d, x_prev):
        y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
        error = output-y

        dxdleakrate = (1.0-self.leak_rate)*dxdleakrate_prev - x_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*X) + np.exp(-2*X))), np.dot(self._W, dxdleakrate_prev))

        dxdrho = (1.0-self.leak_rate)*dxdrho_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*X) + np.exp(-2*X))), np.dot(self.spectral_radius*self._W, dxdrho_prev) + np.dot(self._W,x_prev))

        dxdsin = (1.0-self.leak_rate)*dxdsin_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*X) + np.exp(-2*X))),np.dot(self.spectral_radius*self._W, dxdsin_prev) + np.dot(self._W_input,u))

        self.leak_rate -= self.optimization_rate* (-np.dot(error_next.T, np.dot(self._W_out, np.vstack((dxdleakrate, np.zeros((self.n_input, 1)))))))
        self.spectral_radius -= self.optimization_rate* (-np.dot(error_next.T, np.dot(self._W_out, np.vstack((spectral_radius, np.zeros((self.n_input, 1)))))))
        self.sin -= self.optimization_rate* (-np.dot(error_next.T, np.dot(self._W_out, np.vstack((dxdsin, np.zeros((self.n_input, 1)))))))

    def generate(self, n, initial_input, continuation=True, initial_data=None, update_processor=lambda x:x):
        if (self.n_input != self.n_output):
            raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    #TODO Fix
                    super(SGDESN, self).update(initial_data[t])

        predLength = n

        Y = np.zeros((self.n_output,predLength))
        inputData = initial_input
        for t in range(predLength):
            u = super(SGDESN, self).update(inputData)

            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

            y = self.out_activation(y[:,0])
            Y[:,t] = update_processor(y)
            inputData = y

        return Y.T

    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    super(SGDESN, self).update(initial_data[t])

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=predLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(predLength):
            u = super(SGDESN, self).update(inputData[t])

            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

            Y[:,t] = update_processor(self.out_activation(y[:,0]))
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        return Y.T
