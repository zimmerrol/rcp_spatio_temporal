import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

#improves the output of keras on Windows
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import progressbar


from keras.models import Sequential
from keras.layers import Dense


class ffnnESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive', bias=1.0, output_bias=1.0,
                output_input_scaling=1.0):

        super(ffnnESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, out_activation,
                out_inverse_activation, weight_generation, bias, output_bias, output_input_scaling)


    def fit(self, inputData, outputData, transient_quota=0.05, verbose=0):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        trainLength = inputData.shape[0]

        skipLength = int(trainLength*transient_quota)

        self._x = np.zeros((self.n_reservoir,1))

        #define states' matrix
        X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        # create model
        self.model = Sequential()
        self.model.add(Dense((1+self.n_input+self.n_reservoir)*2, input_dim=1+self.n_input+self.n_reservoir, activation='sigmoid'))
        self.model.add(Dense((1+self.n_input+self.n_reservoir), activation='sigmoid'))
        #model.add(Dense((1+self.n_input+self.n_reservoir)//4, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile model
        self.model.compile(loss='hinge', optimizer='adam')


        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(trainLength):
            u = super(ffnnESN, self).update(inputData[t])
            if (t >= skipLength):
                #add valueset to the states' matrix
                X[:,t-skipLength] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        #define the target values
        #                                  +1
        Y_target = self.out_inverse_activation(outputData).T[:,skipLength:]

        self.model.fit(X.T, Y_target.T, nb_epoch=30, batch_size=10, verbose=2)
        # evaluate the model
        scores = self.model.evaluate(X.T, Y_target.T)
        print(scores)

        train_prediction = self.out_activation(self.model.predict(X.T))

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )



        #train_prediction = self.out_activation(self._ridgeSolver.predict(X.T))


        X = None

        training_error = np.sqrt(np.mean((train_prediction - outputData[skipLength:])**2))
        return training_error

    def generate(self, n, initial_input, continuation=True, initial_data=None, update_processor=lambda x:x):
        if (self.n_input != self.n_output):
            raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    #TODO Fix
                    super(ffnnESN, self).update(initial_data[t])

        predLength = n

        Y = np.zeros((self.n_output,predLength))
        inputData = initial_input
        for t in range(predLength):
            u = super(ffnnESN, self).update(inputData)

            y = self.model.predict(np.vstack((self.output_bias, self.output_input_scaling*u, self._x)).T)

            #y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            y = self.out_activation(y[:,0])
            Y[:,t] = update_processor(y)
            inputData = y

        return Y.T

    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    super(ffnnESN, self).update(initial_data[t])

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=predLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(predLength):
            u = super(ffnnESN, self).update(inputData[t])

            if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd", "sklearn_svr"]):
                y = self._ridgeSolver.predict(np.vstack((self.output_bias, self.output_input_scaling*u, self._x)).T).reshape((-1,1))
            else:
                y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

            Y[:,t] = update_processor(self.out_activation(y[:,0]))
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        return Y.T
