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
                bias=1.0, output_bias=0.0,
                learning_rate=1e-3, optimization_rate=1e-3, sin=1.0, regression_parameters=[2e-4]):

        super(SGDESN, self).__init__(n_input, n_reservoir, n_output, spectral_radius, noise_level, input_scaling, leak_rate, sparseness, random_seed, lambda x:x,
                lambda x:x, "naive", bias, output_bias, output_input_scaling=1.0)

        self.learning_rate = learning_rate
        self.optimization_rate = optimization_rate
        self._regression_parameters = regression_parameters

        self.sin = sin

    def fit(self, inputData, outputData, transient_quota=0.05, verbose=0):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        trainLength = inputData.shape[0]

        skipLength = int(trainLength*transient_quota)

        #define states' matrix
        X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(trainLength):
            u = super(ESN, self).update(inputData[t])
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

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )


        if (self._solver == "pinv"):
            """print("pinv")
            import pycuda.autoinit
            import pycuda.driver as drv
            import pycuda.gpuarray as gpuarray
            import skcuda.linalg as culinalg
            import skcuda.misc as cumisc
            culinalg.init()

            X_gpu = gpuarray.to_gpu(X)
            X_inv_gpu = culinalg.pinv(X_gpu)
            Y_gpu = gpuarray.to_gpu(Y_target)
            W_out_gpu = Y_gpu * W_out_gpu
            pred_gpu = W_out_gpu * X_gpu

            self._W_out = gpuarray.from_gpu(W_out_gpu)
            """
            self._W_out = np.dot(Y_target, np.linalg.pinv(X))

            #calculate the training error now
            train_prediction = self.out_activation((np.dot(self._W_out, X)).T)

        elif (self._solver == "lsqr"):
            self._W_out = np.dot(np.dot(Y_target, X.T),np.linalg.inv(np.dot(X,X.T) + self._regression_parameters[0]*np.identity(1+self.n_input+self.n_reservoir)))

            #calculate the training error now
            train_prediction = self.out_activation(np.dot(self._W_out, X).T)

        elif (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
            mode = self._solver[8:]
            params = self._regression_parameters
            params["solver"] = mode
            self._ridgeSolver = Ridge(**params)

            self._ridgeSolver.fit(X.T, Y_target.T)
            train_prediction = self.out_activation(self._ridgeSolver.predict(X.T))

        elif (self._solver in ["sklearn_svr", "sklearn_svc"]):
            self._ridgeSolver = SVR(**self._regression_parameters)

            self._ridgeSolver.fit(X.T, Y_target.T.ravel())
            train_prediction = self.out_activation(self._ridgeSolver.predict(X.T))

        """
        #alternative represantation of the equation

        Xt = X.T

        A = np.dot(X, Y_target.T)

        B = np.linalg.inv(np.dot(X, Xt)  + regression_parameter*np.identity(1+self.n_input+self.n_reservoir))

        self._W_out = np.dot(B, A)
        self._W_out = self._W_out.T
        """

        X = None

        training_error = np.sqrt(np.mean((train_prediction - outputData[skipLength:])**2))
        return training_error

    def sgd_fit(self, inputData, outputData, transient_quota=0.05, verbose=0):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        total_trainLength = inputData.shape[0]
        skipLength = int(total_trainLength*transient_quota)
        pre_trainLength = int(np.ceil(total_trainLength/5)+skipLength)
        trainLength = total_trainLength-pre_trainLength

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

        sgd_step_size = 100

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        dxdleakrate_prev = 0.0
        dxdrho_prev = 0.0
        dxdsin_prev = 0.0

        for t in range(trainLength):
            u = super(SGDESN, self).update(inputData[t+pre_trainLength])

            """
            #add valueset to the states' matrix
            state_history[:,t] = self._x

            #update W_out
            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            error = outputData[t+pre_trainLength]-y
            u = np.vstack((self.bias , u))

            self._W_out = self._W_out + self.learning_rate*np.dot(error, np.vstack((state_history[:,t-1], u)).T)
            """

            y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))
            error = output-y

            dxdleakrate = (1.0-self.leak_rate)*dxdleakrate_prev - x_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*self._x) + np.exp(-2*self._x))), np.dot(self._W, dxdleakrate_prev))

            dxdrho = (1.0-self.leak_rate)*dxdrho_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*self._x) + np.exp(-2*self._x))), np.dot(self.spectral_radius*self._W, dxdrho_prev) + np.dot(self._W,x_prev))

            dxdsin = (1.0-self.leak_rate)*dxdsin_prev + np.multiply(self.leak_rate*(4/(2 + np.exp(2*self._x) + np.exp(-2*self._x))),np.dot(self.spectral_radius*self._W, dxdsin_prev) + np.dot(self._W_input,u))

            new_leak_rate -= self.optimization_rate* (-np.dot(error.T, np.dot(self._W_out, np.vstack((dxdleakrate, np.zeros((self.n_input, 1)))))))
            new_spectral_radius -= self.optimization_rate* (-np.dot(error.T, np.dot(self._W_out, np.vstack((spectral_radius, np.zeros((self.n_input, 1)))))))
            #new_input_scaling -= self.optimization_rate* (-np.dot(error_next.T, np.dot(self._W_out, np.vstack((dxdsin, np.zeros((self.n_input, 1)))))))

            dxdsin_prev = dxdsin
            dxdrho_prev = dxdrho
            dxdleakrate_prev = dxdleakrate

            self._W = self._W * self.spectral_radius / new_spectral_radius
            self.spectral_radius = new_spectral_radius

            self.leak_rate = new_leak_rate


            if (t % sgd_step_size == 0):
                old_states = X.copy()

                fit_results = fit(inputData, outputData, transient_quota)
                print(fit_results)

                X = old_states.copy()


















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

        dxdleakrate = (1.0-self.leak_rate)*dxdleakrate_prev - x_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*self._x) + np.exp(-2*self._x))), np.dot(self._W, dxdleakrate_prev))

        dxdrho = (1.0-self.leak_rate)*dxdrho_prev + np.multiply(self.leak_rate*(4/(2+np.exp(2*self._x) + np.exp(-2*self._x))), np.dot(self.spectral_radius*self._W, dxdrho_prev) + np.dot(self._W,x_prev))

        dxdsin = (1.0-self.leak_rate)*dxdsin_prev + np.multiply(self.leak_rate*(4/(2 + np.exp(2*self._x) + np.exp(-2*self._x))),np.dot(self.spectral_radius*self._W, dxdsin_prev) + np.dot(self._W_input,u))

        new_leak_rate -= self.optimization_rate* (-np.dot(error.T, np.dot(self._W_out, np.vstack((dxdleakrate, np.zeros((self.n_input, 1)))))))
        new_spectral_radius -= self.optimization_rate* (-np.dot(error.T, np.dot(self._W_out, np.vstack((spectral_radius, np.zeros((self.n_input, 1)))))))
        #new_input_scaling -= self.optimization_rate* (-np.dot(error_next.T, np.dot(self._W_out, np.vstack((dxdsin, np.zeros((self.n_input, 1)))))))

        dxdsin_prev = dxdsin
        dxdrho_prev = dxdrho
        dxdleakrate_prev = dxdleakrate

        self._W = self._W * self.spectral_radius / new_spectral_radius
        self.spectral_radius = new_spectral_radius

        self.leak_rate = new_leak_rate


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
