import numpy as np
import numpy.random as rnd
import pickle

#uses the last STATE for the predictions
class LSESN:
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparsness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0):

                self.n_input = n_input
                self.n_reservoir = n_reservoir
                self.n_output = n_output

                self.spectral_radius = spectral_radius
                self.noise_level = noise_level
                self.sparsness = sparsness
                self.leak_rate = leak_rate

                if (input_scaling is None):
                    input_scaling = np.ones(n_input)
                if (np.isscalar(input_scaling)):
                    input_scaling = np.repeat(input_scaling, n_input)
                else:
                    if (len(input_scaling) != self.n_input):
                        raise ValueError("Dimension of input_scaling ({0}) does not match the input data dimension ({1})".format(len(input_scaling), n_input))

                self._input_scaling_matrix = np.diag(np.append([1.0],input_scaling))

                self.out_activation = out_activation
                self.out_inverse_activation = out_inverse_activation

                if (random_seed is not None):
                    rnd.seed(random_seed)

                self.bias = bias
                self.output_bias = output_bias
                self.output_input_scaling = output_input_scaling
                self._create_reservoir(weight_generation)

    def _create_reservoir(self, weight_generation):
        if (weight_generation == 'naive'):
            #random weight matrix from -0.5 to 0.5
            self._W = rnd.rand(self.n_reservoir, self.n_reservoir) - 0.5

            #set sparsness% to zero
            mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparsness
            self._W[mask] = 0.0

            _W_eigenvalues = np.abs(np.linalg.eig(self._W)[0])
            self._W *= self.spectral_radius / np.max(_W_eigenvalues)

        elif (weight_generation == 'advanced'):
            #two create W we must follow some steps:
            #at first, create a W = |W|
            #make it sparse
            #then scale its spectral radius to rho(W) = 1 (according to Yildiz with x(n+1) = (1-a)*x(n)+a*f(...))
            #then change randomly the signs of the matrix

            #random weight matrix from 0 to 0.5
            self._W = rnd.rand(self.n_reservoir, self.n_reservoir) / 2

            #set sparseness% to zero
            mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparsness
            self._W[mask] = 0.0

            _W_eigenvalues = np.abs(np.linalg.eig(self._W)[0])
            self._W *= self.spectral_radius / np.max(_W_eigenvalues)

            #change random signs
            random_signs = np.power(-1, rnd.random_integers(self.n_reservoir, self.n_reservoir))
            self._W = np.multiply(self._W, random_signs)

        else:
            raise ValueError("The weight_generation property must be one of the following values: naive, advanced")

        #random weight matrix for the input from -0.5 to 0.5
        self._W_input = np.random.rand(self.n_reservoir, self.n_input+1)-0.5
        self._W_input = self._W_input.dot(self._input_scaling_matrix)

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
                u = inputDataList[j][t].reshape(self.n_input, 1)
                self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level
                #print(self._W_input)
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
            u = inputData[t].reshape(self.n_input,1)
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x))

        y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

        Y[:, 0] = update_processor(self.out_activation(y)).reshape(self.n_output)

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
