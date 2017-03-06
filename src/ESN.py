import numpy as np
import numpy.random as rnd
import pickle

class ESN:
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparsness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive'):

                self.n_input = n_input
                self.n_reservoir = n_reservoir
                self.n_output = n_output

                self.spectral_radius = spectral_radius
                self.noise_level = noise_level
                self.sparsness = sparsness
                self.leak_rate = leak_rate

                if (input_scaling is None):
                    input_scaling = np.ones(n_input)
                self._input_scaling_matrix = np.diag(input_scaling)

                self.out_activation = out_activation
                self.out_inverse_activation = out_inverse_activation

                if (random_seed is not None):
                    rnd.seed(random_seed)

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
        self._W_input = np.random.rand(self.n_reservoir, 1+self.n_input)-0.5

    def fit(self, inputData, outputData, transient_quota=0.05, regression_parameter=None):
        if (inputData.shape[0] != outputData.shape[0]):
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        trainLength = inputData.shape[0]-1
        skipLength = int(trainLength*transient_quota)

        #define states' matrix
        self._X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        for t in range(trainLength):
            u = self._input_scaling_matrix.dot(inputData[t].reshape(self.n_input, 1))
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level
            if (t >= skipLength):
                #add valueset to the states' matrix
                self._X[:,t-skipLength] = np.vstack((1,u, self._x))[:,0]

        #define the target values
        #                                  +1
        Y_target = self.out_inverse_activation(outputData).T[:,skipLength+1:trainLength+1]

        #W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
        if (regression_parameter is None):
            self._W_out = np.dot(Y_target, np.linalg.pinv(self._X))
        else:
            self._W_out = np.dot(np.dot(Y_target, self._X.T),np.linalg.inv(np.dot(self._X,self._X.T) + regression_parameter*np.identity(1+self.n_input+self.n_reservoir)))

    def generate(self, n, initial_input, continuation=True, initial_data=None, update_processor=lambda x:x):
        if (self.n_input != self.n_output):
            raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    u = self._input_scaling_matrix.dot(initial_data[t].reshape(self.n_input, 1))
                    self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level

        predLength = n

        Y = np.zeros((self.n_output,predLength))
        u = self._input_scaling_matrix.dot(initial_input.reshape(self.n_input, 1))
        for t in range(predLength):
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x))
            y = np.dot(self._W_out, np.vstack((1,u,self._x)))
            Y[:,t] = update_processor(self.out_activation(y[:,0]))
            u = y

        return Y

    def predict(self, inputData, initial_input, continuation=True, initial_data=None, update_processor=lambda x:x):
        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    u = initial_data[t].reshape(self.n_input, 1)
                    self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x)) + (np.random.rand()-0.5)*self.noise_level

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))
        u = initial_input.reshape(self.n_input,1)
        for t in range(predLength):
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((1, u))) + np.dot(self._W, self._x))
            y = np.dot(self._W_out, np.vstack((1,u,self._x)))
            Y[:,t] = update_processor(self.out_activation(y[:,0]))

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
