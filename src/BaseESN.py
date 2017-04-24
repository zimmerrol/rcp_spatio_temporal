import numpy as np
import numpy.random as rnd
#import pickle
import dill as pickle
import scipy as sp

class BaseESN(object):
    def __init__(self, n_input, n_reservoir, n_output,
                spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                leak_rate=1.0, sparseness=0.2, random_seed=None,
                out_activation=lambda x:x, out_inverse_activation=lambda x:x,
                weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0,
                feedback=False, scale_input_matrix=False):

                self.n_input = n_input
                self.n_reservoir = n_reservoir
                self.n_output = n_output

                self.spectral_radius = spectral_radius
                self.noise_level = noise_level
                self.sparseness = sparseness
                self.leak_rate = leak_rate

                if (input_scaling is None):
                    input_scaling = np.ones(n_input)
                if (np.isscalar(input_scaling)):
                    input_scaling = np.repeat(input_scaling, n_input)
                else:
                    if (len(input_scaling) != self.n_input):
                        raise ValueError("Dimension of input_scaling ({0}) does not match the input data dimension ({1})".format(len(input_scaling), n_input))

                self._input_scaling_matrix = np.diag(input_scaling)
                self._expanded_input_scaling_matrix = np.diag(np.append([1.0], input_scaling))

                self.out_activation = out_activation
                self.out_inverse_activation = out_inverse_activation

                if (random_seed is not None):
                    rnd.seed(random_seed)

                self.bias = bias
                self.output_bias = output_bias
                self.output_input_scaling = output_input_scaling
                self._create_reservoir(weight_generation, feedback)

    def _create_reservoir(self, weight_generation, feedback=False):
        if (weight_generation == 'naive'):
            #random weight matrix from -0.5 to 0.5
            self._W = rnd.rand(self.n_reservoir, self.n_reservoir) - 0.5

            #set sparseness% to zero
            mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparseness
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
            print("aa")
            self._W = np.empty((self.n_reservoir, self.n_reservoir))# rnd.rand(self.n_reservoir, self.n_reservoir) / 2.0
            for i in range(self.n_reservoir):
                self._W[i] = rnd.rand(self.n_reservoir)/2.0
                #if (i % 1000 == 0):
                    #print(i)
        
            #set sparseness% to zero
            print("aa")
            mask = rnd.choice(a=[False, True], size=(self.n_reservoir, self.n_reservoir), p=[self.sparseness,1-self.sparseness])      
            self._W[mask] = 0.0
     
            #just calculate the largest EV - hopefully this is the right code to do so...
            _W_eigenvalue = np.max(np.abs(sp.sparse.linalg.eigs(self._W, k=1)[0]))
            #_W_eigenvalue = np.max(np.abs(np.linalg.eig(self._W)[0]))
 
            self._W *= self.spectral_radius / _W_eigenvalue


            #change random signs
            random_signs = np.power(-1, rnd.random_integers(self.n_reservoir, self.n_reservoir))
            self._W = np.multiply(self._W, random_signs)

        else:
            raise ValueError("The weight_generation property must be one of the following values: naive, advanced")

        #random weight matrix for the input from -0.5 to 0.5
        self._W_input = np.random.rand(self.n_reservoir, 1+self.n_input)-0.5
        self._W_input = self._W_input.dot(self._expanded_input_scaling_matrix)

        if (feedback):
            self._W_feedback = np.random.rand(self.n_reservoir, 1+self.n_output) - 0.5

    def update(self, inputData):
        """
        returns the UNSCALED but reshaped input of this step
        """
        u = inputData.reshape(self.n_input, 1)
        self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x) + (np.random.rand()-0.5)*self.noise_level)

        return u

    def update_feedback(self, inputData, outputData):
        """
        returns the UNSCALED but reshaped input of this step
        """
        #the input is allowed to be "empty" (size=0)
        if (self.n_input != 0):
            u = inputData.reshape(self.n_input, 1)
            outputData = outputData.reshape(self.n_output, 1)
            #TODO: Fix the brackets arround the noise!
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x) +
                np.dot(self._W_feedback, np.vstack((self.output_bias, outputData))) + (np.random.rand()-0.5)*self.noise_level)

            return u
        else:
            outputData = outputData.reshape(self.n_output, 1)
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W, self._x) + np.dot(self._W_feedback, np.vstack((self.output_bias, outputData))) +
                (np.random.rand()-0.5)*self.noise_level)

            return np.empty((0,1))

    def save(self, path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def load(path):
        f = open(path, "rb")
        result = pickle.load(f)
        f.close()
        return result
