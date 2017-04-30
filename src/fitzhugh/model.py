import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import signal

from ESNFHN import ESNFHN

deltaT = 1e-2
maxT = 500.0
Iext = 2.0
tSteps = int(ceil(maxT/deltaT))
tValues = np.arange(tSteps)*deltaT

def I(t):
    if (t > 10.0):# and t < 100.0):
        return (np.random.rand()-0.5)*Iext #np.sin(t)*Iext
    else:
        return 0.0

def step(z, t):
    global a, b, r, deltaT
    v, w = z

    v = v + deltaT*(v-1.0/3.0*v**3-w+I(t))
    w = w + deltaT/r*(v-a-b*w)
    return np.array([v, w])

def testFHN():
    global a, b, r
    r = 1.0/.2 #1.0/0.08
    a = -0.7
    b = 0.8

    values = np.empty((tSteps, 2))
    values[0] = np.array([0.0, 0.0])

    for i, t in enumerate(tValues[1:]):
        values[i+1] = step(values[i], t)

    plt.plot(tValues, values)
    plt.plot(tValues, np.tanh(values), linestyle=":")
    plt.show()

def testESNFHN():
    def sineshit():
        x = np.linspace(1, 100*np.pi, 20000)
        #y = (0*np.log(x)+np.sin(x)*np.cos(x)).reshape(20000,1)*2
        y = np.sin(x).reshape(20000,1)

        esn = ESNFHN(n_input=1, n_output=1, n_reservoir=50, random_seed=42, noise_level=0.00001, output_bias=1.0, output_input_scaling=0.0, r=1/.05, deltaT=1e-1, spectral_radius=1.20, sparseness=0.2, solver="pinv", regression_parameters=[1e-2])
        train_error = esn.fit(inputData=y[:5000, :], outputData=y[1:5001,:], transient_quota=0.4) #,

        print("mean weigth:" + str(np.mean(np.abs(esn._W_out))))

        print(esn._W_out.shape)

        plt.bar(range(1+1+esn.n_reservoir), esn._W_out.T)
        plt.show()

        plt.plot(esn._X.T)
        plt.show()


        Y = esn.generate(n=15000-1, continuation=True, initial_input=y[5000,:])

        print("train error: {0:4f}".format(train_error))
        test_error = np.mean((Y-y[5001:])**2)
        print("test error: {0:4f}".format(test_error))

        print(x[5000:19999].shape)
        print(Y[:, 0].shape)
        plt.plot(x,y[:,0], "b", linestyle="--")
        plt.plot(x[5001:20000],Y[:, 0], "r", linestyle=":")
        plt.plot([x[5000], x[5000]], [-5,5], "g")
        plt.ylim([-2, 2])

        #plt.plot(x[5000:],Y[0,:]-y[5000:,0], linestyle=":")
        plt.show()

    sineshit()

#testFHN()

testESNFHN()
