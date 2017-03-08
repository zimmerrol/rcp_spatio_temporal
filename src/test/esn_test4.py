import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# Code for the Sinewave generator
# first test for the feedback ESN (FESN)
# is supposed to generate autonomously a sinewave
# input: target frequency


from FESN import FESN
import numpy as np
import matplotlib.pyplot as plt

def simple_sine():
    generationLength = 20000
    trainLength = 5000
    testLength = generationLength - trainLength

    x = np.linspace(0, 200*np.pi, generationLength)
    y = (np.sin(x)).reshape(generationLength,1)*0.2+0.5

    #x = np.linspace(1,200*np.pi, 20000)
    #y = (0*np.log(x)+np.sin(x)*np.cos(x)).reshape(20000,1)*2


    plt.plot(x,y)
    #plt.show()

    esn = FESN(n_input=0, n_output=1, output_bias=1.0, n_reservoir=400, random_seed=42, noise_level=0.0001, leak_rate=0.70,
        spectral_radius=0.9, weight_generation="advanced", sparseness=0.2)
    esn.fit(inputData=None, outputData=y[1:trainLength,:], transient_quota=0.1)

    print(np.mean(np.abs(esn._W_out)))

    #Y = esn.predict(n=testLength, inputData=None, continuation=True, start_output = y[trainLength-1])
    Y = esn.predict(n=testLength, inputData=None, continuation=True, start_output = y[trainLength])

    Y = Y.T

    print(Y.shape)
    print(x[:trainLength].shape)

    #plt.plot(np.arange(trainLength), Y, 'r')
    plt.plot(x[trainLength:], Y, "r")
    plt.show()

simple_sine()
