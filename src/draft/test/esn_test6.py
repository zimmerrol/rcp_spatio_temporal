import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from SGDESN import SGDESN
from CESN import CESN
from NLROCESN import NLROCESN
import numpy as np

from sklearn.utils import shuffle

from io import StringIO
from io import BytesIO

import matplotlib.pyplot as plt

def sineshit():
    x = np.linspace(1,100*np.pi, 10000)
    y = np.sin(x).reshape(10000,1)*2 # (0*np.log(x)+np.sin(x)*np.cos(x)).reshape(20000,1)*2

    esn = SGDESN(n_input=1, n_output=1, n_reservoir=200, random_seed=42, noise_level=0.001, leak_rate=0.8, spectral_radius=0.8, sparseness=0.1, regression_parameters=[2e-5])
    train_error = esn.fit(inputData=y[:5000, :], outputData=y[1:5001,:], transient_quota=0.1, verbose=1)
    print("train error: {0:4f}".format(train_error))
    print(np.mean(np.abs(esn._W_out)))

    Y = esn.generate(n=5000, continuation=True, initial_input=y[5000,:])

    #plt.plot(esn._X.T)
    #plt.show()

    #plt.bar( range(1+1+200), esn._W_out.T )
    #plt.show()

    print(x[5000:].shape)
    print(Y[:, 0].shape)
    plt.plot(x,y[:,0], "b", linestyle="--")
    plt.plot(x[5000:],Y[:, 0], "r", linestyle=":")
    #plt.plot(x[5000:],Y[0,:]-y[5000:,0], linestyle=":")
    plt.show()

sineshit()
