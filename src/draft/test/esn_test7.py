import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from ffnnESN import ffnnESN
from CESN import CESN
from NLROCESN import NLROCESN
import numpy as np

from sklearn.utils import shuffle

from io import StringIO
from io import BytesIO

import matplotlib.pyplot as plt

def sineshit():
    x = np.linspace(1,200*np.pi, 20000)
    y = (0*np.log(x)+np.sin(x)*np.cos(x)).reshape(20000,1)*2

    esn = ffnnESN(n_input=1, n_output=1, n_reservoir=200, random_seed=42, noise_level=0.001, leak_rate=0.7, spectral_radius=1.35, sparseness=0.1)
    train_error = esn.fit(inputData=y[:5000, :], outputData=y[1:5001,:], transient_quota=0.4)
    print("train error: {0:4f}".format(train_error))

    Y = esn.generate(n=15000, continuation=True, initial_input=y[5000,:])

    #plt.plot(esn._X.T)
    #plt.show()

    #plt.bar( range(1+1+200), esn._W_out.T )
    #plt.show()

    print(np.mean((y[5000:,0] - Y[:, 0])**2))

    print(x[5000:].shape)
    print(Y[:, 0].shape)
    plt.plot(x,y[:,0], "b", linestyle="--")
    plt.plot(x[5000:],Y[:, 0], "r", linestyle=":")
    #plt.plot(x[5000:],Y[0,:]-y[5000:,0], linestyle=":")
    plt.show()

sineshit()
