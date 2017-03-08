import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from ESN import ESN
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

    esn = ESN(n_input=1, n_output=1, n_reservoir=200, random_seed=42, noise_level=0.001, leak_rate=0.7, spectral_radius=1.35, sparseness=0.1)
    esn.fit(inputData=y[:5000, :], outputData=y[:5000,:], transient_quota=0.4, regression_parameter=2e-4)

    Y = esn.generate(n=15000, continuation=True, initial_input=y[5000,:])

    print(x[5000:].shape)
    print(Y[0,:].shape)
    plt.plot(x,y[:,0], "b", linestyle="--")
    plt.plot(x[5000:],Y[0,:], "r", linestyle=":")
    #plt.plot(x[5000:],Y[0,:]-y[5000:,0], linestyle=":")
    plt.show()

#reset the reservoir after each series!

def read_data(path, sizeString):
    content = ""
    with open(path) as f:
        content = f.readlines()

    indices = [i for i, x in enumerate(content) if x == "\n"]
    indices = np.array(indices)
    indices = np.hstack((0,indices, len(content)-1))

    data = []

    mini = np.zeros((1,12))
    for i in range(indices.shape[0]-1):
        dset = ''.join(content[indices[i]:indices[i+1]])
        dset = BytesIO(dset.encode())
        dset = np.genfromtxt(dset, delimiter=' ')

        for i in range(12):
            mini[0,i] = min(mini[0,i], np.min(dset[:,i]))

#        dset = np.vstack((dset,np.average(dset[:3,:], axis=0)))
#        dset = np.vstack((dset,np.average(dset[:5,:], axis=0)))
#        dset = np.vstack((dset,np.average(dset[:7,:], axis=0)))
#        dset = np.vstack((dset,np.average(dset[:9,:], axis=0)))

        data.append(dset)

    y = []
    ind = 0
    size = sizeString.split(' ')
    for i in range(len(size)):
        for j in range(int(size[i])):
            #print(ind)
            row = np.zeros((data[ind].shape[0],9))
            row[:,i] = 1.0
            y.append(row.copy())
            ind += 1
    return data,y

trainX, trainY = read_data("ae.train", "30 30 30 30 30 30 30 30 30")
testX, testY = read_data("ae.test", "31 35 88 44 29 24 40 50 29")

esn = CESN(n_input=12, n_output=9, n_reservoir=1000, leak_rate=0.8, spectral_radius=0.2, random_seed=41, noise_level=0.01, sparseness=0.30)
esn.fit(inputList=trainX, outputList=trainY)

err = 0.0
for i in range(len(testX)):
    yy = esn.predict(testX[i])*10
    maxi=np.argmax(yy, axis=0)
    sel = np.argmax(np.bincount(maxi))
    err += ((sel - np.argmax(testY[i])) != 0)
    if (((sel - np.argmax(testY[i])) != 0) == 1):
        print(i)
print(err)
print("{:0.2f}%".format(100.0/370*err))
