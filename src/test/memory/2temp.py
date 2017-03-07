import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from LSESN import LSESN
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_run(Ti, T0):
    x = np.zeros((Ti, 6))
    index = np.random.random_integers(low=0, high=3, size=Ti)

    x[np.arange(len(x)), index] = 1.0

    n1 = np.random.random_integers(int(0.1*Ti),int(0.2*T0))
    n2 = np.random.random_integers(int(0.5*T0),int(0.6*T0))

    ind1 = np.random.random_integers(low=4, high=5)
    ind2 = np.random.random_integers(low=4, high=5)
    x[n1] = 0.0
    x[n2] = 0.0
    x[n1,ind1] = 1.0
    x[n2,ind2] = 1.0

    if (ind1 == 5 and ind2 == 5):
        y = [1.0, 0.0, 0.0, 0.0]
    elif (ind1 == 5 and ind2 == 4):
        y = [0.0, 1.0, 0.0, 0.0]
    elif (ind1 == 4 and ind2 == 5):
        y = [0.0, 0.0, 1.0, 0.0]
    else:
        y = [0.0, 0.0, 0.0, 1.0]

    return x,y

def generate_trial():
    T0 = 200
    Ti = int(np.random.uniform(T0, T0*1.1))

    N = 500
    x_train = np.empty((N,Ti,6))
    y_train = np.empty((N,4))
    for i in range(N):
        x, y = generate_run(Ti,T0)
        x_train[i,:] = x
        y_train[i] = y

    return x_train, y_train

x_train, y_train = generate_trial()

np.set_printoptions(linewidth=200)

#it is somehow strange, that we have to set the first 4 inputs to zero for this test... there is probably somewhere a big error!
esn = LSESN(n_input=6, n_output=4, n_reservoir=100, random_seed=42, noise_level=0.0000, leak_rate=0.0001, spectral_radius=0.65, sparsness=0.1,
            weight_generation='naive', bias=0.0, output_bias=0.0, output_input_scaling=0.0, input_scaling=[0,0,0,0,.15,15])
esn.fit(inputDataList=x_train, outputData=y_train)

print(np.mean(np.abs(esn._W_out)))
#print(esn._W_out)


error = 0.0
x_test, y_test = generate_trial()#x_train, y_train#
for i in range(len(x_test)):
    Y = esn.predict(inputData=x_test[i,:,:])
    #err = np.abs(Y.flatten()[0]-y_test[i].flatten()[0])
    #error += (err>0.005)#err#err#

    #print((np.abs(Y.reshape(4) - y_test[i]) > 0.5))
    if ((np.abs(Y.reshape(4) - y_test[i]) > 0.5).any()):
        error += 1
    #print(Y)
    #print(y_test[i])
    #if (input("continue?") == "n"):
    #    break
    if (np.mod(i,100) == 0):
        print(i)
print(error)
