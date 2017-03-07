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
    x = np.zeros((Ti, 2))
    x[:,0] = np.random.rand(Ti)

    n1 = np.random.random_integers(1,int(0.1*T0))
    n2 = np.random.random_integers(1+int(0.1*T0),int(0.5*T0))

    x[n1,1] = 1.0
    x[n2,1] = 1.0

    y = (x[n1,0]*x[n2,0])

    return x,y

def generate_trial():
    T0 = 1000
    Ti = int(np.random.uniform(T0, T0*1.1))

    N = 1000
    x_train = np.empty((N,Ti,2))
    y_train = np.empty(N)
    for i in range(N):
        x, y = generate_run(Ti,T0)
        x_train[i,:] = x
        y_train[i] = y

    return x_train, y_train

x_train, y_train = generate_trial()

#esn = SESN(n_input=2, n_output=1, n_reservoir=100, random_seed=42, noise_level=0.0000, leak_rate=0.0001, spectral_radius=3.0, sparsness=0.1,
#            weight_generation='naive', bias=0.0, output_bias=0.0, output_input_scaling=1.0, input_scaling=[0.01, 0.01])
esn = LSESN(n_input=2, n_output=1, n_reservoir=300, random_seed=42, noise_level=0.0000, leak_rate=0.00001, spectral_radius=12.0, sparsness=0.1,
            weight_generation='naive', bias=0.0, output_bias=0.0, output_input_scaling=0.0, input_scaling=[0.05, 1.0])
esn.fit(inputDataList=x_train, outputData=y_train)

print(np.mean(np.abs(esn._W_out)))

error = 0.0
x_test, y_test = generate_trial()#x_train, y_train#
for i in range(len(x_test)):
    Y = esn.predict(inputData=x_test[i,:,:])
    err = np.abs(Y.flatten()[0]-y_test[i].flatten()[0])
    error += (err>0.005)#err#err#
    print(Y.flatten()[-2:])
    print(y_test[i].flatten()[-2:])
    #if (input("continue?") == "n"):
    #    break
    if (np.mod(i,100) == 0):
        print(i)
print(error)
