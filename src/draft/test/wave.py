import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from helper import *
from ESN import ESN

def generate_data(N, Ngrid=150):
    x = np.linspace(0,1, Ngrid)
    X = np.tile(np.array(x), len(x)).reshape((-1, len(x)))

    w = 0.5
    wavelength = 0.5
    k = 2*np.pi/wavelength

    t = 0
    deltaT = 0.05

    result = np.empty((N, Ngrid, Ngrid))
    for i in range(N):
        result[i] = np.sin(k*X-w*t)
        t += deltaT

    return result

def generate_data2(N, Ngrid=150):
    t = 0
    deltaT = 0.05
    deltaX = 0.1
    c = 1.0

    epsilon = (deltaX/(c*deltaT))**2

    print(epsilon)

    def init_fn(x):
        val = np.exp(-(x**2)/0.25)
        if val<.001:
            return 0.0
        else:
            return val

    u = np.zeros((N, Ngrid))
    u[0:10, 0:1] = 1.0

    """
    for a in range(0,Ngrid):
        u[0,a]=init_fn(-5+a*deltaX)
        u[1,a]=u[0,a]
    """


    for t in range(1, N-1):

        if (t<10):
            u[t+1, 0] = u[t, 1]
            u[t+1, Ngrid-1] = u[t, Ngrid-1]
        else:
            u[t, 0] = 0#u[t, 1]
            u[t, Ngrid-1] = 0#u[t, Ngrid-1]
            #u[t+1, 0] = (u[t, 0] - 2*u[t, 1] + u[t,2])/epsilon + 2*u[t, 0] - u[t-1, 0]
            #u[t+1, Ngrid-1] = (u[t, Ngrid-1] - 2*u[t, Ngrid-2] + u[t,Ngrid-3])/epsilon + 2*u[t, Ngrid-1] - u[t-1, Ngrid-1]


        for x in range(1, Ngrid-1):
            u[t+1, x] = (u[t, x+1] + u[t, x-1] - 2*u[t, x])/epsilon - u[t-1, x] + 2*u[t, x]

    """
    for t in range(1,N-1):
        for a in range(1,Ngrid-1):


            u[t+1,a] = 2*(1-1.0/epsilon)*u[t,a]-u[t-1,a]+1.0/epsilon*(u[t,a-1]+u[t,a+1])

    """

    return np.repeat(u, Ngrid, axis=0).reshape((N, Ngrid, Ngrid))

N = 150
margin = 0
width = 1
n_units = 200
ndata = 10000
trainLength = 8000

data = generate_data2(ndata, N)

print(data.shape)

show_results([("data", data)])
exit()

training_data = data[:ndata-trainLength]
test_data = data[ndata-trainLength:]

print(training_data.shape)

#input_y, input_x, output_y, output_x = create_patch_indices((12,17), (12,17), (13,16), (13,16)) # -> yields MSE=0.0115 with leak_rate = 0.8
input_y, input_x, output_y, output_x = create_patch_indices((margin, N-margin), (margin, N-margin), (margin+width, N-margin-width), (margin+width, N-margin-width)) # -> yields MSE=0.0873 with leak_rate = 0.3

training_data_in =  training_data[:, input_y, input_x].reshape(-1, len(input_y))
training_data_out =  training_data[:, output_y, output_x].reshape(-1, len(output_y))

test_data_in =  test_data[:, input_y, input_x].reshape(-1, len(input_y))
test_data_out =  test_data[:, output_y, output_x].reshape(-1, len(output_y))


print("setting up...")
esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = n_units, #used to be 1700
        weight_generation = "advanced", leak_rate = 0.2, spectral_radius = 0.1,
        random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-0], solver = "lsqr",
        )

print("fitting...")

train_error = esn.fit(training_data_in, training_data_out, verbose=1)
print("train error: {0}".format(train_error))



print("predicting...")
pred = esn.predict(test_data_in, verbose=1)
#pred[pred>1.0] = 1.0
#pred[pred<0.0] = 0.0

merged_prediction = test_data.copy()
merged_prediction[:, output_y, output_x] = pred*0.1

diff = pred.reshape((-1, len(output_y))) - test_data_out
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

difference = np.abs(test_data - merged_prediction)

show_results([("pred", merged_prediction), ("orig", test_data), ("diff", difference)], forced_clim=[-1,1])

print("done.")
