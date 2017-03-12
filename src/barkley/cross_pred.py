import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN

def generate_data(N, trans, sample_rate=1):
    Nx = 30
    Ny = 30
    deltaT = 1e-2
    epsilon = 0.08
    h = 1.0#0.2
    a = 0.75
    b = 0.00006

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    for i in range(trans):
        sim.explicit_step(chaotic=True)

    data = np.empty((N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[i] = sim._u

    return data

print("generating data...")

data = generate_data(5000, 50000, 5)

"""
i = 0
def update_new(d):
    global i
    mat.set_data(data[i,:].reshape((30, 30)))
    plt.title(i)

    i = (i+1) % 500
    return [mat]

fig, ax = plt.subplots()

mat = ax.matshow(data[0,:].reshape((30, 30)), vmin=0, vmax=1, interpolation=None, origin="lower")
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update_new, interval=0, save_count=50)
plt.show()
"""

#data = data.reshape((1000, -1))

training_data = data[:4000]
test_data = data[4000:]

out_ind = [8,9,10]#[14,15,16]
in_ind = list(range(14))
in_ind.extend(list(range(17,30)))

training_data_in =  training_data[:, in_ind][:,:, in_ind].reshape(-1, (30-3)**2)
training_data_out =  training_data[:, out_ind][:,:, out_ind].reshape(-1, 3**2)

test_data_in =  test_data[:, in_ind][:,:, in_ind].reshape(-1, (30-3)**2)
test_data_out =  test_data[:, out_ind][:,:, out_ind].reshape(-1, 3**2)

print("setting up...")

"""
print("starting grid search...")
from GridSearch import GridSearch
grid = GridSearch(param_grid={"n_reservoir": [700, 1000], "spectral_radius": [1.5, 1.6, 1.7, 1.8, 1.9], "leak_rate": [.8, .95, .98, .99], "sparseness": [0.05, 0.1, 0.2]},
    fixed_params={"n_output": 3**2, "n_input": (30-3)**2, "noise_level": 0.001, "random_seed": 42, "weight_generation": "advanced"},
    esnType=ESN)
print("start fitting...")
results = grid.fit(training_data_in, training_data_out, [(test_data_in, test_data_out)])
print("done:\r\n")
print(results)
print("best parameters: " + str(grid._best_params))
print("best mse: {0}".format(grid._best_mse))

best parameters: {'sparseness': 0.1, 'leak_rate': 0.95, 'spectral_radius': 1.5, 'n_reservoir': 500}
best mse: 0.028543328874922003

import sys
sys.exit()
"""

esn = ESN(n_input = (30-3)**2, n_output = 3**2, n_reservoir = 700,
        weight_generation = "advanced", leak_rate = 0.99, spectral_radius = 1.9,
        random_seed=42, noise_level=0.000, sparseness=.1, solver = "lsqr", out_activation = lambda x: 0.5*(1+np.tanh(x/2)), out_inverse_activation = lambda x:2*np.arctanh(2*x-1))

print("fitting...")
train_error = esn.fit(training_data_in, training_data_out, regression_parameters=[2e-2])
print("train error: {0}".format(train_error))
print("predicting...")
pred = esn.predict(test_data_in)
pred = pred.reshape(-1, 3, 3)

merged_prediction = test_data.copy()
merged_prediction[np.ix_(list(range(len(test_data))), out_ind, out_ind)] = pred

diff = pred.reshape((-1, 9))-test_data_out
mse = np.mean(diff.reshape(-1, 3*3)**2)
print(mse)

diff = np.abs(diff)


#import sys
#sys.exit()

i = 0
def update_new(data):
    global i
    plt.title(i)
    #mat.set_data(diff[i])
    mat.set_data(merged_prediction[i])
    i = (i+1) % len(diff)
    return [mat]


fig, ax = plt.subplots()
#mat = plt.imshow(diff[0], origin="lower", vmin=np.min(diff), vmax=np.max(diff), interpolation="none", cmap=plt.get_cmap('gray'))
mat = plt.imshow(merged_prediction[0], origin="lower", vmin=0, vmax=1, interpolation="none")
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update_new, interval=100, save_count=50)
plt.show()

print("done.")
