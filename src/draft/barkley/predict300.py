import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)
sys.path.insert(0, grandgrandparentdir)
sys.path.insert(0, os.path.join(grandgrandparentdir, "barkley"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helper import *
import barkley_helper as bh

from ESN import ESN


N = 10000
trainLength = 8000
testLength = 2000

Ngrid = 150

if (os.path.exists("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid)) == False):
    data = bh.generate_data(N=N, trans=50000, sample_rate=5, Ngrid=Ngrid)
    np.save("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid), data)
else:
    data = np.load("../../cache/barkley/raw/{0}_{1}.dat.npy".format(N, Ngrid))


T = 100

#input_y, input_x, output_y, output_x = create_patch_indices((12,17), (12,17), (13,16), (13,16)) # -> yields MSE=0.0115 with leak_rate = 0.8
#input_y, input_x, output_y, output_x = create_patch_indices((4,23), (4,23), (7,20), (7,20)) # -> yields MSE=0.0873 with leak_rate = 0.3
#index_y, index_x =  create_square((7,9),(7,9))

pointX = 75
pointY = 75
index_y, index_x = [pointY],[pointX]#create_rectangle_indices([74,77],[74,77])
#print(index_x.shape)

yData = data[T:, pointY, pointX]
xData = data[:-T, index_y, index_x]

yData = yData.reshape((-1, 1))
xData = xData.reshape((-1, len(index_y)))

"""
from GridSearch import GridSearch
cv = GridSearch(
    param_grid={
        "n_reservoir": [200, 300, 500, 800], "spectral_radius": [0.3, 0.6, .8, .9, .95, 1.1, 1.2, 1.3, 1.4], "leak_rate": [.2, .6, .8, .9, .95, .99, 1.0],
        "random_seed": [40,41,42,43,44], "sparseness": [.05, .1, .2], "weight_generation": ["advanced"],
        "solver": ["lsqr"], "regression_parameters": [[3e-3], [3e-4], [3e-5], [3e-6], [3e-7], [3e-8]]
    },
        fixed_params={"n_output": 1, "n_input": len(index_y), "noise_level": 0.0001#, "input_scaling":[0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]
    },
    esnType=ESN)
print("start fitting...")

def cutval(val):
    val[val > 1.0] = 1.0
    val[val < 0.0] = 0.0

    return val

results = cv.fit(xData[:trainLength], yData[:trainLength], [(xData[trainLength:trainLength+testLength], yData[trainLength:trainLength+testLength])], cutval, printfreq=100)
print(results)
print("-.-")
print(cv._best_params)
print(cv._best_mse)

exit()
"""



#T=50: (0.033947233385081932, 0.071219920423523347, {'sparseness': 0.05, 'solver': 'lsqr', 'spectral_radius': 0.3, 'random_seed': 42, 'regression_parameters': [0.0003], 'n_reservoir': 500, 'leak_rate': 0.2, 'weight_generation': 'naive'})
#T=10: (0.00078495804606547662, {'leak_rate': 0.9, 'n_reservoir': 500, 'spectral_radius': 0.95, 'solver': 'pinv', 'weight_generation': 'naive', 'random_seed': 44, 'sparseness': 0.05})
#T=100:  (0.050939652813549598, 0.13093701743229968, {'regression_parameters': [0.003], 'sparseness': 0.05, 'spectral_radius': 0.3, 'n_reservoir': 500, 'leak_rate': 0.2, 'solver': 'lsqr', 'weight_generation': 'naive', 'random_seed': 41})
#T=100, solo:  (0.032009262884647588, 0.11699984785782426, {'random_seed': 42, 'sparseness': 0.1, 'n_reservoir': 800, 'solver': 'lsqr', 'weight_generation': 'naive', 'regression_parameters': [0.0003], 'spectral_radius': 1.1, 'leak_rate': 0.6})

###print("setting up...")

#plt.plot(yData)
#plt.show()

"""
#T=10
esn = ESN(n_input = 1, n_output = len(index_y), n_reservoir = 400,
        weight_generation = "advanced", leak_rate = 1.0, spectral_radius = 1.05,
        random_seed=42, noise_level=0.0001, sparseness=.1, solver = "lsqr", regression_parameters=[1e-9])
"""

"""
#T=20
esn = ESN(n_input = 1, n_output = len(index_y), n_reservoir = 500,
        weight_generation = "advanced", leak_rate = 0.75, spectral_radius = 1.15,
        random_seed=41, noise_level=0.0001, sparseness=.1, solver = "lsqr", regression_parameters=[1e-6])
"""

#T=100
#(0.09311725748279541, 0.29777506387407898, {'random_seed': 41, 'n_reservoir': 500, 'solver': 'lsqr', 'sparseness': 0.05, 'regression_parameters': [3e-08], 'leak_rate': 0.95, 'spectral_radius': 1.2, 'weight_generation': 'advanced'})
esn = ESN(n_input = len(index_y), n_output = 1, n_reservoir = 500,
        weight_generation = "advanced", leak_rate = 0.95, spectral_radius = 1.2,
        random_seed=41, noise_level=0.0001, sparseness=.05, solver = "lsqr", regression_parameters=[3e-8])

"""
    T=10:
        weight_generation = "naive", leak_rate = 0.9, spectral_radius = 1.18,
        random_seed=42, noise_level=0.0001, sparseness=.1, solver = "lsqr", regression_parameters=[1e-8])

    T=20:
        weight_generation = "naive", leak_rate = 0.75, spectral_radius = 0.80,
        random_seed=42, noise_level=0.0001, sparseness=.1, solver = "lsqr", regression_parameters=[1e-8])
"""

###print("fitting...")
train_error = esn.fit(xData[:trainLength], yData[:trainLength], transient_quota=0.2)
print("train error: {0}".format(train_error))


###print("predicting...")
pred = esn.predict(xData[trainLength:testLength+trainLength])
pred[pred > 1] = 1
pred[pred < 0] = 0

plt.plot(yData[trainLength:trainLength+testLength], linestyle="-", label="target")
plt.plot(pred, label="prediction")
plt.ylim([-0.2,1.2])
plt.legend()

diff = pred - yData[trainLength:trainLength+testLength]
mse = np.mean((diff)**2)
print("test error: {0}".format(mse))

plt.show()
