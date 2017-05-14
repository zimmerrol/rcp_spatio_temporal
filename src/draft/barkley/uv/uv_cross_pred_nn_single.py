#get V animation data -> [N, 150, 150]
#create 2d delay coordinates -> [N, 150, 150, d]
#create new dataset with small data groups -> [N, 150, 150, d*sigma*sigma]
#create d*sigma*sigma-k tree from this data
#search nearest neighbours (1 or 2) and predict new U value

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
import progressbar
import dill as pickle
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors as NN
from helper import *

def create_2d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, :, :, np.newaxis], repeats=delay_dimension, axis=3)
    print("delayed data copied")

    for n in range(1, delay_dimension):
        result[:, :, :, n] = np.roll(result[:, :, :, n], n*tau, axis=0)
    result[0:delay_dimension-1,:,:] = 0

    return result

def create_2d_delayed_patches(data, delay_dimension, tau, patch_size):
        print("creating delay coordinates")
        delayed_data = create_2d_delay_coordinates(data=data, delay_dimension=delay_dimension, tau=tau)

        patch_radius = patch_size//2
        N = data.shape[1]

        print("setting up delayed patches result array")


        result_data = np.empty((len(data), N, N, delay_dimension*patch_size*patch_size))

        bar = progressbar.ProgressBar(max_value=(N-patch_radius*2)**2, redirect_stdout=True, poll_interval=0.0001)
        bar.update(0)

        for y in range(patch_radius, N-patch_radius):
            for x in range(patch_radius, N-patch_radius):
                result_data[:, y, x] = delayed_data[:, y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1, :].reshape(-1, delay_dimension*patch_size*patch_size)
                bar.update((x-patch_radius) + (y-patch_radius)*(N-patch_radius*2))

        bar.finish()

        return result_data

def generate_delayed_data(N, trans, sample_rate, Ngrid, delay_dimension, patch_size):
    data = None
    if (os.path.exists("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)) == False):
        print("generating data...")
        data = generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
        np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        print("generating finished")
    else:
        print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
        print("loading finished")

    delayed_patched_data = create_2d_delayed_patches(data[1], delay_dimension=delay_dimension, tau=32, patch_size=patch_size)

    patch_radius = patch_size // 2

    return delayed_patched_data, data[0][:, patch_radius:Ngrid-patch_radius, patch_radius:Ngrid-patch_radius].reshape(-1, Ngrid-2*patch_radius, Ngrid-2*patch_radius)

N = 150
ndata = 10000
sigma = 1
ddim = 3

force_generation = True

delayed_patched_v_data = None
u_data = None
if (os.path.exists("../cache/raw/{0}_{1}_{2}.uv.nn.dat.npy".format(ndata, N, sigma)) == False or force_generation):
    print("generating delayed data...")
    delayed_patched_v_data, u_data = generate_delayed_data(ndata, 50000, 5, Ngrid=N, delay_dimension=ddim, patch_size=sigma)
    #np.save("../cache/raw/{0}_{1}_{2}.uv.nn.dat.npy".format(ndata, N, sigma), delayed_patched_v_data)
    print("generating finished")
else:
    print("loading delayed data...")
    delayed_patched_v_data = np.load("../cache/raw/{0}_{1}_{2}.uv.nn.dat.npy".format(ndata, N, sigma))
    print("loading finished")

print("reshaping...")

"""
#Training   MSE
1000        0.00916040941011
4000        0.00686822737285
8000        0.00631390262526
"""

trainLength = 9000
testLength = 1000

if (trainLength +testLength > ndata):
    print("Please adjust the trainig and testing phase length!")
    exit()

#x = 50
#y = 50

for y in range(150):
    print(y)
    for x in range(150):
        delayed_patched_v_data_train = delayed_patched_v_data[:trainLength, y, x]
        u_data_train = u_data[:trainLength, y, x]

        delayed_patched_v_data_test = delayed_patched_v_data[trainLength:trainLength+testLength, y, x]
        u_data_test = u_data[trainLength:trainLength+testLength, y, x]

        flat_v_data_train = delayed_patched_v_data_train.reshape(-1, delayed_patched_v_data.shape[3])
        flat_u_data_train = u_data_train.reshape(-1,1)

        flat_v_data_test = delayed_patched_v_data_test.reshape(-1, delayed_patched_v_data.shape[3])
        flat_u_data_test = u_data_test.reshape(-1,1)

        neigh = NN(2, n_jobs=26)
        #print("fitting")

        neigh.fit(flat_v_data_train)

        #print("predicting...")
        distances, indices = neigh.kneighbors(flat_v_data_test)
        #print(distances)

        flat_u_prediction = (flat_u_data_train[indices[:, 0]] + flat_u_data_train[indices[:, 1]])/2.0

        diff = flat_u_prediction - flat_u_data_test
        #print("{0},{1}: {2}".format(y, x, np.mean(diff**2)))

prediction = flat_u_prediction.reshape(testLength, 1, 1)

show_results({"prediction": prediction})

"""
print("settting up the tree...")
tree = KDTree(flat_v_data_train)
print("tree set up")

print("predicting...")
distances, indices = tree.query(flat_v_data_test)
print(distances)

flat_u_prediction = flat_u_data_train[indices]

diff = flat_u_prediction - flat_u_data_test
print(np.mean(diff**2))
"""
print("done")
