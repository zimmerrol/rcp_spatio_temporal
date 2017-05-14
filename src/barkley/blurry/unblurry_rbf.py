import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,grandparentdir)
# -*- coding: utf-8 -*-

#TODO: Use http://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
from RBF import *
import progressbar
import dill as pickle #we require version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import copy
from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#from pathos.multiprocessing import Pool
import multiprocessing
import ctypes

from scipy.ndimage.filters import gaussian_filter

from helper import *
from barkley_helper import *

N = 150
ndata = 10000
testLength = 2000
trainLength = ndata - testLength
sigma = 31
sigma_skip = 2
eff_sigma = int(np.ceil(sigma/sigma_skip))
patch_radius = sigma // 2
n_units = 100

def setupArrays():
    #TODO: Correct the array dimensions!
    global shared_training_data_base, shared_test_data_base, prediction_base
    global shared_training_data, shared_test_data, prediction

    shared_training_data_base = multiprocessing.Array(ctypes.c_double, 2*trainLength*N*N)
    shared_training_data = np.ctypeslib.as_array(shared_training_data_base.get_obj())
    shared_training_data = shared_training_data.reshape(2, trainLength, N, N)

    shared_test_data_base = multiprocessing.Array(ctypes.c_double, 2*testLength*N*N)
    shared_test_data = np.ctypeslib.as_array(shared_test_data_base.get_obj())
    shared_test_data = shared_test_data.reshape(2, testLength, N, N)

    prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    prediction = np.ctypeslib.as_array(prediction_base.get_obj())
    prediction = prediction.reshape(testLength, N, N)

setupArrays()

def fit_predict_pixel(y, x, running_index, training_data, test_data, generate_new):
    training_data_in = training_data[1][:, y - patch_radius:y + patch_radius+1, x - patch_radius:x + patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape(-1, eff_sigma*eff_sigma)
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, y - patch_radius:y + patch_radius+1, x - patch_radius:x + patch_radius+1][:, ::sigma_skip, ::sigma_skip].reshape(-1, eff_sigma*eff_sigma)
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    rbf = RBF(sigma=5.0)
    rbf.fit(training_data_in, training_data_out, basisQuota=0.02)
    pred = rbf.predict(test_data_in)
    pred = pred.ravel()

    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    return pred

def fit_predict_frame_pixel(y, x, running_index, training_data, test_data, generate_new):
    ind_y, ind_x = y, x

    min_border_distance = np.min([y, x, N-1-y, N-1-x])

    training_data_in = training_data[1][:, y - min_border_distance:y + min_border_distance+1, x - min_border_distance:x + min_border_distance+1].reshape(-1, int((2*min_border_distance+1)**2))
    training_data_out = training_data[0][:, y, x].reshape(-1, 1)

    test_data_in = test_data[1][:, y - min_border_distance:y + min_border_distance+1, x - min_border_distance:x + min_border_distance+1].reshape(-1, int((2*min_border_distance+1)**2))
    test_data_out = test_data[0][:, y, x].reshape(-1, 1)

    rbf = RBF(sigma=5.0)
    rbf.fit(training_data_in, training_data_out, basisQuota=0.02)
    pred = rbf.predict(test_data_in)
    pred = pred.ravel()

    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    return pred

def get_prediction_init(q):
    get_prediction.q = q

def get_prediction(data, def_param=(shared_training_data, shared_test_data)):
    y, x, running_index = data

    #print(np.max(shared_test_data[1]))


    pred = None
    if (y >= patch_radius and y < N-patch_radius and x >= patch_radius and x < N-patch_radius):
        #inner point
        pred = fit_predict_pixel(y, x, running_index, shared_training_data, shared_test_data, True)

    else:
        #frame
        pred = fit_predict_frame_pixel(y, x, running_index, shared_training_data, shared_test_data, True)

    get_prediction.q.put((y, x, pred))

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults):
    global prediction

    bar = progressbar.ProgressBar(max_value=numberOfResults, redirect_stdout=True, poll_interval=0.0001)
    bar.update(0)

    finishedWorkers = 0
    finishedResults = 0

    while True:
        if (finishedResults == numberOfResults):
            return

        newData= q.get()
        finishedResults += 1
        ind_y, ind_x, data = newData

        prediction[:, ind_y, ind_x] = data

        bar.update(finishedResults)


def mainFunction():
    global shared_training_data, shared_test_data

    if (os.path.exists("../cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
        print("generating data...")
        data = generate_data(ndata, 20000, 5, Ngrid=N) #20000 was 50000
        np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(ndata, N), data)
        print("generating finished")
    else:
        print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.dat.npy".format(ndata, N))

        print("loading finished")

    training_data = data[:trainLength]
    test_data = data[trainLength:]

    shared_training_data[0, :, :, :] = training_data[:]
    shared_test_data[0, :, :, :] = test_data[:]

    print("blurring...")

    for t in range(trainLength):
        shared_training_data[1, t, :, :] = gaussian_filter(training_data[t], sigma=9.0)

    for t in range(testLength):
        shared_test_data[1, t, :, :] = gaussian_filter(test_data[t], sigma=9.0)

    show_results([("Orig", shared_training_data[0]), ("Blurred", shared_training_data[1])])

    print("blurring finished")

    queue = Queue() # use manager.queue() ?
    print("preparing threads...")
    pool = Pool(processes=26, initializer=get_prediction_init, initargs=[queue,])

    modifyDataProcessList = []
    jobs = []
    index = 0

    for y in range(N):
        for x in range(N):
            jobs.append((y, x, index))

    print("fitting...")
    processProcessResultsThread = Process(target=processThreadResults, args=("processProcessResultsThread", queue, 26, len(jobs)) )
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    print("finished fitting")

    diff = (test_data-prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean((diff[:, patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    show_results([("Source", shared_test_data[1]), ("Orig", shared_test_data[0]), ("Pred", prediction), ("Diff", diff)])

if __name__== '__main__':
    mainFunction()
