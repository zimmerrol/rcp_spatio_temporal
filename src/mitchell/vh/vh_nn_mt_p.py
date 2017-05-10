import os

id = int(os.getenv("SGE_TASK_ID", 0))
first = int(os.getenv("SGE_TASK_FIRST", 0))
last = int(os.getenv("SGE_TASK_LAST", 0))
print("ID {0}".format(id))
print("Task %d of %d tasks, starting with %d." % (id, last - first + 1, first))

print("This job was submitted from %s, it is currently running on %s" % (os.getenv("SGE_O_HOST"), os.getenv("HOSTNAME")))

print("NHOSTS: %s, NSLOTS: %s" % (os.getenv("NHOSTS"), os.getenv("NSLOTS")))

import sys
print(sys.version)





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
import progressbar
import dill as pickle
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors as NN
from helper import *

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#from pathos.multiprocessing import Pool
import multiprocessing
import ctypes

from multiprocessing import process

process.current_process()._config['tempdir'] =  '/dev/shm/' #'/data.bmp/roland/temp/'

N = 150
ndata = 30000
sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]
eff_sigma = int(np.ceil(sigma/sigma_skip))
ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5][id-1]
patch_radius = sigma//2
trainLength = 28000
testLength = 2000

print("Using parameters:")
print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, sigma, sigma_skip, ddim, k))

def setupArrays():
    global shared_h_data_base, shared_v_data_base, shared_prediction_base
    global shared_h_data, shared_v_data, shared_prediction

    ###print("setting up arrays...")
    shared_h_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_h_data = np.ctypeslib.as_array(shared_h_data_base.get_obj())
    shared_h_data = shared_h_data.reshape(-1, N, N)
   
    shared_v_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_v_data = np.ctypeslib.as_array(shared_v_data_base.get_obj())
    shared_v_data = shared_v_data.reshape(-1, N, N)
    
    shared_prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    shared_prediction = np.ctypeslib.as_array(shared_prediction_base.get_obj())
    shared_prediction = shared_prediction.reshape(-1, N, N)
    ###print("setting up finished") 

setupArrays()

def create_2d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, :, :, np.newaxis], repeats=delay_dimension, axis=3)

    for n in range(1, delay_dimension):
        result[:, :, :, n] = np.roll(result[:, :, :, n], n*tau, axis=0)
    result[0:delay_dimension-1,:,:] = 0

    return result
    
def create_0d_delay_coordinates(data, delay_dimension, tau):
    result = np.repeat(data[:, np.newaxis], repeats=delay_dimension, axis=1)
    
    for n in range(1, delay_dimension):
        result[:, n] = np.roll(result[:, n], n*tau, axis=0)
    result[0:delay_dimension-1,:] = 0

    return result

def generate_data(N, trans, sample_rate, Ngrid):
    data = None
    if (os.path.exists("../cache/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)) == False):
        ###print("generating data...")
        data = generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
        np.save("../cache/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        ###print("generating finished")
    else:
        ###print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))
        ###print("loading finished")
        
        
        #at the moment we are doing a h -> v cross prediction.
        #switch the entries for the v -> h prediction
        tmp = data[0].copy()
        data[0] = data[1].copy()
        data[1] = tmp.copy()
        
        
    return data

def get_prediction(data):
    y, x = data
    
    pred = None
    if (y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius):
        pred = predict_frame_pixel(data)
    else:
        pred = predict_inner_pixel(data)
    get_prediction.q.put((y, x, pred))
    
def predict_frame_pixel(data, def_param=(shared_h_data, shared_v_data)):
    y, x = data
   
    shared_delayed_h_data = create_0d_delay_coordinates(shared_h_data[:, y, x], ddim, tau=32)
   
    delayed_patched_h_data_train = shared_delayed_h_data[:trainLength]
    v_data_train = shared_v_data[:trainLength, y, x]

    delayed_patched_h_data_test = shared_delayed_h_data[trainLength:trainLength+testLength]
    v_data_test = shared_v_data[trainLength:trainLength+testLength, y, x]

    flat_h_data_train = delayed_patched_h_data_train.reshape(-1, ddim)
    flat_v_data_train = v_data_train.reshape(-1,1)

    flat_h_data_test = delayed_patched_h_data_test.reshape(-1, ddim)
    flat_v_data_test = v_data_test.reshape(-1,1)

    neigh = NN(2, n_jobs=1)#n_jobs=26

    neigh.fit(flat_h_data_train)

    distances, indices = neigh.kneighbors(flat_h_data_test)

    pred = ((flat_v_data_train[indices[:, 0]] + flat_v_data_train[indices[:, 1]])/2.0).ravel()
    
    return pred    
    
def predict_inner_pixel(data, def_param=(shared_h_data, shared_v_data)):
    y, x = data
    
    shared_delayed_h_data = create_2d_delay_coordinates(shared_h_data[:, y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1][:, ::sigma_skip, ::sigma_skip], ddim, tau=119)
    shared_delayed_patched_h_data = np.empty((ndata, 1, 1, ddim*eff_sigma*eff_sigma))
    shared_delayed_patched_h_data[:, 0, 0] = shared_delayed_h_data.reshape(-1, ddim*eff_sigma*eff_sigma)
    
    delayed_patched_h_data_train = shared_delayed_patched_h_data[:trainLength, 0, 0]
    v_data_train = shared_v_data[:trainLength, y, x]

    delayed_patched_h_data_test = shared_delayed_patched_h_data[trainLength:trainLength+testLength, 0, 0]
    v_data_test = shared_v_data[trainLength:trainLength+testLength, y, x]

    flat_h_data_train = delayed_patched_h_data_train.reshape(-1, shared_delayed_patched_h_data.shape[3])
    flat_v_data_train = v_data_train.reshape(-1,1)

    flat_h_data_test = delayed_patched_h_data_test.reshape(-1, shared_delayed_patched_h_data.shape[3])
    flat_v_data_test = v_data_test.reshape(-1,1)

    neigh = NN(k, n_jobs=1) #n_jobs=26

    neigh.fit(flat_h_data_train)

    distances, indices = neigh.kneighbors(flat_h_data_test)
    
    with np.errstate(divide='ignore'):
        weights = np.divide(1.0, distances)

    infinity_mask = np.isinf(weights)
    infinity_row_mask = np.any(infinity_mask, axis=1)
    weights[infinity_row_mask] = infinity_mask[infinity_row_mask]

    denominator = np.repeat(np.sum(weights, axis=1),k).reshape((-1,k))
    weights /= denominator

    pred = 0
    for i in range(k):
        pred += np.multiply(weights[:, i, np.newaxis], flat_v_data_train[indices[:, i]])
    pred = pred.ravel()

    #pred = ((flat_v_data_train[indices[:, 0]] + flat_v_data_train[indices[:, 1]])/2.0).ravel()
    
    return pred
   

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults, def_param=(shared_prediction, shared_v_data)):
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

        shared_prediction[:, ind_y, ind_x] = data

        bar.update(finishedResults)

def get_prediction_init(q):
    get_prediction.q = q

def mainFunction():
    if (trainLength +testLength > ndata):
        print("Please adjust the trainig and testing phase length!")
        exit()
    
    delayed_patched_h_data = None
    v_data = None
    ###print("generating data...")
    v_data_t, h_data_t = generate_data(ndata, 20000, 50, Ngrid=N)#20000 was 50000 #, delay_dimension=ddim, patch_size=sigma)
    shared_h_data[:] = h_data_t[:]
    shared_v_data[:] = v_data_t[:]
    ###print("generation finished")

    queue = Queue() # use manager.queue() ?
    ###print("preparing threads...")
    pool = Pool(processes=32, initializer=get_prediction_init, initargs=[queue,])

    processProcessResultsThread = Process(target=processThreadResults, args=("processProcessResultsThread", queue, 32, N*N) )

    modifyDataProcessList = []
    jobs = []
    for y in range(N):
        for x in range(N):
                jobs.append((y, x))

    ###print("fitting...")
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    ###print("finished fitting")
    
    diff = (shared_v_data[trainLength:]-shared_prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean(((shared_v_data[trainLength:]-shared_prediction)[patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    viewData = [("Orig", shared_v_data[trainLength:]), ("Pred", shared_prediction), ("Source", shared_h_data[trainLength:]), ("Diff", diff)]
    f = open("../cache/viewdata/vtoh/nn_viewdata_{0}_{1}_{2}_{3}.dat".format(ndata, sigma, sigma_skip, ddim), "wb")
    pickle.dump(viewData, f)
    f.close()

    #show_results({"Orig" : shared_v_data[trainLength:], "Pred" : shared_prediction, "Diff" : diff})
    
    print("done")


if __name__== '__main__':
    mainFunction()

