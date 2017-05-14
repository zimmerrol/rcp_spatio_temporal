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
from barkley_helper import *
from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing
import ctypes
from multiprocessing import process
import argparse


#get V animation data -> [N, 150, 150]
#create 2d delay coordinates -> [N, 150, 150, d]
#create new dataset with small data groups -> [N, 150, 150, d*sigma*sigma]
#create d*sigma*sigma-k tree from this data
#search nearest neighbours (1 or 2) and predict new U value

process.current_process()._config['tempdir'] =  '/dev/shm/' #'/data.bmp/roland/temp/'

N = 150
ndata = 30000
trainLength = 28000
testLength = 2000

def parseArguments():
    id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="vu: v -> u, otherwise: u -> v")
    args = parser.parse_args()

    reverseDirection = args.direction[0] != "vu"
    if args.mode not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))
    else:
        predictionMode = args.mode

    print("Prediction: {0}".format("u -> v" if reverseDirection else "v -> u"))
parseArguments()

def setupArrays():
    global shared_v_data_base, shared_u_data_base, shared_prediction_base
    global shared_v_data, shared_u_data, shared_prediction

    ###print("setting up arrays...")
    shared_v_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_v_data = np.ctypeslib.as_array(shared_v_data_base.get_obj())
    shared_v_data = shared_v_data.reshape(-1, N, N)

    shared_u_data_base = multiprocessing.Array(ctypes.c_double, ndata*N*N)
    shared_u_data = np.ctypeslib.as_array(shared_u_data_base.get_obj())
    shared_u_data = shared_u_data.reshape(-1, N, N)

    shared_prediction_base = multiprocessing.Array(ctypes.c_double, testLength*N*N)
    shared_prediction = np.ctypeslib.as_array(shared_prediction_base.get_obj())
    shared_prediction = shared_prediction.reshape(-1, N, N)
    ###print("setting up finished")
setupArrays()

def setupConstants():
    global k, ddim, sigma, sigma_skip, eff_sigma, patch_radius, n_units, regression_parameter

    print("Using parameters:")

    if (predictionMode == "ESN"):
        n_units = [50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400,  50,50,50,50,50,50,200,200,200,200,200,200,400,400,400,400,400,400][id-1]
        regression_parameter = [5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,  5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,5e-3,  5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,5e-4,  5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5,5e-5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]

        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(ndata, sigma, sigma_skip, n_units, regression_parameter))
    elif (predictionMode == "NN"):
        ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
        k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]

        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, sigma, sigma_skip, ddim, k))
    elif (predictionMode == "RBF"):
        ddim = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,  3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5][id-1]
        k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5][id-1]
        sigma = [3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7,  3,5,7,5,7,7,3,5,7,5,7,7,3,5,7,5,7,7][id-1]
        sigma_skip = [1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3,  1,1,1,2,2,3,1,1,1,2,2,3,1,1,1,2,2,3][id-1]
        print("\t ndata \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ndata, sigma, sigma_skip, ddim, k))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    eff_sigma = int(np.ceil(sigma/sigma_skip))
    patch_radius = sigma//2
setupConstants()

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
    if (os.path.exists("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)) == False):
        ###print("generating data...")
        data = generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
        np.save("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        ###print("generating finished")
    else:
        ###print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
        ###print("loading finished")

        #at the moment we are doing a v -> u cross prediction.
        if (reverseDirection):
            #switch the entries for the u -> v prediction
            tmp = data[0].copy()
            data[0] = data[1].copy()
            data[1] = tmp.copy()

    return data

def preparePredicter(y, x):
    if (predictionMode == "ESN"):
        if (y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius):
            #frame
            predicter = ESN(n_input = 1, n_output = 1, n_reservoir = n_units,
                    weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                    random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[regression_parameter], solver = "lsqr")
        else:
            #inner
            predicter = ESN(n_input = eff_sigma*eff_sigma, n_output = 1, n_reservoir = n_units,
                        weight_generation = "advanced", leak_rate = 0.70, spectral_radius = 0.8,
                        random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[regression_parameter], solver = "lsqr")

    elif (predictionMode == "NN"):
        predicter = NN(k=k)
    elif (predictionMode == "RBF"):
        predicter = RBF(sigma=5.0, basisQuota=0.05)
    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(predictionMode))

    return predicter

def get_prediction(data):
    y, x = data

    predicter = preparePredicter(y, x)
    pred = None
    if (y < patch_radius or y >= N-patch_radius or x < patch_radius or x >= N-patch_radius):
        #frame
        pred = predict_frame_pixel(data, predicter)
    else:
        #inner
        pred = predict_inner_pixel(data, predicter)
    get_prediction.q.put((y, x, pred))

def predict_frame_pixel(data, predicter, def_param=(shared_v_data, shared_u_data)):
    y, x = data

    delayed_v_data = create_0d_delay_coordinates(shared_v_data[:, y, x], ddim, tau=32)

    delayed_patched_v_data_train = delayed_v_data[:trainLength]
    u_data_train = shared_u_data[:trainLength, y, x]

    delayed_patched_v_data_test = delayed_v_data[trainLength:trainLength+testLength]
    u_data_test = shared_u_data[trainLength:trainLength+testLength, y, x]

    flat_v_data_train = delayed_patched_v_data_train.reshape(-1, ddim)
    flat_u_data_train = u_data_train.reshape(-1,1)

    flat_v_data_test = delayed_patched_v_data_test.reshape(-1, ddim)
    flat_u_data_test = u_data_test.reshape(-1,1)

    predicter.fit(flat_v_data_train, flat_u_data_train)
    pred = predicter.predict(flat_v_data_test)
    pred = pred.ravel()

    return pred

def predict_inner_pixel(data, predicter, def_param=(shared_v_data, shared_u_data)):
    y, x = data

    delayed_v_data = create_2d_delay_coordinates(shared_v_data[:, y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1][:, ::sigma_skip, ::sigma_skip], ddim, tau=119)
    delayed_patched_v_data = np.empty((ndata, 1, 1, ddim*eff_sigma*eff_sigma))
    delayed_patched_v_data[:, 0, 0] = delayed_v_data.reshape(-1, ddim*eff_sigma*eff_sigma)

    delayed_patched_v_data_train = delayed_patched_v_data[:trainLength, 0, 0]
    u_data_train = shared_u_data[:trainLength, y, x]

    delayed_patched_v_data_test = delayed_patched_v_data[trainLength:trainLength+testLength, 0, 0]
    u_data_test = shared_u_data[trainLength:trainLength+testLength, y, x]

    flat_v_data_train = delayed_patched_v_data_train.reshape(-1, delayed_patched_v_data.shape[3])
    flat_u_data_train = u_data_train.reshape(-1,1)

    flat_v_data_test = delayed_patched_v_data_test.reshape(-1, delayed_patched_v_data.shape[3])
    flat_u_data_test = u_data_test.reshape(-1,1)

    predicter.fit(flat_v_data_train, flat_u_data_train)
    pred = predicter.predict(flat_v_data_test)
    pred = pred.ravel()

    return pred

def processThreadResults(threadname, q, numberOfWorkers, numberOfResults, def_param=(shared_prediction, shared_u_data)):
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

    delayed_patched_v_data = None
    u_data = None
    ###print("generating data...")
    u_data_t, v_data_t = generate_data(ndata, 20000, 50, Ngrid=N)
    shared_v_data[:] = v_data_t[:]
    shared_u_data[:] = u_data_t[:]
    ###print("generation finished")

    queue = Queue() # use manager.queue() ?
    ###print("preparing threads...")
    pool = Pool(processes=16, initializer=get_prediction_init, initargs=[queue,])

    modifyDataProcessList = []
    jobs = []
    for y in range(N):
        for x in range(N):
                jobs.append((y, x))

    ###print("fitting...")
    processProcessResultsThread = Process(target=processThreadResults, args=("processProcessResultsThread", queue, 16, len(jobs)))
    processProcessResultsThread.start()
    results = pool.map(get_prediction, jobs)
    pool.close()

    processProcessResultsThread.join()

    ###print("finished fitting")

    diff = (shared_u_data[trainLength:]-shared_prediction)
    mse = np.mean((diff)**2)
    print("test error: {0}".format(mse))
    print("inner test error: {0}".format(np.mean((diff[:, patch_radius:N-patch_radius, patch_radius:N-patch_radius])**2)))

    viewData = [("Orig", shared_u_data[trainLength:]), ("Pred", shared_prediction), ("Source", shared_v_data[trainLength:]), ("Diff", diff)]
    directionName = "utov" if reverseDirection else "vtou"
    f = open("../cache/viewdata/{0}/{1}_viewdata_{2}_{3}_{4}_{5}_{6}.dat".format(directionName, predictionMode.lower(), ndata, sigma, sigma_skip, ddim, k), "wb")
    pickle.dump(viewData, f)
    f.close()

    print("done")

if __name__== '__main__':
    mainFunction()
