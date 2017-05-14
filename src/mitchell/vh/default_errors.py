# -*- coding: utf-8 -*-

#TODO: Use http://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ESN import ESN
import progressbar
import dill as pickle #we require version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import copy
from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#from pathos.multiprocessing import Pool
import multiprocessing
import ctypes

from helper import *
from helper_mitchellschaeffer import *

N = 150
ndata = 10000
ntest = 2000

def mainFunction():
    if (os.path.exists("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N)) == False):
        print("generating data...")
        data = generate_vh_data(ndata, 20000, 50, Ngrid=N) #20000 was 50000 ndata
        np.save("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N), data)
        print("generating finished")
    else:
        print("loading data...")
        data = np.load("../cache/raw/{0}_{1}.vh.dat.npy".format(ndata, N))

        
        #at the moment we are doing a h -> v cross prediction.
        #switch the entries for the v -> h prediction
        tmp = data[0].copy()
        data[0] = data[1].copy()
        data[1] = tmp.copy()
        

        print("loading finished")


  

    training_data = data[:, :ndata-ntest]
    test_data = data[:, ndata-ntest:]
    
    print(test_data[0].shape)
   
    print(np.mean(test_data[0]))
    
    #use mean value as prediciton:
    mean = np.mean(training_data[0])
    meanpredmse = np.mean((test_data[0] - mean)**2)
    
    diff = np.abs(test_data[0] - mean)
    #diff[diff < 0.04] = 0.0
    #diff[diff > 0.04] = 1.0
    
    print(np.mean(np.multiply(diff, diff)))
    
    show_results([("test", test_data[0]), ("error", diff)])
    
    print("mean value: {0}".format(mean))
    
    #use h as value for v
    hvpredmse = np.mean((test_data[0] - test_data[1])**2)
    
    print("Using the mean of v_train as prediction: ")
    print("\tMSE = {0}".format(meanpredmse))
    print("\tmedian of SE = {0}".format(np.median((test_data[0] - mean)**2)))
    
    print("Using the value of h as the v prediction: ")
    print("\tMSE = {0}".format(hvpredmse))


class ForceIOStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if not self.stream.isatty():
            os.fsync(self.stream.fileno())

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

if __name__== '__main__':
    sys.stdout = ForceIOStream(sys.stdout)
    sys.stderr = ForceIOStream(sys.stderr)

    mainFunction()
