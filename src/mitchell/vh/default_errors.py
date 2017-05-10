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

#id=1

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
from GridSearchP import GridSearchP

from helper import *

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

        """
        #switch the entries for the u->v prediction
        tmp = data[0].copy()
        data[0] = data[1].copy()
        data[1] = tmp.copy()
        """

        print("loading finished")


    training_data = data[:, :ndata-ntest]
    test_data = data[:, ndata-ntest:]
    
    print(test_data[0].shape)
    
    #use mean value as prediciton:
    mean = np.mean(training_data[0])
    meanpredmse = np.mean((test_data[0] - mean)**2)
    
    #use h as value for v
    hvpredmse = np.mean((test_data[0] - test_data[1])**2)
    
    print("Using the mean of v_train as prediction: ")
    print("\tMSE = {0}".format(meanpredmse))
    
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
