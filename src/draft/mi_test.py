import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.join(parentdir, "barkley"))
sys.path.insert(0, os.path.join(parentdir, "mitchell"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter
import progressbar
import dill as pickle

from ESN import *
from RBF import *
from NN import *

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing
import ctypes
from multiprocessing import process

from helper import *
import barkley_helper as bh
import mitchell_helper as mh
import argparse



N = 150
ndata = 10000
testLength = 2000
trainLength = 15000

sigma = 3
sigma_skip = 1
eff_sigma = int(np.ceil(sigma/sigma_skip))
patch_radius = sigma//2

direction  = "u"

def generate_data(N, trans, sample_rate, Ngrid):
    data = None

    if (direction == "u"):
        if (os.path.exists("../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid)) == False):
            data = bh.generate_uv_data(N, 50000, 5, Ngrid=Ngrid)
            np.save("../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../cache/barkley/raw/{0}_{1}.uv.dat.npy".format(N, Ngrid))
    else:
        if (os.path.exists("../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid)) == False):
            data = mh.generate_vh_data(N, 20000, 50, Ngrid=Ngrid)
            np.save("../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid), data)
        else:
            data = np.load("../cache/mitchell/raw/{0}_{1}.vh.dat.npy".format(N, Ngrid))

    return data[0, :]

data = generate_data(ndata, 20000, 50, Ngrid=N)

def mutualinformation(x, y, bins):
    pxy, _, _ = np.histogram2d(x, y, bins)
    px, _, = np.histogram(x, bins)
    py, _, = np.histogram(y, bins)

    pxy = pxy/np.sum(pxy)
    px = px/np.sum(px)
    py = py/np.sum(py)

    pxy = pxy[np.nonzero(pxy)]
    px = px[np.nonzero(px)]
    py = py[np.nonzero(py)]

    hxy = -np.sum(pxy*np.log2(pxy))
    hx = -np.sum(px*np.log2(px))
    hy = -np.sum(py*np.log2(py))

    MI = hx+hy-hxy

    return MI

def mi(data, y, x, pr, skip):
    return data[:, y - pr:y + pr+1, x - pr:x + pr+1][:, ::skip, ::skip].reshape(len(data), -1)


std_output = np.std(data[:, 50,50])

#Scott's rule
nbins = int(np.ceil(2/(3.5*std_output/np.power(ndata, 1/3))))
print(nbins )
mis = np.zeros(eff_sigma*eff_sigma)
for y in range(150//2-10,150//2+10):
    print(y)
    for x in range(150//2-10,150//2+10):
        ddat = mi(data, y, x, patch_radius, sigma_skip)
        for i in range(ddat.shape[1]):
            mis[i] += (mutualinformation(ddat[:, i], data[:, y,x], nbins))
mis = mis / (20)**2

np.set_printoptions(precision=2)

mis = mis/np.max(mis)
mis = mis.reshape((eff_sigma,-1))
print(mis)
