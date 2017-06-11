"""
Parses the arguments and sets the constants etc. to run the real cross prediction code (in cross_prediction_mt_p.py) on a unix device.
"""

import cross_prediction_mt_p as cpmtp
import os
import argparse
import numpy as np

def parse_arguments():
    cpmtp.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="vu: v -> u, uv: u -> v, hv: h -> v, vh: v -> h")
    args = parser.parse_args()

    if args.direction[0] not in ["vu", "uv", "hv", "vh"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        cpmtp.direction = args.direction[0]

    if args.mode[0] not in ["ESN", "ESN2", "NN", "NNT", "NN3", "RBF", "RBFP", "RBF3"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        cpmtp.predictionMode = args.mode[0]

    print("Prediction via {0}: {1}".format(cpmtp.predictionMode, cpmtp.direction))

def setup_constants():
    id = cpmtp.id
    direction = cpmtp.direction

    print("Using parameters:")

    if (cpmtp.predictionMode == "ESN"):
        cpmtp.sparseness = {"vh": [.1,.1,.1,.1,.1,.1,.1], "hv": [.1, .1,.1,.1,.2,.2,.2] ,"uv": [.1,.2,.2,.1,.1,.1,.2], "vu": [.1,.1,.2,.1,.1,.1,.1,]}[direction][id-1]
        cpmtp.random_seed = {"vh": [40,40, 41, 40, 39, 39, 40], "hv": [41, 42, 39, 41, 40, 40, 39] ,"uv": [42,40, 41, 40, 40, 40, 42], "vu": [40,40, 40, 40, 41, 40, 40]}[direction][id-1]
        cpmtp.n_units = {"vh": [50,400, 50, 50, 400, 200, 50], "hv": [50, 400, 400, 400, 200, 200, 50] ,"uv": [50,400, 400, 400, 400, 400, 400], "vu": [400,400, 400, 400, 400, 400, 400]}[direction][id-1]
        cpmtp.spectral_radius = {"vh": [3,1.5, 1.5, 1.5, 3.0, 3.0, 3.0], "hv": [1.1, 0.95, 1.1, 0.1, 1.1, 1.1, 0.95] ,"uv": [1.1,1.1, 0.8, 1.1, 1.5, 1.1, 0.5], "vu": [0.1,0.95, 3.0, 0.5, 3.0, 3.0, 0.1]}[direction][id-1]
        cpmtp.regression_parameter = {"vh": [5e-2,5e-04, 5e-03, 5e-04, 5e-02, 5e-02, 5e-02], "hv": [5e-6, 5e-06, 5e-03, 5e-04, 5e-03, 5e-02, 5e-02], "uv": [5e-6,5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06], "vu": [5e-6,5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06]}[direction][id-1]
        cpmtp.leaking_rate = {"vh": [0.05,0.05,0.05,0.05,0.05,0.05,0.05], "hv": [0.95, 0.5, 0.9, 0.95, 0.5, 0.9, 0.05], "uv": [0.95,0.9, 0.2, 0.2, 0.2, 0.2, 0.2], "vu": [0.05,0.05, 0.05, 0.05, 0.05, 0.5, 0.05]}[direction][id-1]
        cpmtp.noise_level = {"vh": [1e-5,1e-4,1e-4,1e-5,1e-4,1e-5,1e-5], "hv": [1e-5,1e-4,1e-4,1e-4,1e-5,1e-5,1e-5], "uv": [1e-5,1e-5,1e-4,1e-5,1e-5,1e-4,1e-4] , "vu": [1e-5,1e-4,1e-4,1e-5,1e-4,1e-4,1e-5]}[direction][id-1]
        cpmtp.sigma = [1, 3, 5, 5, 7, 7, 7][id-1]
        cpmtp.sigma_skip = [1, 1, 1, 2, 1, 2, 3][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(cpmtp.trainLength, cpmtp.sigma, cpmtp.sigma_skip, cpmtp.n_units, cpmtp.regression_parameter))
    elif (cpmtp.predictionMode == "NN"):
        cpmtp.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][id-1]
        cpmtp.k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        cpmtp.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][id-1]
        cpmtp.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(cpmtp.trainLength, cpmtp.sigma, cpmtp.sigma_skip, cpmtp.ddim, cpmtp.k))

    elif (cpmtp.predictionMode == "NNT"):
        cpmtp.predictionMode = "NN"

        cpmtp.sigma = {"vh":7, "uv": 1}[cpmtp.direction]
        cpmtp.sigma_skip = {"vh":1, "uv": 1}[cpmtp.direction]
        cpmtp.ddim = {"vh":3, "uv": 3}[cpmtp.direction]
        cpmtp.k = {"vh":5, "uv": 5}[cpmtp.direction]

        cpmtp.trainLength = 1000*np.arange(2,29)[id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(cpmtp.trainLength, cpmtp.sigma, cpmtp.sigma_skip, cpmtp.ddim, cpmtp.k))
    elif (cpmtp.predictionMode == "RBF"):
        cpmtp.basisPoints = 100

        cpmtp.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][id-1]
        cpmtp.width = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,][id-1]
        cpmtp.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][id-1]
        cpmtp.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}\n\t basisPoints = {5}".format(cpmtp.trainLength, cpmtp.sigma, cpmtp.sigma_skip, cpmtp.ddim, cpmtp.width, cpmtp.basisPoints))

    elif (cpmtp.predictionMode == "RBFP"):
        cpmtp.predictionMode = "RBF"

        cpmtp.sigma = {"vh": 1, "hv": 5, "uv": 1, "vu": 5}[direction]
        cpmtp.sigma_skip = {"vh": 1, "hv": 2, "uv": 1, "vu": 1}[direction]
        cpmtp.ddim = {"vh": 3, "hv": 5, "uv": 3, "vu": 3}[direction]

        cpmtp.width = np.tile([.5, 1.0, 3.0, 5.0, 7.0, 9.0], 22)[id-1]
        cpmtp.basisPoints = np.repeat([5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400], 6)[id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}\n\t basisPoints = {5}".format(cpmtp.trainLength, cpmtp.sigma, cpmtp.sigma_skip, cpmtp.ddim, cpmtp.width, cpmtp.basisPoints))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(cpmtp.predictionMode))

    cpmtp.eff_sigma = int(np.ceil(cpmtp.sigma/cpmtp.sigma_skip))
    cpmtp.patch_radius = cpmtp.sigma//2

if __name__== '__main__':
    parse_arguments()
    setup_constants()

    cpmtp.mainFunction()
