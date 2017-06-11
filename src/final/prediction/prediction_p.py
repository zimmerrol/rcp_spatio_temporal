"""
    Parses the arguments and sets the constants etc. to run the real cross prediction
    code (in cross_prediction_mt_p.py) on a unix device.
"""

import os
import argparse
import numpy as np
import prediction_mt_p as pmtp

def parse_arguments():
    pmtp.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: ESN")
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="u (Barkley), v (Mitchell-Schaeffer)")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        pmtp.direction = args.direction[0]

    if args.mode[0] not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        pmtp.predictionMode = args.mode[0]

    print("Prediction via {0}: {1}".format(pmtp.predictionMode, pmtp.direction))

def setup_constants():
    id = pmtp.id
    direction = pmtp.direction

    print("Using parameters:")

    if (pmtp.predictionMode == "ESN"):
        pmtp.n_units = {"u": [500], "v": []}[direction][id-1]
        pmtp.sparseness = {"u": [.05], "v": []}[direction][id-1]
        pmtp.random_seed = {"u": [41], "v": []}[direction][id-1]

        pmtp.spectral_radius = {"u": [1.2], "v": []}[direction][id-1]
        pmtp.regression_parameter = {"u": [3e-8], "v": []}[direction][id-1]
        pmtp.leaking_rate = {"u": [.95], "v": []}[direction][id-1]
        pmtp.noise_level = {"u": [.0001], "v": []}[direction][id-1]

        pmtp.sigma = [1, 3, 5, 5, 7, 7, 7][id-1]
        pmtp.sigma_skip = [1, 1, 1, 2, 1, 2, 3][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}"
                .format(pmtp.trainLength, pmtp.sigma, pmtp.sigma_skip, pmtp.n_units, pmtp.regression_parameter))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(pmtp.predictionMode))

    pmtp.eff_sigma = int(np.ceil(pmtp.sigma/pmtp.sigma_skip))
    pmtp.patch_radius = pmtp.sigma//2

if __name__ == '__main__':
    parse_arguments()
    setup_constants()

    pmtp.mainFunction()
