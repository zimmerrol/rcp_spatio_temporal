"""
    Parses the arguments and sets the constants etc. to run the real prediction
    code (in prediction_mt_p.py) on a unix device.
"""

import os
import argparse
import numpy as np
import prediction_mt_p as pmtp

def parse_arguments():
    pmtp.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('direction', default="u", nargs=1, type=str, help="u (Barkley), v (Mitchell-Schaeffer)")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        pmtp.direction = args.direction[0]

    print("Prediction: {0}".format(pmtp.direction))

def setup_constants():
    sge_id = pmtp.id
    direction = pmtp.direction

    print("Using parameters:")
    pmtp.n_units = {"u": [200, 200, 400, 300, 200, 200, 200, ], "v": []}[direction][sge_id-1]
    pmtp.sparseness = {"u": [0.05, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, ], "v": []}[direction][sge_id-1]
    pmtp.random_seed = {"u": [39, 40, 42, 39, 40, 39, 41, ], "v": []}[direction][sge_id-1]

    pmtp.spectral_radius = {"u": [0.8, 0.8, 0.5, 2.5, 0.95, 1.75, 1.2, ], "v": []}[direction][sge_id-1]
    pmtp.regression_parameter = {"u": [5.00E-08, 5.00E-07, 0.0005, 5.00E-06, 0.05, 5.00E-05, 0.05, ], "v": []}[direction][sge_id-1]
    pmtp.leaking_rate = {"u": [0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5, ], "v": []}[direction][sge_id-1]
    pmtp.noise_level = {"u": [1.00E-05, 0.0001, 1.00E-05, 1.00E-05, 1.00E-05, 0.0001, 1.00E-05, ], "v": []}[direction][sge_id-1]

    pmtp.sigma = [1, 3, 5, 5, 7, 7, 7][sge_id-1]
    pmtp.sigma_skip = [1, 1, 1, 2, 1, 2, 3][sge_id-1]

    pmtp.useInputScaling = True

    print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}"
          .format(pmtp.trainLength, pmtp.sigma, pmtp.sigma_skip, pmtp.n_units, pmtp.regression_parameter))

    pmtp.eff_sigma = int(np.ceil(pmtp.sigma/pmtp.sigma_skip))
    pmtp.patch_radius = pmtp.sigma//2

if __name__ == '__main__':
    parse_arguments()
    setup_constants()

    pmtp.mainFunction()
