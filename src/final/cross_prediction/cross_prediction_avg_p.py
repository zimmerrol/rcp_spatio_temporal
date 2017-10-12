"""
    Parses the arguments and sets the constants etc. to run the real cross prediction
    code (in cross_prediction_mt_p.py) on a unix device.
"""

import os
import argparse
import numpy as np
import cross_prediction_avg_settings as cpmtps

def parse_arguments():
    cpmtps.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="vu", nargs=1, type=str, help="vu: v -> u, uv: u -> v, hv: h -> v, vh: v -> h, bocf_uv: BOCF u -> v, bocf_uw: BOCF u -> w, bocf_us: BOCF u -> s")
    args = parser.parse_args()

    if args.direction[0] not in ["vu", "uv", "hv", "vh", "bocf_uv", "bocf_uw", "bocf_us"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        cpmtps.direction = args.direction[0]

    if args.mode[0] not in ["ESN", "ESN2", "NN", "NNT", "NN3", "RBF", "RBFP", "RBF3"]:
        raise ValueError("No valid prediction_mode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        cpmtps.prediction_mode = args.mode[0]

    if cpmtps.direction.startswith("bocf"):
        cpmtps.N = 150
        cpmtps.ndata = 20000

    print("Prediction via {0}: {1}".format(cpmtps.prediction_mode, cpmtps.direction))

def setup_constants():
    sge_id = cpmtps.id
    direction = cpmtps.direction

    print("Using parameters:")

    if cpmtps.prediction_mode == "ESN":
        cpmtps.n_units = {"hv": [50,400, 50, 50, 400, 200, 50],
                         "vh": [50, 400, 400, 400, 200, 200, 50],
                         "vu": [50, 400, 400, 400, 400, 400, 400],
                         "uv": [200, 400, 400, 400, 400, 400, 400],
                         "bocf_uv": [50, 200, 200, 400, 400, 400, 400],
                         "bocf_uw": [400,400,400,400,400,400,400,],
                         "bocf_us": [50,200,200,400,400,400,400,]}[direction][sge_id-1]
        cpmtps.spectral_radius = {"hv": [3,1.5, 1.5, 1.5, 3.0, 3.0, 3.0],
                                 "vh": [1.1, 0.95, 1.1, 0.1, 1.1, 1.1, 0.95],
                                 "vu": [1.1, 1.1, 0.8, 1.1, 1.5, 1.1, 0.5],
                                 "uv": [1.1, 0.95, 3.0, 0.5, 3.0, 3.0, 0.1],
                                 "bocf_uv": [1.1, 0.95, 1.1, 3.0, 1.5, 1.5, 1.5],
                                 "bocf_uw": [1.1,1.1,1.5,1.5,1.5,1.5,1.5,],
                                 "bocf_us": [1.1,0.95,1.1,3.0,1.5,1.5,1.5,]}[direction][sge_id-1]
        cpmtps.leaking_rate = {"hv": [0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                              "vh": [0.95, 0.5, 0.9, 0.95, 0.5, 0.9, 0.05],
                              "vu": [0.95, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2],
                              "uv": [0.05, 0.05, 0.05, 0.05, 0.05, 0.5, 0.05],
                              "bocf_uv": [0.95, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05],
                              "bocf_uw": [0.05,0.05,0.05,0.05,0.05,0.05,0.05,],
                              "bocf_us": [0.95,0.2,0.2,0.05,0.05,0.05,0.05,]}[direction][sge_id-1]
        cpmtps.random_seed = {"hv": [40,40, 41, 40, 39, 39, 40],
                             "vh": [41, 42, 39, 41, 40, 40, 39],
                             "vu": [42, 40, 41, 40, 40, 40, 42],
                             "uv": [39, 40, 40, 40, 41, 40, 40],
                             "bocf_uv": [41, 41, 40, 41, 41, 40, 39],
                             "bocf_uw": [42,40,40,40,40,40,42,],
                             "bocf_us": [41,41,40,41,41,40,39,]}[direction][sge_id-1]
        cpmtps.sparseness = {"hv": [.1,.1,.1,.1,.1,.1,.1],
                            "vh": [.1, .1,.1,.1,.2,.2,.2],
                            "vu": [.1,.2,.2,.1,.1,.1,.2],
                            "uv": [.1,.1,.2,.1,.1,.1,.1,],
                            "bocf_uv": [0.1, 0.22, 0.1, 0.1, 0.1, 0.2, 0.1],
                            "bocf_uw": [0.1,0.1,0.1,0.2,0.1,0.1,0.2,],
                            "bocf_us": [0.1,0.2,0.1,0.1,0.1,0.2,0.1,]}[direction][sge_id-1]
        cpmtps.noise_level = {"hv": [1e-5,1e-4,1e-4,1e-5,1e-4,1e-5,1e-5],
                             "vh": [1e-5,1e-4,1e-4,1e-4,1e-5,1e-5,1e-5],
                             "vu": [1e-5,1e-5,1e-4,1e-5,1e-5,1e-4,1e-4],
                             "uv": [1e-4,1e-4,1e-5,1e-5,1e-4,1e-4,1e-5],
                             "bocf_uv": [1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4],
                             "bocf_uw": [1e-5, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4],
                             "bocf_us": [1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]}[direction][sge_id-1]
        cpmtps.regression_parameter = {"hv": [5e-2,5e-04, 5e-03, 5e-04, 5e-02, 5e-02, 5e-02],
                                      "vh": [5e-6, 5e-06, 5e-03, 5e-04, 5e-03, 5e-02, 5e-02],
                                      "vu": [5e-6,5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06],
                                      "uv": [5e-6,5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06],
                                      "bocf_uv": [5e-6, 5e-3, 5e-2, 5e-6, 5e-2, 5e-2, 5e-3],
                                      "bocf_uw": [5e-6, 5e-3, 5e-2, 5e-4, 5e-2, 5e-2, 5e-3],
                                      "bocf_us": [5e-6, 5e-3, 5e-2, 5e-6, 5e-2, 5e-2, 5e-3]}[direction][sge_id-1]
        cpmtps.sigma = [1, 3, 5, 5, 7, 7, 7][sge_id-1]
        cpmtps.sigma_skip = [1, 1, 1, 2, 1, 2, 3][sge_id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(cpmtps.trainLength, cpmtps.sigma, cpmtps.sigma_skip, cpmtps.n_units, cpmtps.regression_parameter))
    elif cpmtps.prediction_mode == "NN":
        cpmtps.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][sge_id-1]
        cpmtps.k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][sge_id-1]
        cpmtps.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][sge_id-1]
        cpmtps.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][sge_id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(cpmtps.trainLength, cpmtps.sigma, cpmtps.sigma_skip, cpmtps.ddim, cpmtps.k))

    elif cpmtps.prediction_mode == "NNT":
        cpmtps.prediction_mode = "NN"

        cpmtps.sigma = {"vh":7, "uv": 1}[cpmtps.direction]
        cpmtps.sigma_skip = {"vh":1, "uv": 1}[cpmtps.direction]
        cpmtps.ddim = {"vh":3, "uv": 3}[cpmtps.direction]
        cpmtps.k = {"vh":5, "uv": 5}[cpmtps.direction]

        cpmtps.trainLength = 1000*np.arange(2,29)[sge_id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(
            cpmtps.trainLength, cpmtps.sigma, cpmtps.sigma_skip, cpmtps.ddim, cpmtps.k))

    elif cpmtps.prediction_mode == "RBF":
        cpmtps.basis_points = 100

        cpmtps.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][sge_id-1]
        cpmtps.width = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,][sge_id-1]
        cpmtps.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][sge_id-1]
        cpmtps.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][sge_id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}\n\t basis_points = {5}".format(
            cpmtps.trainLength, cpmtps.sigma, cpmtps.sigma_skip, cpmtps.ddim, cpmtps.width, cpmtps.basis_points))

    elif cpmtps.prediction_mode == "RBFP":
        cpmtps.prediction_mode = "RBF"

        cpmtps.sigma = {"vh": 1, "hv": 5, "uv": 1, "vu": 5}[direction]
        cpmtps.sigma_skip = {"vh": 1, "hv": 2, "uv": 1, "vu": 1}[direction]
        cpmtps.ddim = {"vh": 3, "hv": 5, "uv": 3, "vu": 3}[direction]

        cpmtps.width = np.tile([.5, 1.0, 3.0, 5.0, 7.0, 9.0], 22)[sge_id-1]
        cpmtps.basis_points = np.repeat([5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400], 6)[sge_id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}\n\t basis_points = {5}".format(
            cpmtps.trainLength, cpmtps.sigma, cpmtps.sigma_skip, cpmtps.ddim, cpmtps.width, cpmtps.basis_points))

    else:
        raise ValueError("No valid prediction_mode choosen! (Value is now: {0})".format(cpmtps.prediction_mode))

    cpmtps.noise = 0.075
    cpmtps.eff_sigma = int(np.ceil(cpmtps.sigma/cpmtps.sigma_skip))
    cpmtps.patch_radius = cpmtps.sigma//2

if __name__ == '__main__':
    parse_arguments()
    setup_constants()

    import cross_prediction_avg_mt_p as cpmtp
    cpmtp.mainFunction()
