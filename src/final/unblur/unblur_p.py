import unblur_mt_p as ubmtp
import os
import argparse
import numpy as np

def parse_arguments():
    ubmtp.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="u", nargs=1, type=str, help="u: unblur u, v: unblurr v")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        ubmtp.direction = args.direction[0]

    if args.mode[0] not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        ubmtp.predictionMode = args.mode[0]

    print("Prediction via {0}".format(ubmtp.predictionMode))

def setup_constants():
    id = ubmtp.id
    direction = ubmtp.direction

    print("Using parameters:")

    if (ubmtp.predictionMode == "ESN"):
        ubmtp.sparseness = {"v": [.2,.2,.2,.2,.1,.1,.2], "u": [.1,.1,.1,.1,.1,.1,.1]}[direction][id-1]
        ubmtp.random_seed = {"v": [42,41,41,39,40,42,39], "u": [39,39,40,41,42,41,42]}[direction][id-1]
        ubmtp.n_units = {"v": [200,50,200,50,50,50,50], "u": [50,200,400,50,50,50,50]}[direction][id-1]
        ubmtp.spectral_radius = {"v": [0.95,0.1,0.1,3.0,0.1,0.5,3.0], "u": [1.1,1.1,0.95,0.95,1.5,1.5,0.95]}[direction][id-1]
        ubmtp.regression_parameter = {"v": [5e-5,5e-06, 5e-03, 5e-06, 5e-06, 5e-03, 5e-06], "u": [5e-6,5e-06, 5e-06, 5e-06, 5e-06, 5e-04, 5e-04]}[direction][id-1]
        ubmtp.leaking_rate = {"v": [0.05,0.05,0.05,0.05,0.05,0.05,0.05], "u": [0.2,0.2,0.2,0.05,0.2,0.05,0.05]}[direction][id-1]
        ubmtp.noise_level = {"v": [1e-5,1e-5,1e-4,1e-5,1e-5,1e-4,1e-4], "u": [1e-5,1e-5,1e-4,1e-5,1e-5,1e-4,1e-4]}[direction][id-1]
        ubmtp.sigma = [1, 3, 5, 5, 7, 7, 7][id-1]
        ubmtp.sigma_skip = [1, 1, 1, 2, 1, 2, 3][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(ubmtp.trainLength, ubmtp.sigma, ubmtp.sigma_skip, ubmtp.n_units, ubmtp.regression_parameter))

    elif (ubmtp.predictionMode == "NN"):
        ubmtp.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][id-1]
        ubmtp.k = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][id-1]
        ubmtp.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][id-1]
        ubmtp.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][id-1]

        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(ubmtp.trainLength, ubmtp.sigma, ubmtp.sigma_skip, ubmtp.ddim, ubmtp.k))

    elif (ubmtp.predictionMode == "RBF"):
        basisPoints = 100

        ubmtp.ddim = [3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,  3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,][id-1]
        ubmtp.width = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,][id-1]
        ubmtp.sigma = [1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,  1,3,5,7,5,7,7,1,3,5,7,5,7,7,1,3,5,7,5,7,7,][id-1]
        ubmtp.sigma_skip = [1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,  1,1,1,1,2,2,3,1,1,1,1,2,2,3,1,1,1,1,2,2,3,][id-1]
        print("\t trainLength \t = {0} \n\t sigma \t = {1}\n\t sigma_skip \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}".format(ubmtp.trainLength, ubmtp.sigma, ubmtp.sigma_skip, ubmtp.ddim, ubmtp.width))

    else:
        raise ValueError("No valid predictionMode choosen! (Value is now: {0})".format(ubmtp.predictionMode))

    ubmtp.eff_sigma = int(np.ceil(ubmtp.sigma/ubmtp.sigma_skip))
    ubmtp.patch_radius = ubmtp.sigma//2

if __name__== '__main__':
    #turn this flag on to use the mutual information to calculate the input scaling
    ubmtp.useInputScaling = False

    parse_arguments()
    setup_constants()

    ubmtp.mainFunction()
