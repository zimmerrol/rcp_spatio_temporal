"""
    Parses the arguments and sets the constants etc. to run the real inner cross prediction
    code (in inner_cross_pred_mt_p.py) on a unix device.
"""

import os
import argparse
import inner_cross_pred_mt_p as icpmtp
import numpy as np

def parse_arguments():
    icpmtp.id = int(os.getenv("SGE_TASK_ID", 0))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', nargs=1, type=str, help="Can be: NN, RBF, ESN")
    parser.add_argument('direction', default="u", nargs=1, type=str, help="u: unblur u, v: unblurr v")
    args = parser.parse_args()

    if args.direction[0] not in ["u", "v"]:
        raise ValueError("No valid direction choosen! (Value is now: {0})".format(args.direction[0]))
    else:
        icpmtp.direction = args.direction[0]

    if args.mode[0] not in ["ESN", "NN", "RBF"]:
        raise ValueError("No valid prediction_mode choosen! (Value is now: {0})".format(args.mode[0]))
    else:
        icpmtp.prediction_mode = args.mode[0]

    print("Prediction via {0}".format(icpmtp.prediction_mode))

def setup_constants():
    sge_id = icpmtp.id
    direction = icpmtp.direction

    #there is a difference between odd and even numbers for the inner size (a)
    #odd size  => there is a center point and the left and the right area without this center are even spaced
    #even size => right and left half of the square are even spaced

    """
    even      odd
    aaaaaaaa  aaaaaaaaa
    a┌────┐a  a┌─────┐a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a│ooxx│a  a│oo0xx│a
    a└────┘a  a│oo0xx│a
    aaaaaaaa  a└─────┘a
              aaaaaaaaa
    """

    print("Using parameters:")

    if (icpmtp.prediction_mode == "ESN"):
        """
            These constant values are the result of the GridSearch (done by inner_cross_pred_esn_g.py) and some manual tweaking of them.
            The first block of constants has arised by using a small range of values for the regression parameter lambda. The second block yields
            much better accuracies and has arised from a GridSearch on a broader range of lambda values (including much higher values)
        """

        """
        icpmtp.n_units = {"u": [400,400,50,50,50,200,400,400,50,50,50,200,400,400,50,50,50,200,50,50,50],
                   "v": [50,50,200,200,   400,400,50,50,200,200,50,50,50,200,200,200,50,400,200,200,200]}[direction][sge_id-1]
        icpmtp.seed = {"u": [40,39,42,41,41,39,40,39,42,41,41,39,40,39,42,41,41,39,40,40,42],
                "v": [40,40,42,41,40,41,40,40,42,41,40,40,40,42,41,41,40,41,39,39,39]}[direction][sge_id-1]
        icpmtp.regression_parameter = {"u": [5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-2,5e-6],
                                "v": [5e-2,5e-2,5e-2,5e-3,5e+5,5e-2,5e-2,5e-2,5e-2,5e-3,5e-3,5e-2,5e-2,5e-2,5e-3,5e-3,5e-3,5e-2,5e-2,5e-2,5e-2]}[direction][sge_id-1]
        icpmtp.spectral_radius = {"u": [0.1,3,3,0.1,3,1.5,0.1,3,3,0.1,3,1.5,0.1,3,3,0.1,0.5,1.5,0.1,0.1,0.8],
                           "v": [0.8,3,3,3,1.5,1.5,0.8,3,3,3,1.5,0.8,3,3,3,3,1.5,1.5,0.1,0.1,1.5,]}[direction][sge_id-1]
        icpmtp.leak_rate = {"u": [.95,.9,.9,.05,.5,.95,.95,.9,.9,.05,.5,.95,.95,.9,.9,.05,.5,.95,.2,.2,.05],
                     "v": [.95,.5,.05,.2,.2,.7,.95,.5,.05,.2,.2,.95,.5,.05,.2,.2,.2,.7,.5,.5,.05]}[direction][sge_id-1]
        icpmtp.sparseness = {"u": [2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,1,2,2,1,1,1],
                      "v": [1,1,2,1,2,2,1,1,2,1,2,1,1,2,1,1,2,2,2,2,2]}[direction][sge_id-1]/10
        icpmtp.noise_level = {"u": [1e-4,1e-5,1e-4,1e-5,1e-5,1e-5,1e-4,1e-5,1e-4,1e-5,1e-5,1e-5,1e-4,1e-5,1e-4,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,],
                       "v": [1e-5,1e-5,1e-5,1e-4,1e-5,1e-4,1e-5,1e-5,1e-5,1e-4,1e-5,1e-5,1e-5,1e-5,1e-4,1e-4,1e-5,1e-4,1e-4,1e-4,1e-4,]}[direction][sge_id-1]
        """

        icpmtp.n_units = {"u": [400, 400, 50, 200, 400, 200, 400, 400, 50, 200, 400, 200, 400, 400, 50, 200, 400, 200, 50, 50, 400, ],
                   "v": [400, 50, 200, 50, 400, 200, 400, 50, 200, 50, 400, 200, 400, 50, 200, 50, 400, 200, 200, 200, 200, ]}[direction][sge_id-1]
        icpmtp.seed = {"u": [41, 41, 42, 40, 39, 41, 41, 41, 42, 40, 39, 41, 41, 41, 42, 40, 39, 41, 41, 41, 39, ],
                "v": [42, 42, 42, 42, 39, 41, 42, 42, 42, 42, 39, 41, 42, 42, 42, 42, 39, 41, 39, 39, 40, ]}[direction][sge_id-1]
        icpmtp.regression_parameter = {"u": [5e-4, 5e-1, 5e-1, 5e+3, 5e+4, 5e+3, 5e-4, 5e-1, 5e-1, 5e+3, 5e+4, 5e+3, 5e-4, 5e-1, 5e-1, 5e+3, 5e+4, 5e+3, 5e+2, 5e+2, 5e+2, ],
                                "v": [5e+0, 5e+0, 5e-1, 5e+3, 5e+3, 5e+4, 5e+0, 5e+0, 5e-1, 5e+4, 5e+3, 5e+4, 5e+0, 5e+0, 5e-1, 5e+4, 5e+3, 5e+4, 5e+4, 5e+4, 5e+4, ]}[direction][sge_id-1]
        icpmtp.spectral_radius = {"u": [0.8, 0.5, 0.5, 1.5, 1.5, 3, 0.8, 0.5, 0.5, 1.5, 1.5, 3, 0.8, 0.5, 0.5, 1.5, 1.5, 3, 3, 3, 3, ],
                           "v": [1.5, 0.1, 3, 0.8, 3, 3, 1.5, 0.1, 3, 0.8, 3, 3, 1.5, 0.1, 3, 0.8, 3, 3, 0.1, 0.1, 3, ]}[direction][sge_id-1]
        icpmtp.leak_rate = {"u": [0.7, 0.5, 0.2, 0.05, 0.05, 0.2, 0.7, 0.5, 0.2, 0.05, 0.05, 0.2, 0.7, 0.5, 0.2, 0.05, 0.05, 0.2, 0.2, 0.05, 0.05, ],
                     "v": [0.95, 0.7, 0.95, 0.95, 0.2, 0.05, 0.95, 0.7, 0.05, 0.95, 0.2, 0.05, 0.95, 0.7, 0.05, 0.95, 0.2, 0.05, 0.05, 0.05, 0.05, ]}[direction][sge_id-1]
        icpmtp.sparseness = {"u": [2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2,],
                      "v": [1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, ]}[direction][sge_id-1]/10
        icpmtp.noise_level = {"u": [1e-5, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-4, 1e-5, ],
                       "v": [1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5, ]}[direction][sge_id-1]


        icpmtp.inner_size = [4,8,16,32,64,128, 4,8,16,32,64,128, 4,8,16,32,64,128, 146,146,148][sge_id-1]
        icpmtp.border_size = [1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 1,2,1][sge_id-1]

        icpmtp.constants_setup = True

        print("\t trainLength \t = {0} \n\t a \t = {1}\n\t b \t = {2}\n\t n_units \t = {3}\n\t regular. \t = {4}".format(icpmtp.trainLength, icpmtp.inner_size, icpmtp.border_size, icpmtp.n_units, icpmtp.regression_parameter))

    elif (icpmtp.prediction_mode == "NN"):
        #3*30 elements
        icpmtp.ddim = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][sge_id-1]
        icpmtp.k = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
             4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][sge_id-1]
        icpmtp.inner_size = [4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,][sge_id-1]
        icpmtp.border_size = [1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,
                      1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  1,2,3,  ][sge_id-1]

        print("\t trainLength \t = {0} \n\t a \t = {1}\n\t b \t = {2}\n\t ddim \t = {3}\n\t k \t = {4}".format(
            icpmtp.trainLength, icpmtp.inner_size, icpmtp.border_size, icpmtp.ddim, icpmtp.k))

    elif icpmtp.prediction_mode == "RBF":
        icpmtp.basisPoints = 100

        super_id = 0
        while sge_id > 108:
            sge_id -= 108
            super_id += 1

        icpmtp.ddim = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,][sge_id-1]
        icpmtp.width = [[0.5, 1.0], [3.0, 5.0], [7.0, 9.0]][super_id][[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,][sge_id-1]]
        icpmtp.inner_size = [4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146, 148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  128,128,128, 146,146, 148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  128,128,128, 146,146, 148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146, 148,
                     4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64,  128,128,128, 146,146, 148,  4,4,4, 8,8,8, 16,16,16, 32,32,32, 64,64,64, 128,128,128, 146,146, 148,][sge_id-1]
        icpmtp.border_size = [1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,  1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,
                      1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,  1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,
                      1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,  1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2, 1,][sge_id-1]
        print("\t trainLength \t = {0} \n\t a \t = {1}\n\t b \t = {2}\n\t ddim \t = {3}\n\t width \t = {4}".format(
            icpmtp.trainLength, icpmtp.inner_size, icpmtp.border_size, icpmtp.ddim, icpmtp.width))

    else:
        raise ValueError("No valid prediction_mode choosen! (Value is now: {0})".format(icpmtp.prediction_mode))

    icpmtp.half_inner_size = int(np.floor(icpmtp.innerSize / 2))
    icpmtp.border_size = 1
    icpmtp.center = icpmtp.N//2
    icpmtp.right_border_add = 1 if icpmtp.inner_size != 2*icpmtp.half_inner_size else 0

if __name__ == '__main__':
    parse_arguments()
    setup_constants()
    icpmtp.setup_arrays()

    icpmtp.mainFunction()
