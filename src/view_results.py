from helper import *

import argparse
import dill as pickle

parser = argparse.ArgumentParser(description='Shows the visualisation of the results of a 2D field predicted by ML algorithms.')
parser.add_argument('file', type=str,help='the file of the saved visualisation dictionary', nargs=1)
parser.add_argument('clim', default=None, nargs="*", type=int)
args = parser.parse_args()

print("Loading results from '{0}'".format(args.file[0]))

f = open(args.file[0], "rb")
viewData = pickle.load(f)
f.close()

clim = args.clim
if (len(clim) == 0):
	clim  = None
else:
	print("Using clim range {0}".format(clim))

print("Showing results from '{0}'".format(args.file[0]))

if (type(viewData) is dict):
	diff = viewData["diff"]
	print("MSE: {0}".format(np.mean(diff**2)))
else:
	for (name, diff) in viewData:
		if (name.lower() == "diff"):
				print("MSE: {0}".format(np.mean(diff**2)))

show_results(viewData, forced_clim=clim)
