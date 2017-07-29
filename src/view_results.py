"""
	Loads and visualizes computated fields from a pickle object.
"""

from helper import *

import argparse
import dill as pickle
import os

#parse user input
parser = argparse.ArgumentParser(description='Shows the visualisation of the results of a 2D field predicted by ML algorithms.')
parser.add_argument('file', type=str, help='the file of the saved visualisation dictionary', nargs=1)
parser.add_argument('clim', default=None, nargs="*", type=float, help="the limit of the fields' values - by default, a dynamic range will be used")
parser.add_argument('-splitscreen', action='store_true')
args = parser.parse_args()

print("Loading results from '{0}'".format(args.file[0]))

#load results
f = open(args.file[0], "rb")
viewData = pickle.load(f)
f.close()

clim = args.clim
if (len(clim) == 0):
	clim  = None
else:
	print("Using clim range {0}".format(clim))

#calculate the MSE again
if (type(viewData) is dict):
	diff = viewData["diff"]
	print("MSE: {0}".format(np.mean(diff**2)))
else:
	for (name, diff) in viewData:
		if (name.lower() == "diff"):
				print("MSE: {0}".format(np.mean(diff**2)))

for i in range(len(viewData)):
    viewData[i] = (viewData[i][0], viewData[i][1].reshape((1, 150, 150)).astype(np.float))

#show the results
print("Showing results from '{0}'".format(args.file[0]))
if (args.splitscreen == False):
	show_results(viewData, forced_clim=clim)
else:
	show_results_splitscreen(viewData, forced_clim=clim, name=os.path.basename(args.file[0]))
