"""
	Loads and renderes computated fields from a pickle object and saves the images.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import dill as pickle
import os
import sys

#parse user input
parser = argparse.ArgumentParser(description='Renders the visualisation of the results of a 2D field predicted by ML algorithms and saves the images.')
parser.add_argument('file', type=str, help='the file of the saved visualisation dictionary', nargs=1)
parser.add_argument('--clim', default=[0.0, 1.0], nargs=2, type=float, help="the limit of the fields' values (default: 0.0 1.0)")
parser.add_argument('--times', "--t", default=[2000], type=int, nargs="*")
parser.add_argument('-fieldname', "-fn", type=str, default=None, nargs="*", help='the name of the field to render and save')
parser.add_argument("--extension", "--ext", default="pdf", type=str, help="extension of the image (default: pdf)")
parser.add_argument("--colormap", "--cmp", default=None, type=str, help="colormap")
parser.add_argument("--colorbar", "--cb", action='store_true', help="show colorbar")
parser.add_argument("--axes", "--ax",  action='store_true', help="show axes")
parser.add_argument("--dynamicclim", "--dclim",  action='store_true', help="detect clim dynamically")
args = parser.parse_args()

if args.fieldname is not None and len(args.fieldname) == 0:
	print("You need to specify at least one name of a field to render")
	sys.exit()
if args.fieldname is None:
	print("Rendering all fields as no field name has been specified.")

print("Loading results from '{0}'".format(args.file[0]))

#load results
f = open(args.file[0], "rb")
viewData = pickle.load(f)
f.close()

clim = args.clim
print("Using clim range {0}".format(clim))

data = {}
#convert all input data into a dict
if type(viewData) is list:
	for key, value in viewData:
		data[key] = value
else:
	data = viewData

#for key in data.keys():
#	data[key] = data[key].reshape((-1, 150, 150)).astype(np.float)

if args.fieldname is None:
	args.fieldname = []
	for key in data.keys():
		args.fieldname.append(key)

#show the results
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
print("Rendering results from '{0}'...".format(args.file[0]))
for name in args.fieldname:
	for i in args.times:
		path = os.path.basename(args.file[0]) + ".{0}.{1}.{2}".format(name, i, args.extension)

		if args.dynamicclim:
			clim = [np.min(data[name][i]), np.max(data[name][i])]

		if True:
			fig = plt.figure(figsize=(5,5))
			savemat = plt.imshow(data[name][i], origin="lower", interpolation="none", cmap=args.colormap)

			if not args.axes:
				plt.axis('off')
				plt.gca().set_axis_off()
				plt.subplots_adjust(top = 0.95, bottom = 0.05, left = 0, hspace = 0, wspace = 0)
				plt.margins(0,0)
				plt.gca().xaxis.set_major_locator(NullLocator())
				plt.gca().yaxis.set_major_locator(NullLocator())

			if args.colorbar:
				divider = make_axes_locatable(plt.gca())
				cax = divider.append_axes("bottom", size="5%", pad=0.15)
				saveclb = plt.colorbar(savemat, orientation="horizontal", cax=cax)
				saveclb.ax.tick_params(labelsize=25)
				saveclb.set_clim(vmin=clim[0], vmax=clim[1])

				saveclb.ax.set_xticklabels(saveclb.ax.get_xticklabels(), rotation=315)

				saveclb.draw_all()

			if not (args.colorbar or args.axes):
				plt.gca().set_axis_off()
				plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
				plt.margins(0,0)
				plt.gca().xaxis.set_major_locator(NullLocator())
				plt.gca().yaxis.set_major_locator(NullLocator())
			#plt.tight_layout()
			plt.savefig(path, bbox_inches='tight', pad_inches = 0)

			if args.colorbar:
				saveclb.remove()
			plt.gca().cla()

		print("Saved {0} at time {1} as {2}".format(name, i, path))
