import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.colorbar as cbar
from scipy import stats
import pandas as pd
import dill as pickle

f_weights = open("weights.dat", "rb")
weights = pickle.load(f_weights)
f_weights.close()

fig, ax = plt.subplots()
mat = plt.imshow(weights.T, vmin=-2, vmax=2)
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = make_axes_locatable(ax).append_axes("top", size="07.5%", pad=0.25)
clb = fig.colorbar(mat, cax=cax, orientation="horizontal")
clb.ax.xaxis.set_ticks_position('top')
clb.ax.xaxis.set_label_position('top')

ax.set_ylabel("Eintrag $i$")
ax.set_xlabel("Pixel")

plt.savefig("images/weights.pdf", bbox_inches='tight')

plt.show()
