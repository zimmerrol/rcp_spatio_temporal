import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
from scipy import stats
import dill as pickle

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../src/'))
import helper as hp

def gather_data(filename):
    f = open(filename, "rb")
    viewData = pickle.load(f)
    f.close()

    print("data loaded")

    diff = None
    for (name, value) in viewData:
        if name.lower() == "diff":
            diff = value
            break

    N = 150

    indices = []

    for i in range(31):
        outer = (N//2-32+i, N//2+32-i)
        inner = (N//2-31+i, N//2+31-i)
        outer_y, outer_x, inner_y, inner_x = hp.create_patch_indices(outer, outer, inner, inner)
        indices.append((outer_y, outer_x))

    indices.append(([75], [75]))

    errors = []
    for ind in indices:
        mse = np.mean(diff[:, ind[0], ind[1]]**2)
        errors.append(mse)

    return errors

errors1 = gather_data("D:\\rcp\\esn_viewdata_15000_64_1_0.05_50.dat")
errors2 = gather_data("D:\\rcp\\esn_viewdata_15000_64_1_50000.0_400.dat")

plt.plot(errors1, "o", label="$\lambda=0.05$")
plt.plot(errors2, "x", label="$\lambda=50000$")

plt.xlabel("Abstand zum Rand")
plt.ylabel("MSE")

plt.legend()
plt.savefig("images/inner_errors.pdf", bbox_inches='tight')
plt.show()
