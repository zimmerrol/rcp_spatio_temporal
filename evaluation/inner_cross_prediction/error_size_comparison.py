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

sizes = [4, 8, 16, 32, 64, 128, 148]

nn_u_errors  = [0.003679282, 0.015717236, 0.094095959, 0.204580931, 0.265710499, np.nan, np.nan]
rbf_u_errors = [0.000537872, 0.005446511, 0.035442545, 0.11308356, 0.139028466, np.nan, np.nan ]
esn_u_errors = [4.79445E-05, 0.001106557, 0.014466029, 0.093005119, 0.130930648, 0.151063438, 0.183800752, ]

nn_v_errors  = [0.004632133, 0.021815471, 0.082946634, 0.106986577, 0.122971027, np.nan, np.nan]
rbf_v_errors = [0.000457106, 0.004913831, 0.034048691, 0.058552435, 0.067503695, np.nan, np.nan ]
esn_v_errors = [0.000233437, 0.001765347, 0.029685474, 0.050612656, 0.063301057, 0.068415167, 0.067612346, ]

plt.plot(sizes, nn_u_errors, "--o", label="NN")
plt.plot(sizes, rbf_u_errors, "--o", label="RBF")
plt.plot(sizes, esn_u_errors, "--o", label="ESN")
plt.xticks([4, 8, 16, 32, 64, 128, 148])
plt.xlabel("Innere Größe $a$")
plt.ylabel("MSE")
plt.legend()
plt.savefig("images/barkley_error_size_comparison.pdf")
#plt.show()

plt.plot(sizes, nn_v_errors, "--o", label="NN")
plt.plot(sizes, rbf_v_errors, "--o", label="RBF")
plt.plot(sizes, esn_v_errors, "--o", label="ESN")
plt.xticks([4, 8, 16, 32, 64, 128, 148])
plt.xlabel("Innere Größe $a$")
plt.ylabel("MSE")
plt.legend()
plt.savefig("images/mitchell_error_size_comparison.pdf")
#plt.show()
