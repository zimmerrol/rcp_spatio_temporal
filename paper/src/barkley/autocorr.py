"""
    Calculates the optimal delay time tau for the delay reconstruction using the auto correlation for the Barkley model.
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from BarkleySimulation import BarkleySimulation
import progressbar
from barkley_helper import generate_uv_data


"""
    Generate the data of the model.
"""
def generate_data(N, trans, sample_rate=1, Ngrid=100):
    #return the u variable
    return generate_uv_data(N, trans, sample_rate=sample_rate, Ngrid=Ngrid)[0]

"""
    Calculates the auto correlation of dat with delay k.
"""
def autoCorr(dat, k):
    s = np.mean(dat)
    v = np.var(dat)
    N = dat.size

    sum = np.dot((dat[:N-k]-s),(dat[k:]-s))

    return 1/v/(N-k)*sum

N = 150
ndata = 10000
data = None

#load or generate the data
if (os.path.exists("cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
    print("data missing")
    print("generating data...")
    data = generate_data(ndata, 50000, 5, Ngrid=N)
    np.save("cache/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    print("loading finished")

#find roots of the auto correlation for every pixel
results = []
for i in range(150):
    for j in range(150):
        autoCorrIn = data[:, i, j]

        autoK = np.arange(200)
        autoY = np.empty(200)

        autoY[0] = autoCorr(autoCorrIn, 0)
        for k in autoK[1:]:
            autoY[k] = autoCorr(autoCorrIn, k)
            if (autoY[k-1] >= 0 and autoY[k] <= 0):
                results.append(k)
                break

#calculate the mean of the first root for each pixel and print it
print(np.mean(np.array(results)))

#plot the auto correlation
plt.plot(autoK, autoY)
plt.plot([0, 50], [0, 0])
plt.show()
