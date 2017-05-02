import os
import numpy as np
from matplotlib import pyplot as plt
from helper import *
import progressbar
def autoCorr(dat, k):
    s = np.mean(dat)
    v = np.var(dat)
    N = dat.size

    sum = np.dot((dat[:N-k]-s),(dat[k:]-s))

    return 1/v/(N-k)*sum

N = 150
ndata = 10000
data = None
if (os.path.exists("cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
    print("data missing")
    print("generating data...")
    data = generate_data(ndata, 50000, 50, Ngrid=N)
    np.save("cache/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    print("loading finished")

#find roots
results = []
for i in range(50,100):
    for j in range(50,100):
        autoCorrIn = data[:, i, j]

        autoK = np.arange(200)
        autoY = np.empty(200)

        autoY[0] = autoCorr(autoCorrIn, 0)
        for k in autoK[1:]:
            autoY[k] = autoCorr(autoCorrIn, k)
            if (autoY[k-1] >= 0 and autoY[k] <= 0):
                results.append(k)
                break
                
print(results)
print(np.mean(np.array(results)))
plt.plot(autoK, autoY)
plt.plot([0, 50], [0, 0])
plt.show()

