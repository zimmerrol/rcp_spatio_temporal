import os
import numpy as np
from matplotlib import pyplot as plt
from BarkleySimulation import BarkleySimulation
import progressbar

def generate_data(N, trans, sample_rate=1, Ngrid=100):
    Nx = Ngrid
    Ny = Ngrid
    deltaT = 1e-2
    epsilon = 0.08
    delta_x = 0.1
    D = 1/50
    h = D/delta_x**2
    print("h=" + str(h))
    #h = D over delta_x
    a = 0.75
    b = 0.06

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    sim = BarkleySimulation(Nx, Ny, deltaT, epsilon, h, a, b)
    sim.initialize_one_spiral()

    bar = progressbar.ProgressBar(max_value=trans+N, redirect_stdout=True)

    for i in range(trans):
        sim.explicit_step(chaotic=True)
        bar.update(i)

    data = np.empty((N, Nx, Ny))
    for i in range(N):
        for j in range(sample_rate):
            sim.explicit_step(chaotic=True)
        data[i] = sim._u
        data[i] = sim._v
        bar.update(i+trans)

    bar.finish()
    return data

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
    data = generate_data(ndata, 50000, 5, Ngrid=N)
    np.save("cache/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data...")
    data = np.load("cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    print("loading finished")

#find roots
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
                
print(np.mean(np.array(results)))
plt.plot(autoK, autoY)
plt.plot([0, 50], [0, 0])
plt.show()
