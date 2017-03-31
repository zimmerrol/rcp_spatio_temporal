import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import signal

deltaT = 1e-2
maxT = 500.0
Iext = 2.0
tSteps = int(ceil(maxT/deltaT))
tValues = np.arange(tSteps)*deltaT

def I(t):
    if (t > 10.0 ):
        return np.random.rand()*Iext #np.sin(t)*Iext
    else:
        return 0.0

def step(z, t):
    global a, b, r, deltaT
    v, w = z

    v = v + deltaT*(v-1.0/3.0*v**3-w+I(t))
    w = w + deltaT/r*(v-a-b*w)
    return np.array([v, w])

r = 1.0/0.08
a = -0.7
b = 0.8

values = np.empty((tSteps, 2))
values[0] = np.array([0.0, 0.0])

for i, t in enumerate(tValues[1:]):
    values[i+1] = step(values[i], t)

plt.plot(tValues, values)
plt.plot(tValues, np.tanh(values))
plt.show()
