import numpy as np
import scipy.sparse.linalg as linalg
import scipy as sp



N = 150
n_units = N*N
np.set_printoptions(threshold=n_units*n_units+10, linewidth=200)

W = np.zeros((n_units, n_units))

W[0, 0] = 1.0
W[0, 1] = 1.0
for i in range(1, N - 1):
    W[i, i] = 1.0
    W[i, i-1] = 1.0
    W[i, i+1] = 1.0
    W[i, i+N] = 1.0
    W[i, i+N-1] = 1.0
    W[i, i+N+1] = 1.0

for i in range(N - 1, n_units - N - 1):
    if (i % 100 == 0):
        print(i/n_units)
    W[i, i] = 1.0
    W[i, i-N] = 1.0
    W[i, i+N] = 1.0

    if i % N == 0:
        #left border
        W[i, i+1] = 1.0
        W[i, i-N+1] = 1.0
        W[i, i+N+1] = 1.0
    elif i % N == N-1:
        W[i, i-1] = 1.0
        W[i, i-N-1] = 1.0
        W[i, i+N-1] = 1.0
    else:
        W[i, i+1] = 1.0
        W[i, i-N+1] = 1.0
        W[i, i+N+1] = 1.0

        W[i, i-1] = 1.0
        W[i, i-N-1] = 1.0
        W[i, i+N-1] = 1.0

for i in range(n_units - N - 1, n_units-1):
    W[i, i] = 1.0
    W[i, i-1] = 1.0
    W[i, i+1] = 1.0
    W[i, i-N] = 1.0
    W[i, i-N-1] = 1.0
    W[i, i-N+1] = 1.0
W[n_units-1, n_units-1] = 1.0
W[n_units-1, n_units-2] = 1.0

W = sp.sparse.dia_matrix(W)
#print(W)
#print(np.sum((W == 1.0))/(n_units*n_units))

eigenvalue, _ = linalg.eigs(W, 1)
print(np.abs(eigenvalue))
W /= eigenvalue
print(np.abs(eigenvalue))
input("aa")
