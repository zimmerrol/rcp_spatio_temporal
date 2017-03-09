import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode

trainLength = 10000
skipLength = 500
testLength = 5000

def roessler(n_max):
    return integrate([0.0, 0.0, 0.0], n_max, 0.05 )

def D(t, dat):
    x, y, z = dat
    return [-y-z, x+0.25*y, 0.4+(x-8.5)*z]

def integrate(z0, steps, delta_t):
    #z0: start condition

    solver = ode(D).set_integrator('dopri5')
    solver.set_initial_value(z0,0.0)

    t = np.linspace(0.0, delta_t*steps, steps)
    solution = np.empty((steps, 3))
    solution[0] = z0

    iteration = 1
    while solver.successful() and solver.t < steps*delta_t:
        solver.integrate(t[iteration])
        solution[iteration] = solver.y
        iteration += 1

    return solution


data = roessler(16000)
#data = np.loadtxt('MackeyGlass_t17.txt')

#t=np.linspace(0,10*np.pi, 20000)
#data = 2*np.sin(t)
#data = data.reshape((20000,1))

#data = data[:,2].reshape((15000,1))


print(data.shape)

reservoirSize = 2000
reservoirInputCount = 3
reservoirOutputCount = 3

leakRate = 0.7

weightMatrixNorm = 1.0

np.random.seed(42)

#random weight matrix from -0.5 to 0.5
W = np.random.rand(reservoirSize, reservoirSize) - 0.5

#set 20% to zero
mask = np.random.rand(reservoirSize, reservoirSize) > 0.2
W[mask] = 0.0

W_eigenvalues = np.abs(np.linalg.eigvals(W))
W *= weightMatrixNorm / np.max(np.abs(np.linalg.eig(W)[0]))

#random weight matrix for the input from -0.5 to 0.5
W_input = np.random.rand(reservoirSize, 1+reservoirInputCount)-0.5

#define states' matrix
X = np.empty((1+reservoirInputCount+reservoirSize,trainLength-skipLength))

x = np.zeros((reservoirSize,1))

for t in range(trainLength):
    u = data[t].reshape(reservoirInputCount, 1)
    x = (1.0-leakRate)*x + leakRate*np.arctan(np.dot(W_input, np.vstack((1, u))) + np.dot(W, x)) + (np.random.rand()-0.5)*0.01
    if (t >= skipLength):
        #add valueset to the states' matrix
        X[:,t-skipLength] = np.vstack((1,u, x))[:,0]

#define the target values
Y_target = data.T[:,skipLength+1:trainLength+1]
print(Y_target.shape)

#now calculate the readout matrix W_out
regressionParameter = 2e-8
#W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )
W_out = np.dot(Y_target, np.linalg.pinv(X))





Y = np.zeros((reservoirOutputCount,testLength))
u = data[trainLength].reshape(reservoirInputCount,1)
for t in range(testLength):
    x = (1.0-leakRate)*x + leakRate*np.arctan(np.dot(W_input, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(W_out, np.vstack((1,u,x)))
    Y[:,t] = y[:,0]
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLength+t+1].reshape(reservoirInputCount,1)


# compute MSE for the first errorLen time steps
errorLength = 500
mse = np.sum( np.square( data.T[0,trainLength+1:trainLength+errorLength+1] - Y[0,0:errorLength] ) ) / errorLength
print ('MSE = ' + str( mse ))

# plot some signals

Y[np.isinf(Y)] = 0.0
Y[np.isnan(Y)] = 0.0

plt.figure()
plt.plot( data.T[0,trainLength+1:trainLength+testLength+1], 'g', linestyle=":" )
plt.plot( Y[0,:], 'b' , linestyle="--")
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.ylim([-20,20])
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure()
plt.plot( data.T[1,trainLength+1:trainLength+testLength+1], 'g', linestyle=":" )
plt.plot( Y[1,:], 'b' , linestyle="--")
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.ylim([-20,20])
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure()
plt.plot( data.T[2,trainLength+1:trainLength+testLength+1], 'g', linestyle=":" )
plt.plot( Y[2,:], 'b' , linestyle="--")
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.ylim([-20,20])
plt.legend(['Target signal', 'Free-running predicted signal'])

#plt.figure(3).clear()
#plt.bar( range(1+reservoirInputCount+reservoirSize), W_out.T )
#plt.title('Output weights $\mathbf{W}^{out}$')

plt.show()
