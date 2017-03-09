import numpy as np
import matplotlib.pyplot as plt


trainLength = 2000
skipLength = 100
testLength = 2000

data = np.loadtxt('MackeyGlass_t17.txt')
trainingData = data.copy()

reservoirSize = 1000
reservoirInputCount = 1
reservoirOutputCount = 1

leakRate = 0.3

weightMatrixNorm = 1.25

np.random.seed(42)

#random weight matrix from -0.5 to 0.5
W = np.random.rand(reservoirSize, reservoirSize) - 0.5
W_eigenvalues = np.abs(np.linalg.eigvals(W))
W *= weightMatrixNorm / np.max(np.abs(np.linalg.eig(W)[0]))

#random weight matrix for the input from -0.5 to 0.5
W_input = np.random.rand(reservoirSize, 1+reservoirInputCount)-0.5

#define states' matrix
X = np.empty((1+reservoirInputCount+reservoirSize,trainLength-skipLength))

x = np.empty((reservoirSize,1))

for t in range(trainLength):
    u = trainingData[t]
    x = (1.0-leakRate)*x + leakRate*np.arctan(np.dot(W_input, np.vstack((1, u))) + np.dot(W, x))
    if (t >= skipLength):
        #add valueset to the states' matrix
        X[:,t-skipLength] = np.vstack((1,u, x))[:,0]

#define the target values
Y_target = data[None,skipLength+1:trainLength+1]


#now calculate the readout matrix W_out
regressionParameter = 1e-8
W_out = Y_target.dot(X.T).dot(np.linalg.inv(X.dot(X.T) + regressionParameter*np.identity(1+reservoirInputCount+reservoirSize)) )







Y = np.zeros((reservoirOutputCount,testLength))
u = data[trainLength]
for t in range(testLength):
    x = (1.0-leakRate)*x + leakRate*np.arctan(np.dot(W_input, np.vstack((1, u))) + np.dot(W, x))
    y = np.dot(W_out, np.vstack((1,u,x)))
    Y[:,t] = y
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLength+t+1]
print(Y)


# compute MSE for the first errorLen time steps
errorLength = 500
mse = np.sum( np.square( data[trainLength+1:trainLength+errorLength+1] - Y[0,0:errorLength] ) ) / errorLength
print ('MSE = ' + str( mse ))

# plot some signals
plt.plot( data[trainLength+1:trainLength+testLength+1], 'g', linestyle=":" )
plt.plot( Y.T, 'b' , linestyle="--")
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
#plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure(3).clear()
plt.bar( range(1+reservoirInputCount+reservoirSize), W_out.T )
plt.title('Output weights $\mathbf{W}^{out}$')

plt.show()
