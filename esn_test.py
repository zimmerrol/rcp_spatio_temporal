from ESN import ESN
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode

trainLength = 5000
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


data = roessler(20000)
data = data[:,:]


print("fitting...")
#esn.save("dat.obj")

#esn = ESN.load("dat.obj")

print("generating...")

mode = "gen"

if mode == "gen":
    print("set up")
    esn = ESN(n_reservoir=2000, n_input=3, n_output=3, leak_rate=0.7, spectral_radius=0.90, random_seed=42)#0.4

    esn.fit(inputData=data[:trainLength,:], outputData=data[:trainLength,:])

    testLength=5000
    Y = esn.generate(n=testLength, initial_input=data[trainLength])
    errorLength = 4000
    mse = np.sum(np.square(data.T[0,trainLength:trainLength+errorLength] - Y[0,:errorLength]))/errorLength
    print ('MSE = ' + str( mse ))

    plt.figure()
    plt.plot( data.T[0,trainLength:trainLength+testLength], 'g', linestyle=":" )
    plt.plot(Y[0,:], 'b' , linestyle="--")
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Target signal', 'Free-running predicted signal'])

    plt.figure()
    plt.plot( data.T[0,trainLength+1:trainLength+testLength+1]-Y[0,:testLength], 'g', linestyle=":" )
    plt.title('Error of target and predicted signals $y(n)$ starting at $n=0$')
    plt.ylim([-10,10])
    plt.legend(['Error of predicted signal'])

    plt.show()

if mode == "pred":
    print("set up")
    esn = ESN(n_reservoir=2000, n_input=3, n_output=3, leak_rate=0.7, spectral_radius=0.40, random_seed=42)#0.4

    esn.fit(inputData=data[:trainLength,0].reshape(trainLength, 1), outputData=data[:trainLength,1].reshape(trainLength, 1))


    Y = esn.predict(inputData=data[trainLength:trainLength+testLength,0], initial_input=data[trainLength-1, 0])
    print("done.")

    # compute MSE for the first errorLen time steps
    errorLength = 4000
    mse = np.sum( np.square(data.T[1,trainLength:trainLength+errorLength] - Y[0,:errorLength] ) ) / errorLength
    print ('MSE = ' + str( mse ))

    # plot some signals


    plt.figure()
    plt.plot( data.T[1,trainLength:trainLength+testLength], 'g', linestyle=":" )
    plt.plot(Y[0,:], 'b' , linestyle="--")
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Target signal', 'Free-running predicted signal'])


    plt.figure()
    plt.plot( data.T[1,trainLength+1:trainLength+testLength+1]-Y[0,:], 'g', linestyle=":" )
    plt.title('Error of target and predicted signals $y(n)$ starting at $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Error of predicted signal'])

    plt.show()
