import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from ESN import ESN
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode

trainLength = 3000
skipLength = 3000
testLength = 3000

def roessler(n_max):
    return integrate([0.0, 0.0, 0.0], n_max+skipLength, 0.1 )[skipLength:]

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

mode = "pred50"
if mode == "gen":
    print("set up")
    esn = ESN(n_reservoir=2000, n_input=3, n_output=3, leak_rate=0.55, spectral_radius=0.60, random_seed=42, weight_generation='advanced')#0.4
    print("fitting...")
    train_error = esn.fit(inputData=data[:trainLength,:], outputData=data[1:trainLength+1,:])
    print("train error: {0:4f}".format(train_error))

    testLength=5000
    print("generating...")
    Y = esn.generate(n=testLength, initial_input=data[trainLength])
    errorLength = 4000

    mse = np.sum(np.square(data[trainLength:trainLength+errorLength, 0] - Y[:errorLength, 0]))/errorLength
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(data[trainLength:trainLength+errorLength, 0])
    print ('MSE = ' + str( mse ))
    print ('RMSE = ' + str( rmse ))
    print ('NRMSE = ' + str( nrmse ))

    plt.figure()
    plt.plot(data[trainLength:trainLength+testLength, 0], 'g', linestyle=":" )
    plt.plot(Y[:, 0], 'b' , linestyle="--")
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Target signal', 'Free-running predicted signal'])

    plt.figure()
    plt.plot( data[trainLength+1:trainLength+testLength+1, 0]-Y[:testLength, 0], 'g', linestyle=":" )
    plt.title('Error of target and predicted signals $y(n)$ starting at $n=0$')
    plt.ylim([-10,10])
    plt.legend(['Error of predicted signal'])

    plt.show()

if mode == "pred50":
    predDist = 50
    print("set up")
    esn = ESN(n_reservoir=300, n_input=3, n_output=3, leak_rate=0.25, spectral_radius=0.80, random_seed=44, weight_generation='advanced', solver="lsqr", regression_parameters=[1e-6])#0.4
    print("fitting...")
    trainError = esn.fit(inputData=data[:trainLength,:], outputData=data[predDist:trainLength+predDist,:])
    print("train error: {0:4f}".format(trainError))

    testLength=3000
    print("generating...")
    #Y = esn.generate(n=testLength, initial_input=data[trainLength+200])
    Y = esn.predict(inputData=data[trainLength:trainLength+testLength,:])
    errorLength = 3000

    mse = np.mean((data[trainLength+predDist:trainLength+errorLength+predDist, 0] - Y[:errorLength, 0])**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(data[trainLength+predDist:trainLength+errorLength+predDist, 0])
    print ('MSE = ' + str( mse ))
    print ('RMSE = ' + str( rmse ))
    print ('NRMSE = ' + str( nrmse ))

    import matplotlib
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble="\\usepackage{mathtools}")

    plt.figure(figsize=(8,5))
    plt.plot( data[trainLength+predDist:trainLength+testLength+predDist, 0], 'r', linestyle=":" )
    plt.plot(Y[:, 0], 'b' , linestyle="--")
    #plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.ylim([-17,23])
    plt.legend(['Signal $x(n+50)$', 'Vorhersage $x\'(n) \\approx x(n+50)$'], loc="upper center", fancybox=True, shadow=True, ncol=2)
    plt.xlabel("Zeitschritt n")
    plt.ylabel("Signal")

    plt.savefig("roessler_pred50.pdf")

    plt.figure()
    plt.plot( data[trainLength+predDist:trainLength+testLength+predDist, 0]-Y[:testLength, 0], 'g', linestyle=":" )
    plt.title('Error of target and predicted signals $y(n)$ starting at $n=0$')
    plt.ylim([-10,10])
    plt.legend(['Error of predicted signal'])

    plt.show()

if mode == "pred100":
    predDist = 100
    print("set up")
    esn = ESN(n_reservoir=300, n_input=3, n_output=3, leak_rate=0.10, spectral_radius=0.40, random_seed=42, weight_generation='advanced')#0.4
    print("fitting...")
    esn.fit(inputData=data[:trainLength,:], outputData=data[predDist+1:trainLength+predDist+1,:])

    testLength=5000
    print("generating...")
    #Y = esn.generate(n=testLength, initial_input=data[trainLength+200])
    Y = esn.predict(inputData=data[trainLength:trainLength+testLength,:], initial_input=data[trainLength-1, :])
    errorLength = 4000

    mse = np.sum(np.square(data[trainLength+predDist:trainLength+errorLength+predDist, 0] - Y[:errorLength, 0]))/errorLength
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(data[trainLength+predDist:trainLength+errorLength+predDist, 0])
    print ('MSE = ' + str( mse ))
    print ('RMSE = ' + str( rmse ))
    print ('NRMSE = ' + str( nrmse ))

    plt.figure()
    plt.plot( data[trainLength+predDist:trainLength+testLength+predDist, 0], 'g', linestyle=":" )
    plt.plot(Y[:, 0], 'b' , linestyle="--")
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Target signal', 'Free-running predicted signal'])

    plt.figure()
    plt.plot( data[trainLength+predDist:trainLength+testLength+predDist, 0]-Y[:testLength, 0], 'g', linestyle=":" )
    plt.title('Error of target and predicted signals $y(n)$ starting at $n=0$')
    plt.ylim([-10,10])
    plt.legend(['Error of predicted signal'])


    plt.show()

if mode == "cross":
    print("set up")
    esn = ESN(n_reservoir=500, n_input=1, n_output=1, leak_rate=0.20,
                spectral_radius=0.70, random_seed=42,
                weight_generation='advanced', solver="pinv")#0.4
    print("fitting...")
    trainError = esn.fit(inputData=data[:trainLength,0].reshape(trainLength, 1),
        outputData=data[:trainLength,1].reshape(trainLength, 1))

    print ('training MSE = ' + str( trainError ))
    print("predicting...")
    Y = esn.predict(inputData=data[trainLength:trainLength+testLength,0])
    print("done.")

    # compute MSE for the first errorLen time steps
    errorLength = 3000
    mse =np.mean((data[trainLength:trainLength+testLength,1]-Y[:,0])**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(data[trainLength:trainLength+errorLength,1])
    print ('MSE = ' + str( mse ))
    print ('RMSE = ' + str( rmse ))
    print ('NRMSE = ' + str( nrmse ))

    import matplotlib
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    plt.rc('text', usetex=True)

    plt.figure(figsize=(8,5))
    plt.plot( data[trainLength:trainLength+testLength,1], 'r', linestyle=":" )
    plt.plot(Y[:,0], 'b' , linestyle="--")
    #plt.title('Signal $y(n)$ und vorhergesagtes Signal $v(n)$ beginndend ab $n=0$')
    plt.ylim([-20,20])
    plt.legend(['Signal $y(n)$', 'Vorhergesagtes Signal $v(n)$'], loc="upper center", fancybox=True, shadow=True, ncol=2)
    plt.xlabel("Zeitschritt n")
    plt.ylabel("Signal")
    plt.savefig("roessler_cross_pred.pdf")

    plt.figure(figsize=(8,3))
    plt.plot( data[trainLength:trainLength+testLength,1]-Y[:,0], 'g', linestyle=":" )
    #plt.title('Fehler von $y(n)$ und $v(n)$ beginndend ab $n=0$')
    plt.ylim([-1,1])
    plt.legend(['Fehler des vorhergesagten Signals'], loc="upper center", fancybox=True, shadow=True, ncol=2)
    plt.xlabel("Zeitschritt n")
    plt.ylabel("Differenz $y(n) - v(n)$")
    plt.gcf().subplots_adjust(bottom=0.16)

    plt.savefig("roessler_cross_err.pdf")

    plt.show()
