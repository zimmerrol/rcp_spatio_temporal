import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from ESN import ESN
import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("MackeyGlass_t17.txt").reshape(-1,1)

def generation():
    y_train = y[:2000]
    y_test = y[2000:4000]

    esn = ESN(n_input=1, n_output=1, n_reservoir=1000, noise_level=0.0001, spectral_radius=1.25, leak_rate=0.7, random_seed=42)
    train_acc = esn.fit(inputData=y_train[:-1], outputData=y_train[1:], regression_parameter=4e-7)
    print("training acc: {0:4f}\r\n".format(train_acc))

    y_test_pred = esn.generate(n=len(y_test), initial_input=y_train[-1]).T

    mse = np.mean( (y_test_pred-y_test)[:500]**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(y_test)
    print("testing mse: {0}".format(mse))
    print("testing rmse: {0:4f}".format(rmse))
    print("testing nrmse: {0:4f}".format(nrmse))

    plt.plot(y_test_pred, "g")
    plt.plot( y_test, "b")
    plt.show()

def pred48():
    y_train = y[:8000]
    y_test = y[8000-48:]

    esn = ESN(n_input=1, n_output=1, n_reservoir=500, noise_level=0.001, spectral_radius=.3, leak_rate=0.2, random_seed=42, sparseness=0.2)
    train_acc = esn.fit(inputData=y_train[:-48], outputData=y_train[48:])
    print("training acc: {0:4f}\r\n".format(train_acc))

    y_test_pred = esn.predict(y_test[:-48]).T

    mse = np.mean( (y_test_pred-y_test[48:])[:]**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(y_test)
    print("testing mse: {0}".format(mse))
    print("testing rmse: {0:4f}".format(rmse))
    print("testing nrmse: {0:4f}".format(nrmse))

    plt.plot(y_test_pred, "g", label="prediction")
    plt.plot(y_test[48:], "b", label="target")
    plt.legend()
    plt.show()

pred48()
