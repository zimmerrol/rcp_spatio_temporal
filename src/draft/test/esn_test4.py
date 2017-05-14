import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# Code for the Sinewave generator & a simple sine wave generator
# first test for the feedback ESN (FESN)
# is supposed to generate autonomously a sinewave
# input: target frequency

from FESN import FESN
import numpy as np
import matplotlib.pyplot as plt

def simple_sine():
    x = np.linspace(1,100*np.pi, 2000)
    y = (0*np.log(x)+np.sin(x)*np.cos(x)).reshape(2000,1)*2

    #esn = ESN(n_input=1, n_output=1, n_reservoir=200, random_seed=42, noise_level=0.001, leak_rate=0.7, spectral_radius=1.35, sparseness=0.1)
    #esn.fit(inputData=y[:5000, :], outputData=y[:5000,:], transient_quota=0.4, regression_parameter=2e-4)

    esn = FESN(n_input=0, n_output=1, output_bias=1.0, n_reservoir=400, random_seed=42, noise_level=0.001, leak_rate=0.70,
        spectral_radius=1.35, sparseness=0.1)
    esn.fit(inputData=None, outputData=y[0:500,:], transient_quota=0.4)

    print(np.mean(np.abs(esn._W_out)))

    #Y = esn.predict(n=testLength, inputData=None, continuation=True, start_output = y[trainLength-1])
    Y = esn.predict(n=1500, inputData=None, continuation=True, start_output = y[500,:])

    #plt.plot(np.arange(trainLength), Y, 'r')

    plt.plot(x,y[:,0], "b", linestyle="--")
    plt.plot(x[500:],Y[:,0], "r")
    plt.show()

#simple_sine()


def frequency_generator(N,min_period,max_period,n_changepoints):
    """returns a random step function with N changepoints
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    # vector of random indices < N, padded with 0 and N at the ends:
    changepoints = np.insert(np.sort(np.random.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # list of interval boundaries between which the control sequence should be constant:
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    # populate a control sequence
    frequency_control = np.zeros((N,1))
    for (t0,t1) in const_intervals:
        frequency_control[t0:t1] = np.random.rand()

    frequency_control[const_intervals[0][0]:const_intervals[0][1]] = 0.8
    frequency_control[const_intervals[len(const_intervals)-1][0]:const_intervals[len(const_intervals)-1][1]] = 0.2


    periods = frequency_control * (max_period - min_period) + max_period
    # run time through a sine, while changing the period length
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1)/2
    return np.hstack([np.ones((N,1)),1-frequency_control]), frequency_output




def freq_gen():
    np.random.seed(42)

    N = 20000 # signal length
    min_period = 2
    max_period = 10
    n_changepoints = int(N/200)
    frequency_control,frequency_output = frequency_generator(N,min_period,max_period,n_changepoints)

    frequency_control2, frequency_output2 = frequency_generator(N,min_period,max_period,n_changepoints)

    traintest_cutoff = int(np.ceil(0.5*N))

    train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
    test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]


    #esn = FESN(n_input=1, n_output=1, output_bias=0.0, n_reservoir=200, output_input_scaling=1.0, random_seed=42, noise_level=0.001, leak_rate=0.6,
    #    spectral_radius=0.2, sparseness=0.30)

    #for random see 42 for the generation!
    esn = FESN(n_input=1, n_output=1, output_bias=0.0, n_reservoir=200, output_input_scaling=1.0, random_seed=42, noise_level=0.001, leak_rate=0.7,
        spectral_radius=0.13, sparseness=0.30)

    esn.fit(train_ctrl[:,1], train_output, transient_quota=0.1)
    pred_test = esn.predict(test_ctrl[:,1])
    pred_test2 = esn.predict(frequency_control2[:,1])

    #test acc
    print(np.sqrt(np.mean((pred_test - test_output)**2)))
    #second test acc for validation with completly new generated samples
    print(np.sqrt(np.mean((pred_test2 - frequency_output2)**2)))

    #window_test = range(2000)
    plt.figure(figsize=(10,2.5))
    plt.plot(test_ctrl[-2000:,1],label='control')
    plt.plot(test_output[-2000:],label='target')
    plt.plot(pred_test[-2000:],label='model')
    plt.legend(fontsize='x-small')
    plt.title('test (excerpt)')
    plt.ylim([-1.5,1.5]);


    plt.figure(figsize=(10,2.5))
    plt.plot(frequency_control2[:,1],label='control')
    plt.plot(frequency_output2[:],label='target')
    plt.plot(pred_test2[:],label='model')
    plt.legend(fontsize='x-small')
    plt.title('test2 (excerpt)')
    plt.ylim([-1.5,1.5]);

    plt.show()

freq_gen()
#simple_sine()
