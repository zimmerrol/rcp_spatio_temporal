import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from BarkleySimulation import BarkleySimulation
from ESN import ESN
from helper import *
import progressbar

N = 150
ndata = 10000
trainLength = 2000
n_units = 5000

#there is a difference between odd and even numbers
#odd size  => there is a center point and the left and the right area without this center are even spaced
#even size => right and left half of the square are even spaced

"""
even      odd
aaaaaaaa  aaaaaaaaa  
a┌────┐a  a┌─────┐a
a│ooxx│a  a│oo0xx│a
a│ooxx│a  a│oo0xx│a
a│ooxx│a  a│oo0xx│a
a│ooxx│a  a│oo0xx│a
a└────┘a  a│oo0xx│a
aaaaaaaa  a└─────┘a
          aaaaaaaaa
"""

innerSize = 40 #see above
halfInnerSize = int(np.floor(innerSize / 2))
borderSize = 1
center = N//2
rightBorderAdd = 1 if innerSize != 2*halfInnerSize else 0


"""
n_units accuricy
10000   0.18049000161901738
15000   0.17985260548782417
"""

if (os.path.exists("cache/raw/{0}_{1}.dat.npy".format(ndata, N)) == False):
    print("generating data...")
    data = generate_data(ndata, 50000, 5, Ngrid=N)
    np.save("cache/raw/{0}_{1}.dat.npy".format(ndata, N), data)
    print("generating finished")
else:
    print("loading data... from: " + "cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    data = np.load("cache/raw/{0}_{1}.dat.npy".format(ndata, N))
    print("loading finished")



training_data = data[:ndata-trainLength]
test_data = data[ndata-trainLength:]

print(training_data.shape)


input_y, input_x, output_y, output_x = create_patch_indices(
                                            (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                            (center - (halfInnerSize+borderSize), center + (halfInnerSize+borderSize) + rightBorderAdd),
                                            (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                            (center - (halfInnerSize), center + (halfInnerSize) + rightBorderAdd),
                                        )
 

training_data_in =  training_data[:, input_y, input_x].reshape(-1, len(input_y))
training_data_out =  training_data[:, output_y, output_x].reshape(-1, len(output_y))

test_data_in =  test_data[:, input_y, input_x].reshape(-1, len(input_y))
test_data_out =  test_data[:, output_y, output_x].reshape(-1, len(output_y))



merged_prediction = test_data.copy()


def solve_single(merged_prediction):
    esn = ESN(n_input = len(input_y), n_output = 1, n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.2, spectral_radius = 0.1,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e-0], solver = "lsqr")

    print("fitting and predicting...")

    bar = progressbar.ProgressBar(max_value=len(output_y), redirect_stdout=True)
    bar.update(0)

    mse_train = []
    for i in range(len(output_y)):
        y = output_y[i]
        x = output_x[i]
        
        train_error = esn.fit(training_data_in, training_data[:, y, x].reshape((-1,1)), verbose=0)
        mse_train.append(train_error)
        
        pred = esn.predict(test_data_in, verbose=0)
        pred[pred>1.0] = 1.0
        pred[pred<0.0] = 0.0

        merged_prediction[:, y, x] = pred.ravel()
        
        bar.update(i)
    bar.finish()
    
    return (merged_prediction, mse_train)
    
def solve_multi(merged_prediction):
    esn = ESN(n_input = len(input_y), n_output = len(output_y), n_reservoir = n_units,
            weight_generation = "advanced", leak_rate = 0.2, spectral_radius = 0.1,
            random_seed=42, noise_level=0.0001, sparseness=.1, regression_parameters=[5e+1], solver = "lsqr")

    print("fitting and predicting...")
        
    train_error = esn.fit(training_data_in, training_data_out, verbose=1)
    
    pred = esn.predict(test_data_in, verbose=1)
    pred[pred>1.0] = 1.0
    pred[pred<0.0] = 0.0

    merged_prediction[:, output_y, output_x] = pred

    
    return (merged_prediction, train_error)


#merged_prediction, mse_train = solve_single(merged_prediction)
merged_prediction, mse_train = solve_multi(merged_prediction)

#print("predicting...")

print("train MSE: {0}".format(np.mean(mse_train)))

pred = merged_prediction[:, output_y, output_x]


diff = pred.reshape((-1, len(output_y))) - test_data_out
mse = np.mean((diff)**2)

meantrainpredmse = np.mean((np.mean(training_data_out) - test_data_out)**2)
meantborderpredmse = np.mean((np.mean(test_data_in) - test_data_out)**2)

print("test MSE: {0}".format(mse))
print("mean(train) MSE: {0}".format(meantrainpredmse))
print("mean(border) MSE: {0}".format(meantborderpredmse))

print("NMSE by mean(train): {0}".format(mse/meantrainpredmse))
print("NMSE by mean(border): {0}".format(mse/meantborderpredmse))

difference = np.abs(test_data - merged_prediction)

show_results({"pred": merged_prediction , "orig": test_data, "diff": difference})

print("done.")
