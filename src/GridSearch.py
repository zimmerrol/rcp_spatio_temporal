import numpy as np
import itertools
from ESN import ESN

class GridSearch:
    def __init__(self, param_grid, fixed_params, esnType):
        self.esnType = esnType
        self.param_grid = param_grid
        self.fixed_params = fixed_params


    def fit(self, trainingInput, trainingOutput, testingDataSequence):
        def enumerate_params():
            keys, values = zip(*self.param_grid.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))

        results = []

        for params in enumerate_params():
            esn = self.esnType(**params, **self.fixed_params)
            training_acc = esn.fit(trainingInput, trainingOutput)

            current_state = esn._x

            test_mse = []
            for (testInput, testOutput) in testingDataSequence:
                esn._x = current_state
                out_pred = esn.predict(testInput).T
                test_mse.append(np.mean((testOutput - out_pred)**2))

            test_mse = np.mean(test_mse)

            results.append((test_mse, params))

        import operator
        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[1]
        self._best_mse = res[0]

        return results
