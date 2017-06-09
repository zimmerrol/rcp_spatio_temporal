import numpy as np
import itertools
from ESN import ESN
import operator
import progressbar
import sys

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing

class GridSearchP:
    def __init__(self, param_grid, fixed_params, esnType):
        self.esnType = esnType
        self.param_grid = param_grid
        self.fixed_params = fixed_params

    def _get_score(data):
        params, fixed_params, trainingInput, trainingOutput, testingDataSequence, esnType = data


        output_postprocessor = lambda x:x

        try:
            esn = esnType(**params, **fixed_params)
            training_acc = esn.fit(trainingInput, trainingOutput)

            current_state = esn._x

            test_mse = []
            for (testInput, testOutput) in testingDataSequence:
                esn._x = current_state
                out_pred = output_postprocessor(esn.predict(testInput))
                test_mse.append(np.mean((testOutput - out_pred)**2))

            test_mse = np.mean(test_mse)

            dat = (test_mse, training_acc, params)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            import traceback
            print(traceback.format_exc())

            dat = ([np.nan]*len(testingDataSequence), [np.nan]*len(testingDataSequence), params)


        GridSearchP._get_score.q.put(dat)

        return dat

    def _get_score_init(q):
        GridSearchP._get_score.q = q

    def processThreadResults(threadname, q, numberOfWorkers, numberOfResults, verbose):
        if (verbose == 0):
            return

        if (verbose == 1):
            bar = progressbar.ProgressBar(max_value=numberOfResults, redirect_stdout=True)
            bar.update(0)
        finishedResults = 0

        print_step = numberOfResults//200

        print(print_step)

        while True:
            if (finishedResults == numberOfResults):
                break

            newData = q.get()
            finishedResults += 1

            if (verbose == 1):
                bar.update(finishedResults)
            else:
                if (finishedResults % print_step == 0):
                    print("{0}/{1}".format(finishedResults, numberOfResults))
                    sys.stdout.flush()

        if (verbose == 1):
            bar.finish()

    def fit(self, trainingInput, trainingOutput, testingDataSequence, output_postprocessor = lambda x: x, printfreq=None, n_jobs=1, verbose=0):
        def enumerate_params():
            keys, values = zip(*self.param_grid.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))

        jobs = []
        for x in enumerate_params():
            jobs.append((x, self.fixed_params, trainingInput, trainingOutput, testingDataSequence, self.esnType))

        queue = Queue()
        pool = Pool(processes=n_jobs, initializer=GridSearchP._get_score_init, initargs=[queue,] )

        processProcessResultsThread = Process(target=GridSearchP.processThreadResults, args=("processProcessResultsThread", queue, n_jobs, len(jobs), verbose))

        processProcessResultsThread.start()
        results = pool.map(GridSearchP._get_score, jobs)
        pool.close()

        processProcessResultsThread.join()

        print(results)

        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[2]
        self._best_mse = res[0]

        return results
