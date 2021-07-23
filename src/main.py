import numpy as np

import utils.configs as CONFIG

from core.svm import SVM
from utils.dataset import DatasetGenerator, DatasetUtils
from utils.plot import Plotter

###

np.random.seed(CONFIG.RANDOM_SEED)

###

###
### TODO: MISSING TASKS
###
### [x] add soft-margin formulation
### [ ] add multiple kernels support
### [ ] re-write all problem equation/formulas in the comments
### [ ] export the dataset used
### [ ] find a way for unifying plots
###

###


def main():
    # X, Y = DatasetGenerator.random()
    # X, Y = DatasetGenerator.linear()
    X, Y = DatasetGenerator.non_linear1()
    # X, Y = DatasetGenerator.non_linear2()
    # X, Y = DatasetGenerator.non_linear3()
    # X, Y = DatasetGenerator.non_linear4()

    X_train, _, Y_train, _ = DatasetUtils.split(X, Y)

    ###

    Plotter.data(X_train, Y_train)

    ###

    # svm = SVM()
    svm = SVM(C=1.)

    svm.fit(X_train, Y_train)

    Y_train_predicted = svm.predict(X_train)

    Plotter.svm(X_train, Y_train_predicted, svm)

    ###


###

if __name__ == "__main__":
    main()
