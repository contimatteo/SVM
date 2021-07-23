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
    # dataset = DatasetGenerator.random()
    # dataset = DatasetGenerator.linear()
    dataset = DatasetGenerator.non_linear1()
    # dataset = DatasetGenerator.non_linear2()
    # dataset = DatasetGenerator.non_linear3()
    # dataset = DatasetGenerator.non_linear4()

    ###

    X, Y = dataset

    X_train, _, Y_train, _ = DatasetUtils.split(X, Y)

    # Plotter.data(X_train, Y_train)

    ###

    # svm = SVM(kernel='linear')
    svm = SVM(kernel='linear', C=1.)
    # svm = SVM(kernel='poly')
    # svm = SVM(kernel='poly', C=1.)
    # svm = SVM(kernel='sigmoid')
    # svm = SVM(kernel='sigmoid', C=1.)

    svm.fit(X_train, Y_train)

    Y_train_predicted = svm.predict(X_train)

    Plotter.svm(X_train, Y_train_predicted, svm)


###

if __name__ == "__main__":
    main()
