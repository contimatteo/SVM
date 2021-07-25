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


def __accuracy(Y, Y_predicted):
    return Y[Y == Y_predicted].shape[0] / Y.shape[0]


def main():
    svm = None

    # dataset = DatasetGenerator.random()
    # dataset = DatasetGenerator.linear()
    # dataset = DatasetGenerator.non_linear1()
    dataset = DatasetGenerator.non_linear2()
    # dataset = DatasetGenerator.non_linear3()
    # dataset = DatasetGenerator.non_linear4()

    # use_cases = [
    #     {
    #         'dataset': DatasetGenerator.linear(),
    #         'kernels': ['linear', 'poly']
    #     },
    #     {
    #         'dataset': DatasetGenerator.non_linear1(),
    #         'kernels': ['linear', 'poly']
    #     },
    #     {
    #         'dataset': DatasetGenerator.non_linear2(),
    #         'kernels': ['poly', 'gaussian']
    #     },
    #     {
    #         'dataset': DatasetGenerator.non_linear3(),
    #         'kernels': ['poly', 'gaussian']
    #     },
    #     {
    #         'dataset': DatasetGenerator.random(),
    #         'kernels': ['linear', 'poly', 'gaussian']
    #     },
    # ]

    ### Dataset

    X, Y = dataset

    X_train, X_test, Y_train, Y_test = DatasetUtils.split(X, Y)

    # Plotter.data(X_train, Y_train)

    ### Training

    # svm = SVM(kernel='linear')
    # svm = SVM(kernel='linear', C=1.)
    # svm = SVM(kernel='poly')
    # svm = SVM(kernel='poly', C=1.)
    svm = SVM(kernel='poly', C=1., deg=3)

    svm.fit(X_train, Y_train)

    Y_train_predicted = svm.predict(X_train)

    train_accuracy = __accuracy(Y_train, Y_train_predicted)

    print(" $ [INFO] training-set accuracy = {:.2%}".format(train_accuracy))

    ### Testing

    Y_test_predicted = svm.predict(X_test)

    test_accuracy = __accuracy(Y_test, Y_test_predicted)

    print(" $ [INFO] testing-set accuracy = {:.2%}".format(test_accuracy))

    ### Plot

    # Plotter.svm(X, Y, svm)
    Plotter.regions(X, Y, svm)


###

if __name__ == "__main__":
    print()
    main()
    print()
