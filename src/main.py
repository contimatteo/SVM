import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np

from libs.svm import SVM
from libs.dataset import DatasetGenerator, DatasetUtils
from libs.plot import Plotter

###

warnings.filterwarnings("ignore")

np.random.seed(666)

###

###
### TODO: MISSING TASKS
###
### [ ] re-write all problem equation/formulas in the comments
### [ ] export the dataset used
###

###

datasets = [
    DatasetGenerator.linear(),
    DatasetGenerator.non_linear1(),
    DatasetGenerator.non_linear2(),
    DatasetGenerator.non_linear3(),
    DatasetGenerator.non_linear4(),
    DatasetGenerator.random()
]

###


def __accuracy(Y, Y_predicted):
    return Y[Y == Y_predicted].shape[0] / Y.shape[0]


def __analyze_svm(fig, axs, config, dataset):
    ### Dataset

    X, Y = dataset

    X_train, X_test, Y_train, Y_test = DatasetUtils.split(X, Y)

    ### Training

    svm = SVM(kernel=config['kernel'], deg=config['deg'], C=config['C'])

    svm.fit(X_train, Y_train)

    Y_train_predicted = svm.predict(X_train)

    train_accuracy = __accuracy(Y_train, Y_train_predicted)

    ### Testing

    Y_test_predicted = svm.predict(X_test)

    test_accuracy = __accuracy(Y_test, Y_test_predicted)

    ### Stats

    print()
    print(" > {}: correct predictions".format(config['title']))
    print(" >  - training-set  =  {:.1%}".format(train_accuracy))
    print(" >  -  testing-set  =  {:.1%}".format(test_accuracy))

    ### Plot

    Plotter.advanced(fig, axs, X_train, Y_train, X_test, Y_test, svm)


def main(dataset_index):

    dataset = datasets[dataset_index]

    fig, axs = plt.subplots(2, 2)

    ### Linear (hard)

    title = 'Linear (hard)'

    __analyze_svm(
        fig, axs[0, 0], {
            'title': title,
            'kernel': 'linear',
            'C': None,
            'deg': None
        }, dataset
    )

    axs[0, 0].set_title(title)

    ### Linear (soft)

    title = 'Linear (soft, C=.5)'

    __analyze_svm(
        fig, axs[0, 1], {
            'title': title,
            'kernel': 'linear',
            'C': .5,
            'deg': None
        }, dataset
    )

    axs[0, 1].set_title(title)

    ### Poly (soft)

    title = 'Poly-3 (soft, C=.5)'

    __analyze_svm(fig, axs[1, 0], {'title': title, 'kernel': 'poly', 'C': .5, 'deg': 3}, dataset)

    axs[1, 0].set_title(title)

    ### Poly (soft)

    title = 'Poly-5 (hard)'

    __analyze_svm(fig, axs[1, 1], {'title': title, 'kernel': 'poly', 'C': None, 'deg': 5}, dataset)

    axs[1, 1].set_title(title)

    ###

    plt.axis("tight")
    plt.show()


###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matteo Conti - SVM")

    parser.add_argument(
        "--dataset",
        help="dataset index to use.",
        choices=['1', '2', '3', '4', '5', '6'],
        default=1,
        required=False
    )

    args = parser.parse_args()

    print()
    print(args)

    main(dataset_index=int(args.dataset) - 1)
    print()
