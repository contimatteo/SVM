import numpy as np

from core.svm import SVM
import utils.configs as CONFIG
from utils.datasets import DatasetGenerator, DatasetUtils
from utils.plot import Plotter

###

np.random.seed(CONFIG.RANDOM_SEED)

###


def main():
    X, Y = DatasetGenerator.linear()

    X_train, _, Y_train, _ = DatasetUtils.split(X, Y)

    ###

    # Plotter.data(X_train, Y_train)

    ###

    svm = SVM()

    svm.fit(X_train, Y_train)

    Y_train_predicted = svm.predict(X_train)

    Plotter.svm(X_train, Y_train_predicted, svm)

    ###


###

if __name__ == "__main__":
    main()
