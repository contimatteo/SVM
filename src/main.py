import numpy as np

from core.svm import SVM
import utils.configs as CONFIG
from utils.datasets import DatasetGenerator, DatasetUtils
from utils.plot import Plotter

###

np.random.seed(CONFIG.RANDOM_SEED)

###


def main():
    X, Y = dataset = DatasetGenerator.linear()

    X_train, X_test, Y_train, _ = DatasetUtils.split(X, Y)

    #

    svm = SVM()

    svm.fit(X_train, Y_train)

    _ = svm.predict(X_test)

    #

    Plotter.data_and_plane(dataset)

    #


###

if __name__ == "__main__":
    main()
