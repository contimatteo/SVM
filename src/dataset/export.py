import numpy as np

from utils import DatasetGenerator

###


def __save(dataset, fileName):
    X, Y = dataset
    XY = np.column_stack((X, Y))
    np.savetxt(f"{fileName}.txt", XY, delimiter=",")


if __name__ == "__main__":

    __save(DatasetGenerator.linear(), 'linear')

    __save(DatasetGenerator.non_linear1(), 'non_linear1')
    
    __save(DatasetGenerator.non_linear2(), 'non_linear2')

    __save(DatasetGenerator.non_linear3(), 'non_linear3')

    __save(DatasetGenerator.non_linear4(), 'non_linear4')

    __save(DatasetGenerator.random(), 'random')
