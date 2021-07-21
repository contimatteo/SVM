import matplotlib.pyplot as plt
import numpy as np

###


class Plotter:
    @staticmethod
    def __colors_from_classes(Y):
        def color_from_class(y):
            return 'C1' if y == 1 else 'C2'

        return np.array([color_from_class(y) for y in Y])

    @staticmethod
    def data(dataset):
        X, Y = dataset

        colors = Plotter.__colors_from_classes(Y)

        plt.scatter(X[:, 0], X[:, 1], color=colors)
        plt.show()

    @staticmethod
    def data_and_plane(dataset, plane=([.1, .2], [.8, .7])):
        X, Y = dataset
        plane_x1y1, plane_x2y2 = plane

        colors = Plotter.__colors_from_classes(Y)

        plt.scatter(X[:, 0], X[:, 1], color=colors)
        plt.axline(plane_x1y1, plane_x2y2, color='C4')
        plt.show()
