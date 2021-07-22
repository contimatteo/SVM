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

    @staticmethod
    def svm(dataset, svm):
        X, Y = dataset
        support_vectors = svm.support_vectors

        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=50, c="g")

        # # Plot the lines

        # # w.x + b = 0
        # a0 = -4
        # a1 = f(a0, weight, bias)
        # b0 = 4
        # b1 = f(b0, weight, bias)
        # plt.plot([a0, b0], [a1, b1], "k")

        # # w.x + b = 1
        # a0 = -4
        # a1 = f(a0, weight, bias, 1)
        # b0 = 4
        # b1 = f(b0, weight, bias, 1)
        # plt.plot([a0, b0], [a1, b1], "k--")

        # # w.x + b = -1
        # a0 = -4
        # a1 = f(a0, weight, bias, -1)
        # b0 = 4
        # b1 = f(b0, weight, bias, -1)
        # plt.plot([a0, b0], [a1, b1], "k--")

        # plt.axis("tight")

        plt.show()
