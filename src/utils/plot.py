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
    def data(X, Y):
        colors = Plotter.__colors_from_classes(Y)

        plt.scatter(X[:, 0], X[:, 1], color=colors)
        plt.show()

    # @staticmethod
    # def data_and_plane(dataset, plane=([.1, .2], [.8, .7])):
    #     X, Y = dataset
    #     plane_x1y1, plane_x2y2 = plane

    #     colors = Plotter.__colors_from_classes(Y)

    #     plt.scatter(X[:, 0], X[:, 1], color=colors)
    #     plt.axline(plane_x1y1, plane_x2y2, color='C4')
    #     plt.show()

    @staticmethod
    def svm(X, Y, svm):
        colors = Plotter.__colors_from_classes(Y)
        support_vectors = svm.support_vectors

        x1_limit = np.min(support_vectors[:, 0])
        x2_limit = np.max(support_vectors[:, 0])

        margin_xy1 = [x1_limit, svm.hyperplane_equation(x1_limit, 0)]
        margin_xy2 = [x2_limit, svm.hyperplane_equation(x2_limit, 0)]
        # up_margin_xy1 = [x1_limit, svm.hyperplane_equation(x1_limit, 1)]
        # up_margin_xy2 = [x2_limit, svm.hyperplane_equation(x2_limit, 1)]
        # low_margin_xy1 = [x1_limit, svm.hyperplane_equation(x1_limit, -1)]
        # low_margin_xy2 = [x2_limit, svm.hyperplane_equation(x2_limit, -1)]

        plt.scatter(X[:, 0], X[:, 1], color=colors, s=15)
        # plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, c="C4")
        plt.axline(margin_xy1, margin_xy2)
        # plt.axline(up_margin_xy1, up_margin_xy2)
        # plt.axline(low_margin_xy1, low_margin_xy2)

        plt.axis("tight")
        plt.show()
