import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

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

    @staticmethod
    def regions(X, Y, classifier):
        ### markers: 's', 'x', 'o', '^', 'v'

        predefined_colors = ['red', 'blue', 'cyan']
        classes_unique_values = np.unique(Y)
        region_colormap = ListedColormap(predefined_colors[:classes_unique_values.shape[0]])

        x_max, y_max  = X[:, 0].max(), X[:, 1].max()
        x_min, y_min  = X[:, 0].min(), X[:, 1].min()
        x_min -= 2
        y_min -= 2
        x_max += 2
        y_max += 2

        x_margin_length = np.arange(x_min, x_max, .025)
        y_margin_length =  np.arange(y_min, y_max, .025)
        xx, yy = np.meshgrid(x_margin_length, y_margin_length)

        generated_points_for_plotting = np.array([xx.flatten(), yy.flatten()]).T
        classes = classifier.predict(generated_points_for_plotting).reshape(xx.shape)

        plt.contourf(xx, yy, classes, alpha=0.4, cmap=region_colormap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='black', s=25, marker='o')
        plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='black', s=25, marker='x')

        plt.axis("tight")
        plt.show()
