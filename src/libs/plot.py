import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from libs.svm import SVM

###


class Plotter:
    @staticmethod
    def __colors_from_classes(Y):
        def color_from_class(y):
            return 'C1' if y == 1 else 'C2'

        return np.array([color_from_class(y) for y in Y])

    @staticmethod
    def __color_map(Y):
        classes_unique_values = np.unique(Y)
        # predefined_colors = ['red', 'blue', 'cyan']
        predefined_colors = ['red', 'blue']
        return ListedColormap(predefined_colors[:classes_unique_values.shape[0]])

    ###

    @staticmethod
    def regions(X, Y, svm: SVM):
        x_margin = np.arange(X[:, 0].min() - 2, X[:, 0].max() + 2, .05)
        y_margin = np.arange(X[:, 1].min() - 2, X[:, 1].max() + 2, .05)
        xx, yy = np.meshgrid(x_margin, y_margin)

        generated_points_for_plotting = np.array([xx.flatten(), yy.flatten()]).T
        classes = svm.predict(generated_points_for_plotting).reshape(xx.shape)
        cmap = Plotter.__color_map(Y)

        plt.contourf(xx, yy, classes, alpha=0.4, cmap=cmap)
        plt.axline(x_margin, y_margin)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        ### markers: 's', 'x', 'o', '^', 'v'
        plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='black', s=25, marker='o')
        plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='black', s=25, marker='x')

        plt.axis("tight")
        plt.show()

    @staticmethod
    def contours(X, Y, svm: SVM):
        ax = plt.gca()
        cmap = Plotter.__color_map(Y)

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_min -= 1  # (x_min / 5)
        y_min -= 1  # (y_min / 5)
        x_max += 1  # (x_max / 5)
        y_max += 1  # (y_max / 5)

        x_margin = np.arange(x_min, x_max, .05)
        y_margin = np.arange(y_min, y_max, .05)
        xx, yy = np.meshgrid(x_margin, y_margin)
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        projections = svm.project(xy).reshape(xx.shape)

        ax.contour(
            xx,
            yy,
            projections,
            colors='k',
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=['--', '-', '--']
        )

        plt.scatter(X[:, 0], X[:, 1], c=Y, s=15, marker='o', cmap=cmap)

        ax.scatter(
            svm.support_vectors[:, 0],
            svm.support_vectors[:, 1],
            s=150,
            edgecolors='k',
            linewidths=1,
            facecolors='none'
        )

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))

        plt.axis("tight")
        plt.show()

    @staticmethod
    def advanced(fig, axs, X_train, Y_train, X_test, Y_test, svm: SVM):
        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((Y_train, Y_test))

        cmap = Plotter.__color_map(Y)

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_min -= 1  # (x_min / 5)
        y_min -= 1  # (y_min / 5)
        x_max += 1  # (x_max / 5)
        y_max += 1  # (y_max / 5)

        x_margin = np.arange(x_min, x_max, .05)
        y_margin = np.arange(y_min, y_max, .05)
        xx, yy = np.meshgrid(x_margin, y_margin)
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        projections = svm.project(xy).reshape(xx.shape)

        axs.contour(
            xx,
            yy,
            projections,
            colors='k',
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=['--', '-', '--']
        )

        # axs.scatter(X[:, 0], X[:, 1], c=Y, s=15, marker='o', cmap=cmap)
        axs.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, marker='o', cmap=cmap)
        axs.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=20, marker='x', cmap=cmap)

        axs.scatter(
            svm.support_vectors[:, 0],
            svm.support_vectors[:, 1],
            s=130,
            edgecolors='k',
            linewidths=1,
            facecolors='none'
        )

        axs.set_xlim((x_min, x_max))
        axs.set_ylim((y_min, y_max))

    @staticmethod
    def data(fig, axs, X, Y):
        cmap = Plotter.__color_map(Y)

        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_min -= 1  # (x_min / 5)
        y_min -= 1  # (y_min / 5)
        x_max += 1  # (x_max / 5)
        y_max += 1  # (y_max / 5)

        axs.scatter(X[:, 0], X[:, 1], c=Y, s=15, marker='o', cmap=cmap)

        axs.set_xlim((x_min, x_max))
        axs.set_ylim((y_min, y_max))
