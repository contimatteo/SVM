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
    def plot_decision_regions(X, y, Xtest, Ytest, classifier, kernel='<MISSING>', resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_train_min, x1_train_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_train_min, x2_train_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x1_test_min, x1_test_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
        x2_test_min, x2_test_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1

        x1_min = min(x1_train_min, x1_test_min)
        x1_max = max(x1_train_max, x1_test_max)
        x2_min = min(x2_train_min, x2_test_min)
        x2_max = max(x2_train_max, x2_test_max)
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
            plt.scatter(x=Xtest[Ytest == cl, 0], y=Xtest[Ytest == cl, 1], alpha=0.8, c=cmap(idx),
                        marker=markers[3], label=cl)

        plt.axis("tight")
        if kernel == "linear":
            plt.title("Linear Kernel")
        elif kernel == "non_linear":
            plt.title("Non Linear Kernel")
        plt.show()
