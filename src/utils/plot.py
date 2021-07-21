import matplotlib.pyplot as plt
import numpy as np

###


class Plotter:
    @staticmethod
    def data(dataset):
        X1, _, X2, _ = dataset

        plt.plot(X1[:, 0], X1[:, 1], 'ro', alpha=0.75)
        plt.plot(X2[:, 0], X2[:, 1], 'bo', alpha=0.75)

        plt.show()

    @staticmethod
    def data_and_plane(dataset, plane=([.1, .2], [.8, .7])):
        X1, _, X2, _ = dataset

        plane_x1y1, plane_x2y2 = plane

        plt.plot(X1[:, 0], X1[:, 1], 'o', color='C1')
        plt.plot(X2[:, 0], X2[:, 1], 'o', color='C2')

        plt.axline(plane_x1y1, plane_x2y2, color='C4')

        plt.show()
