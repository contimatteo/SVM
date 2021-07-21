import numpy as np

import utils.configs as CONFIG
from utils.datasets import Dataset
from utils.plot import Plotter

###

np.random.seed(CONFIG.RANDOM_SEED)

###

dataset = Dataset.linear()

# Plotter.dataset_scatter(dataset)

Plotter.data_and_plane(dataset)
