import numpy as np
from pdmproject.environment import generate_environment
import matplotlib

matplotlib.use("TkAgg")
np.random.seed(42)
# np.random.seed(99000)

generate_environment(inspect=True)