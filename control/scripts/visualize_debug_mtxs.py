import os
import matplotlib.pyplot as plt
from scipy.io import mmread
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load debug_fatrop_expected.mtx and debug_fatrop_actual.mtx (matrix market)
expected = mmread('debug_fatrop_expected.mtx').toarray()
actual = mmread('debug_fatrop_actual.mtx').toarray()

# overlay spy plots (circles for expected, dots for actual)
plt.figure()
plt.spy(expected, marker='o', markersize=5, label='Expected', color='blue')
plt.spy(actual, marker='.', markersize=5, label='Actual', color='red')
plt.show()