# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


data = np.load("Latents/InvertedPendulum-v2_17:29:32.134576.npy")

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

xs = []
ys = []
zs = []

for i in data:
    xs.append(i[0])
    ys.append(i[2])
    zs.append(i[1])
ax.scatter(xs, ys, zs, marker='o')

plt.title('Center Title')


plt.show()

