import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline





# Load data sources
data = np.load('results/keep/kl/0001-2.npy')
data2 = np.load('results/keep/kl/001-2.npy')


data = data[:2600]

data2 = data2[:2600]


# Add timesteps
xi = list(range(len(data)))

# splp = make_interp_spline(xi, P, k=1)
# p_smooth = splp(xnew)

plt.plot(xi ,data, label="C = 0.001")
plt.plot(xi ,data2, label="C = 0.01")

plt.legend()




plt.xlabel('Episodes Count')
plt.ylabel('Reward per Episode')

# TODO: label the data
# TODO: label the axis
plt.show()