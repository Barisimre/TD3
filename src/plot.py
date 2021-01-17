import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


# Load data sources
data = np.load('results/keep/InvertedPendulum-v2_12:53:50.353312.npy')
# times = np.load('results/keep/b_times.npy')


data = data[:2800]

# data2 = data2[:2600]
# for a in times:

#     plt.axvline(x=a, ymin=0, ymax=40, color='tab:orange')

# Add timesteps
xi = list(range(len(data)))

# splp = make_interp_spline(xi, P, k=1)
# p_smooth = splp(xnew)

plt.plot(xi ,data, label="Buffer size = 512")
# plt.plot(xi ,data2, label="C = 0.01")

plt.legend()




plt.xlabel('Episodes Count')
plt.ylabel('Reward per Episode')

# TODO: label the data
# TODO: label the axis
plt.show()