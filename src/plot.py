import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


font = {'family' : 'normal',
        'size'   : 16}

plt.rc('font', **font)


# Load data sources
data = np.load('results/keep/delay/a.npy')
times1 = np.load('results/keep/delay/a_td3.npy')
# times1 = np.load('results/incoming/a_vae.npy')
# times2 = np.load('results/incoming/a_td3.npy')


for a in times1:

    plt.axvline(x=a, ymin=0, ymax=2, color='tab:orange')

# for b in times2:

#     plt.axvline(x=b, ymin=0, ymax=2, color='tab:orange')


# Add timesteps
xi = list(range(len(data)))


plt.plot(xi ,data, label="Training with TD3 freezes")
# plt.plot(xi ,data2, label="No Experience Reaply")

plt.legend()




plt.xlabel('Episodes Count')
plt.ylabel('Reward per Episode')

# TODO: label the data
# TODO: label the axis
plt.show()