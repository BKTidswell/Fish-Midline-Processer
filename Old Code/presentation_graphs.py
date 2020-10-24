import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema, correlate, hilbert, square

def get_slope(x,y):
	slope_array = np.zeros(len(x))

	for i in range(2,len(x)-2):
		#This gets the slope from the points surrounding i so that the signal is less noisy
		slope = (((y[i+1]-y[i-1]) / (x[i+1]-x[i-1])) + ((y[i+2]-y[i-2]) / (x[i+2]-x[i-2]))) / 2
		slope_array[i] = slope

	return(slope_array[2:-2])

t = np.linspace(0,4*np.pi,1000)
f1 = np.sin(t)
f2 = np.cos(t)# + 2 * np.random.random((1000,)) - 1

analytic_signal_f1 = hilbert(f1)
instantaneous_phase_f1 = np.unwrap(np.angle(analytic_signal_f1))

analytic_signal_f2 = hilbert(f2)
instantaneous_phase_f2 = np.unwrap(np.angle(analytic_signal_f2))

slope = get_slope(instantaneous_phase_f1,instantaneous_phase_f2)


fig, axs = plt.subplots(4)
axs[0].plot(t, f1)
axs[0].plot(t, f2)
axs[1].plot(t, instantaneous_phase_f1)
axs[1].plot(t, instantaneous_phase_f2)
axs[2].plot(instantaneous_phase_f1,instantaneous_phase_f2)
axs[3].plot(t[2:-2], slope)
#axs[3].axhline(y=1, c="black")
axs[3].set_xlabel("time in seconds")
axs[3].set_ylim(-2, 2)


plt.show()