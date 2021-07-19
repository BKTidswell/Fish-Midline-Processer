import os, sys
import math
import matplotlib as mpl
from scipy.signal import argrelextrema, correlate,hilbert
import pandas as pd
import numpy as np
from fish_core_4P import *


mike_data = np.asarray(pd.read_csv("Fin_Angle.csv")["Fin_Angle"])

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

signal_1 = normalize_signal(mike_data) - max(normalize_signal(mike_data))/2
analytic_signal_1 = hilbert(signal_1)
instantaneous_phase_1 = np.unwrap(np.angle(analytic_signal_1))

fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(range(len(mike_data)), mike_data)

axs[1].plot(range(len(analytic_signal_1)), analytic_signal_1)

axs[2].plot(range(len(instantaneous_phase_1)), instantaneous_phase_1)

plt.show()
