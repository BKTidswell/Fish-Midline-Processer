
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random
import math
import os

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

demo_fish_1_x = np.arange(0,300,1)
demo_fish_1_y = np.sin(demo_fish_1_x/5)*10

demo_fish_2_x = np.arange(0,300,1)
demo_fish_2_y = np.sin(demo_fish_1_x/7)*10

demo_fish_3_x = np.arange(300,0,-1)
demo_fish_3_y = np.sin(demo_fish_1_x/5)*10+10

demo_fish_4_x = np.sin(demo_fish_1_x/5)*10+150
demo_fish_4_y = np.arange(0,300,1)-150

demo_fish_5_x = np.arange(0,300,1)
demo_fish_5_y = np.sin(demo_fish_1_x**1.4/50)*10

demo_fish_6_circle = np.linspace(0, 2*np.pi, num=300)
demo_fish_6_x = 30*np.cos(demo_fish_6_circle)+150
demo_fish_6_y = 30*np.sin(demo_fish_6_circle)


x = np.arange(10)
y = np.arange(10)

alphas = np.linspace(0.1, 1, 10)

plt.scatter(x, y, alpha=alphas)
plt.show()