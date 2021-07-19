from fish_core import *

t = np.linspace(0,20,60*20)

y1 = np.sin(t)
y2 = np.sin(1.415*t+np.pi/2)

h1 = hilbert(y1)
h2 = hilbert(y2)

a1 = np.unwrap(np.angle(h1),0.5)
a2 = np.unwrap(np.angle(h2),0.5)

d = a2 - a1

#d2 = get_slope(range(len(d)),d)

d2 = np.gradient(d)*60

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(t, y1, "r")
axs[0].plot(t, y2, "b")
axs[1].plot(t, a1, "r")
axs[1].plot(t, a2, "b")
axs[2].plot(t, d, "m")
axs[3].plot(t, d2, "g")
axs[3].set_ylim(-1,5)
plt.show()