import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data = np.loadtxt('./res.dat')

r = np.linspace(0, 200, data.shape[1])

fig = plt.figure()
axis = fig.add_subplot(111)
axis.set_yscale('log')

line = axis.plot(r, data[0])[0]

def update_line(num):
    line.set_data(r, data[num])
    return (line, 't')

ani = animation.FuncAnimation(fig, update_line, frames=data.shape[0], interval=200, repeat=True)

plt.show()
