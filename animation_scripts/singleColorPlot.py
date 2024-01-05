import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation

cmap = plt.cm.jet  # Choose the colormap

mat = np.genfromtxt("../results/N_TIME_STEPS.csv", delimiter=",")
N_TIME_STEPS = int(mat[0])
delta_t = mat[1]
SKIP_FRAMES = int(mat[2])

norm = mcolors.Normalize(vmin=0, vmax=400)

def animate(i):
    data = np.genfromtxt(f"../results/current_run_csvs/U_P_timestep_{i}.csv", delimiter=",")
    ax.clear()
    colors = cmap(norm(data[:, 2]))
    ax.vlines(data[:, 0], 0, 1, colors=colors, linewidth=5.5)
    ax.set_ylim(0, 1)

# Create a figure and axis object
fig, ax = plt.subplots()
plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'))

# Create an animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, int(N_TIME_STEPS), SKIP_FRAMES), interval=100)

plt.show()

