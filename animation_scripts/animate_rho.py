import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import numpy as np

mat = np.genfromtxt("results/N_TIME_STEPS.csv", delimiter=",")
N_TIME_STEPS = int(mat[0])
delta_t = mat[1]
SKIP_FRAMES = int(mat[2])

# Function to update the plot
def animate(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = pd.read_csv(file_name)
    ax.clear()
    ax.plot(data.iloc[:, 0], data.iloc[:, 1])
    ax.set_title(f"RHO: Time Step: {i}, Time: {i * delta_t:.4f} s, P: {i/N_TIME_STEPS * 100:.2f}%")
    ax.set_ylim(0, 5)

# Create a figure and axis object
fig, ax = plt.subplots()

# Create an animation
ani = animation.FuncAnimation(fig, animate, frames=range(0, int(N_TIME_STEPS), SKIP_FRAMES), interval=100)

plt.show()
