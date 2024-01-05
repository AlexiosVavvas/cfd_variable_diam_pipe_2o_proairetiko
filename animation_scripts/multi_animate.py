import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from myFunctions import *
from constantsToUse import *
# import os


# Remove any previous results
# os.system("rm animation_results/*.png")
# os.system("rm animation_results/*.gif")

# Load your data parameters
mat = np.genfromtxt("results/N_TIME_STEPS.csv", delimiter=",")
N_TIME_STEPS = int(mat[0])
delta_t = mat[1]
SKIP_FRAMES = int(mat[2])

# Colormap and normalization for U color plot
cmap = plt.cm.jet

# Define Upper Limits
rho_L, rho_H, u_L, u_H, p_L, p_H = resultingSolBoundaries("results/current_run_csvs", N_TIME_STEPS, SKIP_FRAMES)

norm_rho = mcolors.Normalize(vmin=rho_L, vmax=rho_H)
norm_u = mcolors.Normalize(vmin=u_L, vmax=u_H)
norm_p = mcolors.Normalize(vmin=p_L, vmax=p_H)

# Create a figure with four subplots
# ax1: P
# ax2: RHO
# ax3: U
# ax4: RHO color
# ax5: U color
# ax6: P color
# ax7: S(x)
# ax8: R(x)
fig, ((ax7, ax8), (ax2, ax4), (ax3, ax5), (ax1, ax6)) = plt.subplots(4, 2, figsize=(10, 8))

# ax7: S(x)
x = np.linspace(0, L, N_TIME_STEPS)
s_x = S(x, a, b, c, k)
ax7.plot(x, s_x, 'k-')
ax7.set_title("S(x)")
ax7.set_xlabel("x [m]")
ax7.set_ylabel("S [m^2]")
ax7.set_ylim(0, 1.1 * np.max(S(np.linspace(0, 1, N_TIME_STEPS), a, b, c, k)))
ax7.grid()

# ax8: R(x) = sqrt(S(x)/pi)
r_x = np.sqrt(s_x/np.pi)
ax8.plot(x, r_x, 'k-')
ax8.plot(x, -r_x, 'k-')
ax8.set_title("R(x)")
ax8.set_xlabel("x [m]")
ax8.set_ylabel("r [m]")
ax8.grid()

def animate_p(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = pd.read_csv(file_name)
    ax1.clear()
    ax1.plot(data.iloc[:, 0], data.iloc[:, 3])
    ax1.set_title(f"Pressure: Time: {i * delta_t:.4f} s ({i/N_TIME_STEPS * 100:.2f} %)")
    ax1.set_ylim(p_L*0.99, p_H*1.01)
    ax1.set_ylabel("P [Pa]")
    ax1.grid()

def animate_rho(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = pd.read_csv(file_name)
    ax2.clear()
    ax2.plot(data.iloc[:, 0], data.iloc[:, 1])
    ax2.set_title(f"Density: {i * delta_t:.4f} s ")
    ax2.set_ylim(rho_L*0.99, rho_H*1.01)
    ax2.set_ylabel("Rho [kg/m^3]")
    ax2.grid()

def animate_u(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = pd.read_csv(file_name)
    ax3.clear()
    ax3.plot(data.iloc[:, 0], data.iloc[:, 2])
    ax3.set_title(f"Velocity: Time: {i * delta_t:.4f} s ")
    ax3.set_ylim(u_L*0.99, u_H*1.01) 
    ax3.grid()
    ax3.set_ylabel("U [m/s]")
    ax3.set_yticks(np.arange(0, u_H*1.01, 100))

def animate_rho_color(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = np.genfromtxt(file_name, delimiter=",")
    ax4.clear()
    colors = cmap(norm_rho(data[:, 1]))
    ax4.vlines(data[:, 0], 0, 1, colors=colors, linewidth=5.5)
    ax4.set_title("Density")
    ax4.set_ylim(0, 1)
    ax4.grid()
    ax4.set_yticks([])


def animate_u_color(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = np.genfromtxt(file_name, delimiter=",")
    ax5.clear()
    colors = cmap(norm_u(data[:, 2]))
    ax5.vlines(data[:, 0], 0, 1, colors=colors, linewidth=5.5)
    ax5.set_ylim(0, 1)
    ax5.set_title("Velocity")
    ax5.grid()
    ax5.set_yticks([])

def animate_p_color(i):
    file_name = f"results/current_run_csvs/U_P_timestep_{i}.csv"
    data = np.genfromtxt(file_name, delimiter=",")
    ax6.clear()
    colors = cmap(norm_p(data[:, 3]))
    ax6.vlines(data[:, 0], 0, 1, colors=colors, linewidth=5.5)
    ax6.set_ylim(0, 1)
    ax6.set_title("Pressure")
    ax6.grid()
    ax6.set_yticks([])


def update_all(i):
    try:
        print(f"Saving frame {int(i/SKIP_FRAMES)} / {int(N_TIME_STEPS/SKIP_FRAMES)} to disk")
        animate_p(i)
        animate_rho(i)
        animate_u(i)
        animate_rho_color(i)
        animate_u_color(i)
        animate_p_color(i)

        fig.savefig(f"results/animation/animation_frame_{i}.png")
    
    except Exception as e:
        print(f"Error saving frame {i}: {e}")

# Create animations for each subplot
ANIM_INTERVAL = 10
ani = animation.FuncAnimation(fig, update_all, frames=range(0, N_TIME_STEPS, SKIP_FRAMES), interval=ANIM_INTERVAL, repeat=False)

# Adjust layout
plt.title("Simulation Results")
plt.tight_layout(pad=3)
plt.show()

print("Animation Frames Extracted")