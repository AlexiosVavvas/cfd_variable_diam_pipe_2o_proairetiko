import numpy as np
from myFunctions import *
from constantsToUse import *

# Create folders if they don't exist
checkOrCreateFolderResults("results")
checkOrCreateFolderResults("results/current_run_csvs")
checkOrCreateFolderResults("results/animation")

# Remove any previous results
os.system("rm results/current_run_csvs/*.csv")

# --------------------------------
N = 70                     # Number of grid points
delta_t = 2e-6             # Time step
SKIP_FRAMES = 30           # Number of frames to skip when saving animation
SKIP_PRINTS = 10           # Number of frames to skip when saving animation
CONV_E = 3e-2              # Convergence criteria
N_TIME_STEPS_MAX = 50000   # Number of time steps to solve./

# Plot S(x), R(x) and Save them
plotAndSaveS(a, b, c, k, L)

# Initial Conditions
P0 = 6.4e4           # [Pa] - Aerofilakio
T0 = 275             # [K]  - Aerofilakio
Pout = 2.5e4         # [Pa] - Outlet
Patm = Pout          # [Pa] - Atmospheric
Tatm = 273.15        # [K]  - Atmospheric
# --------------------------------

# Grid Definition
delta_x = L / (N - 1)

# General Initialization
U_P = np.zeros((3, N))
U_C = np.zeros((3, N))
U_u_old = np.zeros((1, N))   # u at previous time step

U_P[0, :] = Patm/ (287 * Tatm)  # Initial Density - Ideal Gas Law
U_P[1, :] = 0                   # Initial Velocity
U_P[2, :] = Patm                # Initial Pressure

# Boundary Conditions
U_P[0, 0] = P0 / (287 * T0)  # rho 0
U_P[1, 0] = U_P[1, 1]        # u 0
U_P[2, 0] = P0               # p 0

U_P[0, N-1] = U_P[0, N-2]    # rho Outlet
U_P[1, N-1] = U_P[1, N-2]    # u Outlet
U_P[2, N-1] = Pout           # p Outlet

# Time Loop
t = 0
i = 0

# Save initial state, initial conditions and shape constants
np.savetxt(f"results/current_run_csvs/U_P_init.csv", np.hstack((np.linspace(0, 1, N).reshape(N, 1), np.transpose(U_P))), delimiter=",")
np.savetxt("results/INIT_CONDITIONS.csv", np.array([["P0", "T0", "Pout"], [P0, T0, Pout]]), delimiter=",", fmt="%s")
np.savetxt("results/SHAPE_CONSTS.csv", np.array([["k", "a", "b", "c", "L"], [k, a, b, c, L]]), delimiter=",", fmt="%s")

# Solving in Time
conv_error = 1
np.savetxt(f"results/N_TIME_STEPS.csv", np.array([N_TIME_STEPS_MAX, delta_t, SKIP_FRAMES]), delimiter=",")

print("Solving in Time...")
while i < N_TIME_STEPS_MAX and (conv_error > CONV_E or i % SKIP_FRAMES != 0):

    # Print progress
    if i % SKIP_PRINTS == 0:
        print(f"Solving Time Step {i} of {N_TIME_STEPS_MAX} at t = {t:.6f} \t {i/N_TIME_STEPS_MAX * 100:.2f}% \t \
            conv_error = {conv_error:.2e}/{CONV_E:.1e} -> {(CONV_E)/conv_error * 100:.2f}% ")

    # Save old Velocity Profile
    U_u_old = np.copy(U_P[1, :])

    # Boundary Conditions
    # Inlet
    U_P[0, 0] = P0 / (287 * T0)  # rho 0
    U_P[1, 0] = U_P[1, 1]        # u 0
    U_P[2, 0] = P0               # p 0
    # Outlet
    U_P[0, N-1] = U_P[0, N-2]    # rho Outlet
    U_P[1, N-1] = U_P[1, N-2]    # u Outlet
    U_P[2, N-1] = Pout           # p Outlet

    # Write U_P to file
    if i % SKIP_FRAMES == 0:
        np.savetxt(f"results/current_run_csvs/U_P_timestep_{i}.csv", np.hstack((np.linspace(0, 1, N).reshape(N, 1), np.transpose(U_P))), delimiter=",")
    
    # Calculate U_C
    U_C = upToUc(U_P)

    # Update U_C using RK
    U_C = customRKstep(U_C, delta_x, delta_t, a, b, c, k)

    # Convert U_C to U_P
    U_P = ucToUp(U_C)

    # Calc current conv_error
    conv_error = np.max(np.abs(U_P[1, :] - U_u_old))

    # Update time and iteration
    t = t + delta_t
    i = i + 1

np.savetxt(f"results/N_TIME_STEPS.csv", np.array([i, delta_t, SKIP_FRAMES]), delimiter=",")

if i == N_TIME_STEPS_MAX:
    print("Maximum number of time steps reached!")
else:
    print("Convergence reached!")
    
print(f"Finished at t = {t:.4f}, iter = {i}/{N_TIME_STEPS_MAX}")
