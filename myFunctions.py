import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
gamma1 = gamma - 1

'''
Equation:

 dUi      - 1
---- =  -------- * R(U_i) + Q_i
 dt      DeltaX


Uc = [rho, rho*u, rho*E]
Up = [rho, u, p]

The following functions need as input:
    - roeTilda      : uL - uR (uPrime)
    - solveRiemann  : uL - uR (uPrime)
    - F             : Up - 3x1
    - calcR         : Up - 3xN
    - calcQ         : Up - 3xN

'''

# Surface Area
def S(x, a, b, c, k):
    x_ = x - c
    return k + a * x_**2 * (x_**2 - b**2)
def dSdX(x, a, b, c, k):
    x_ = x - c
    return 4 * a * x_**3 - 2 * a * b**2 * x_
def plotAndSaveS(a, b, c, k, L):
    x = np.linspace(0, L, 1000)
    y = S(x, a, b, c, k)
    
    plt.plot(x, y, 'k-')
    plt.title("Surface Area S(x)")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("S [m^2]")
    plt.ylim(0, 1.1 * np.max(y))
    plt.savefig("results/S(x).png")
    plt.close()

    r = np.sqrt(y/np.pi)
    plt.plot(x, r, 'k-')
    plt.plot(x, -r, 'k-')
    plt.title("Radius r(x)")
    plt.xlabel("x [m]")
    plt.ylabel("r [m]")
    plt.grid()
    plt.savefig("results/r(x).png")
    plt.close()


# Convert from Uc to Up and vice versa
def ucToUp(uc):
    uc = np.atleast_2d(uc).T if uc.ndim == 1 else uc
    up1 = uc[0, :]
    up2 = uc[1, :]/uc[0, :]
    up3 = gamma1 * (uc[2, :] - 0.5 * uc[1, :]**2 / uc[0, :])
    return np.vstack((up1, up2, up3)).squeeze()

def upToUc(up):
    up = np.atleast_2d(up).T if up.ndim == 1 else up
    uc1 = up[0, :]
    uc2 = up[1, :] * up[0, :]
    uc3 = up[2, :] / gamma1 + 0.5 * up[0, :] * up[1, :]**2
    return np.vstack((uc1, uc2, uc3)).squeeze()

# Input upL, upR
# Returns [ρ_, u_, H_, c_]
def roeTilda(uL, uR):
    s_rhoL = np.sqrt(uL[0])
    s_rhoR = np.sqrt(uR[0])

    rho_ = s_rhoL * s_rhoR
    u_ = (s_rhoL * uL[1] + s_rhoR * uR[1]) / (s_rhoL + s_rhoR)

    hL = (upToUc(uL)[2] + uL[2])/uL[0]  # to check if it is correct
    hR = (upToUc(uR)[2] + uR[2])/uR[0]  # to check if it is correct
    H_ = (s_rhoL * hL + s_rhoR * hR) / (s_rhoL + s_rhoR)

    c_ = np.sqrt((gamma - 1) * (H_ - 0.5 * u_**2))

    return np.array([rho_, u_, H_, c_])

# Flux
def F(uP):
    F1 = uP[0] * uP[1]
    F2 = uP[0] * np.abs(uP[1]) * uP[1] + uP[2]
    # F2 = uP[0] * uP[1]**2 + uP[2]
    F3 = (upToUc(uP)[2] + uP[2]) * uP[1]
    return np.array([F1, F2, F3])

# Solve Riemann problem - |A|
# Input uL, uR
def solveRiemann(uL, uR):
    rho_, u_, H_, c_ = roeTilda(uL, uR)

    # Calculate eigenvalues
    lam1 = u_       # to check if it is correct
    lam2 = u_ + c_
    lam3 = u_ - c_
    Lamda = np.diag(np.abs([lam1, lam2, lam3]))

    # Calculate Right eigenvectors 3x3
    R = np.zeros((3,3))
    R[0, 0] = 1
    R[0, 1] = + 0.5 * rho_ / c_
    R[0, 2] = - 0.5 * rho_ / c_
    R[1, 0] = u_
    R[1, 1] = + 0.5 * (u_ + c_) * rho_ / c_
    R[1, 2] = - 0.5 * (u_ - c_) * rho_ / c_
    R[2, 0] = 0.5 * u_**2
    R[2, 1] = + (0.5 * u_**2 + u_*c_ + c_**2 / gamma1) * 0.5 * rho_ / c_
    R[2, 2] = - (0.5 * u_**2 - u_*c_ + c_**2 / gamma1) * 0.5 * rho_ / c_

    # Calculate Left eigenvectors 3x3
    L = np.zeros((3,3))
    L[0, 0] = 1 - 0.5 * (gamma - 1) * u_**2 / c_**2
    L[0, 1] = gamma1 * u_ / c_**2
    L[0, 2] = - gamma1 / c_**2
    L[1, 0] = + (0.5 * gamma1 * u_**2 - u_*c_) * 1 / (rho_ * c_)
    L[1, 1] = - (gamma1 * u_ - c_) / (rho_ * c_)
    L[1, 2] = gamma1 / (rho_ * c_)
    L[2, 0] = - (0.5 * gamma1 * u_**2 + u_*c_) * 1 / (rho_ * c_)
    L[2, 1] = + (gamma1 * u_ + c_) / (rho_ * c_)
    L[2, 2] = - gamma1 / (rho_ * c_)


    # Calculate resulting |A| -> R * Lamda * L
    A = np.matmul(np.matmul(R, Lamda), L)

    return A


# R = Fe - Fw
# U_P: 3xN 
def calcR(i, U_P):
    uP = U_P[:, i]
    uE = U_P[:, i+1]
    uW = U_P[:, i-1]

    # Calculate As
    A_E = solveRiemann(uP, uE)
    A_W = solveRiemann(uW, uP)

    # Calculate Flux
    Fe = 0.5 * (F(uP) + F(uE)) + 0.5 * np.matmul(A_E, (uP - uE))
    Fw = 0.5 * (F(uW) + F(uP)) + 0.5 * np.matmul(A_W, (uW - uP))

    # print(f"iter_{i}: R = {Fe - Fw}")

    return Fe - Fw


# External Q
# U_P = 3xN
def calcQ(i, U_P, delta_x, a, b, c, k):

    p = U_P[2, i]
    S_i = S(i*delta_x, a, b, c, k)
    dSdX_i = dSdX(i*delta_x, a, b, c, k)

    Q = np.zeros(3)
    Q[1] = p / S_i * dSdX_i

    # print(f"iter_{i}: Q2 = {Q[1]}")
    return Q
    
# U_C : 3xN
def customRKstep(U_C, delta_x, delta_t, a, b, c, k):

    # Define rk constants
    rk = [0.1084, 0.2602, 0.5052, 1]
    
    N = U_C.shape[1]
    U_ = np.copy(U_C)    # Resulting U_C

    # For every RK constant
    for j in range(len(rk)):
        U_P_ = ucToUp(U_)

        # For every internal node
        for i in range(1, N-1):
            U_[:, i] = U_C[:, i] + delta_t * rk[j] * (calcQ(i, U_P_, delta_x, a, b, c, k) - calcR(i, U_P_)/delta_x)

    return U_  # U_C


# Resulting Solution Boundaries
def resultingSolBoundaries(folder_filename, N_TIME_STEPS, SKIP_FRAMES):
    rho_low, rho_high = 0, 0
    u_low, u_high = 0, 0
    p_low, p_high = 0, 0

    # Read all files
    for i in range(0, N_TIME_STEPS, SKIP_FRAMES):
        U_P = np.loadtxt(f"{folder_filename}/U_P_timestep_{i}.csv", delimiter=",")
        U_P = np.transpose(U_P[:, 1:])

        rho = U_P[0, :]
        u = U_P[1, :]
        p = U_P[2, :]

        rho_low = np.min([rho_low, np.min(rho)])
        rho_high = np.max([rho_high, np.max(rho)])

        u_low = np.min([u_low, np.min(u)])
        u_high = np.max([u_high, np.max(u)])

        p_low = np.min([p_low, np.min(p)])
        p_high = np.max([p_high, np.max(p)])
    
    return rho_low, rho_high, u_low, u_high, p_low, p_high

import os
def checkOrCreateFolderResults(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created successfully in the current directory.")
    else:
        print(f"Folder {folder_path} already exists in the current directory.")
