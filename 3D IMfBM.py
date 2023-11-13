# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:50:42 2023

@author: huxin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:50:14 2023

@author: huxin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Hurst_function(t, N):
    return 0.1 if t < N/2 else 0.9  # Increasing Hurst exponent

def Diffusion_function(t):
    return 5  # Diffusion constant

def generate_1D_MfBM(N, Hurst_func, Diff_func, seed=None):
    # Set the random seed for reproducibility if given
    if seed is not None:
        np.random.seed(seed)

    # Initialize the covariance matrix
    cov = np.zeros((N, N))
    mean = np.zeros(N)

    # Compute the covariance matrix
    for i in range(N):
        for j in range(N):
            # Distance between points
            d = abs(i - j)
            # Using a small offset for the diagonal to avoid division by zero or log of zero
            d = max(d, 0.01)
            # Compute the covariance between points i and j
            cov[i, j] = np.sqrt(Diff_func(i) * Diff_func(j)) * (d ** (Hurst_func(i, N) + Hurst_func(j, N) - 2))

    # Generate a sample path
    sample_path = np.random.multivariate_normal(mean, cov, 1).T
    sample_path = np.cumsum(sample_path, axis=0)  # Cumulative sum to get the path

    return sample_path

# Usage
N = 1000
seed = 1
generated_path = generate_1D_MfBM(N, Hurst_function, Diffusion_function, seed)

# Plotting
timestep = np.linspace(0, N-1, N)
plt.plot(timestep, generated_path)
plt.title("MfBM process")
plt.xlabel("time steps")
plt.ylabel("1D displacement")
plt.show()


import numpy as np

# Assuming you have a function to generate 1D MfBM:
# generate_1d_mfbm(N, H_function) where N is the number of steps
# and H_function is a function that provides H(t) for the given step t

def generate_3d_mfbm(N, H_functions):
    # H_functions is a tuple of functions (H_x, H_y, H_z)
    mfbm_x = generate_1D_MfBM(N, Hurst_function, Diffusion_function)
    mfbm_y = generate_1D_MfBM(N, Hurst_function, Diffusion_function)
    mfbm_z = generate_1D_MfBM(N, Hurst_function, Diffusion_function)
    
    # Combine the three 1D MfBMs into a single 3D MfBM
    mfbm_3d = np.vstack((mfbm_x, mfbm_y, mfbm_z)).T  # .T to transpose for (N, 3) shape
    return mfbm_3d

# Example usage:
N = 1000  # Number of steps
H_x = lambda t: 0.5  # Example Hurst function for x dimension
H_y = lambda t: 0.6  # Example Hurst function for y dimension
H_z = lambda t: 0.7  # Example Hurst function for z dimension

mfbm_3d = generate_3d_mfbm(N, (H_x, H_y, H_z))


from mpl_toolkits.mplot3d import Axes3D

# Generate or load your data here
# For demonstration, I'll just create some dummy data
# Replace these lines with:
# generated_path_x = generate_1D_MfBM(N, Hurst_function, Diffusion_function, seed)
# generated_path_y = generate_1D_MfBM(N, Hurst_function, Diffusion_function, seed)
# generated_path_z = generate_1D_MfBM(N, Hurst_function, Diffusion_function, seed)

N = 1000
mfbm_x = generate_1D_MfBM(N, Hurst_function, Diffusion_function, 1)
mfbm_y = generate_1D_MfBM(N, Hurst_function, Diffusion_function, 2)
mfbm_z = generate_1D_MfBM(N, Hurst_function, Diffusion_function, 3)
# Plotting the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
time = np.array([np.linspace(0, N-1, N)]).T
MfBM = np.squeeze(np.array([time, mfbm_x, mfbm_y, mfbm_z])).T

# Plot the 3D trajectory
ax.plot(mfbm_x, mfbm_y, mfbm_z)

# Set labels according to the axis
ax.set_xlabel('X displacement')
ax.set_ylabel('Y displacement')
ax.set_zlabel('Z displacement')

# Title of the plot
ax.set_title('3D MfBM Trajectory')

# Show the plot
#plt.show()


np.savetxt("output_3D.csv", MfBM, delimiter=",")