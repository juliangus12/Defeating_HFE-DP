import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to compute the number of iterations
def calculate_iterations(epsilon, psnr_target, base_psnr, scale_factor=5):
    return scale_factor * (psnr_target - base_psnr) / epsilon

# Define the range of epsilon and PSNR values with finer resolution
epsilons = np.linspace(0.01, 1, 200)  # Increased resolution for epsilon
psnr_targets = np.arange(30, 60, 0.1)   # Increased resolution for PSNR

# Base PSNR obtained without any iterations
base_psnr = 30

# Create a meshgrid for epsilon and PSNR values
epsilons_mesh, psnr_mesh = np.meshgrid(epsilons, psnr_targets)

# Calculate the number of iterations for each pair of epsilon and PSNR values
iterations_mesh = calculate_iterations(epsilons_mesh, psnr_mesh, base_psnr)
# Apply log base 10 to the number of iterations
log_iterations_mesh = np.log10(iterations_mesh)

# Plot the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(epsilons_mesh, psnr_mesh, log_iterations_mesh, cmap='viridis')

# Create the transparent red plane
plane_epsilon = np.linspace(0, 0.05, 200)
plane_psnr = np.linspace(30, 60, 200)
plane_epsilon_mesh, plane_psnr_mesh = np.meshgrid(plane_epsilon, plane_psnr)
plane_iterations = np.zeros_like(plane_epsilon_mesh)  # The plane is at Z=0

# Plot the transparent red plane
ax.plot_surface(plane_epsilon_mesh, plane_psnr_mesh, plane_iterations, color='red', alpha=0.3)

# Set labels
ax.set_xlabel('Epsilon')
ax.set_ylabel('PSNR (dB)')
ax.set_zlabel('Log10(Number of Iterations)')

# Set title
ax.set_title('PSNR, Epsilon, and Iterations')

# Show the plot
plt.show()
