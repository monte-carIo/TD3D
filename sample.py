import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set up random coordinates and features
cube_size = 1.0  # Size of the cube
voxel_size = 0.1  # Larger voxel size
grid_points_per_axis = int(cube_size / voxel_size)
x = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
y = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
z = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
coords = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
coords_int = torch.tensor((coords / voxel_size).astype(int))
breakpoint()
num_voxels = coords_int.shape[0]
features = torch.zeros(num_voxels, 3)  # Random feature values for color (RGB)

# Convert data to numpy for visualization
coords_np = coords
features_np = features.numpy()

# Plotting the 3D points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color based on features
scatter = ax.scatter(
    coords_np[:, 0],  # x-coordinates
    coords_np[:, 1],  # y-coordinates
    coords_np[:, 2],  # z-coordinates
    c=features_np,    # Use features as RGB colors
    s=50,             # Size of points
    marker='o'
)

# Setting axis labels for clarity
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Sparse Tensor Visualization with Random Colors')

plt.savefig('3d_sparse_tensor.png')
