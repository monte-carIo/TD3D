import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MinkowskiEngine import SparseTensor, MinkowskiConvolution, MinkowskiBatchNorm, MinkowskiReLU, MinkowskiPooling

class GaussianCompressionModel(nn.Module):
    def __init__(self, input_channels=6, latent_dim=4, codebook_size=8192):
        super(GaussianCompressionModel, self).__init__()
        
        # Encoder: Compresses 3D Gaussians into latent representation
        self.encoder = nn.Sequential(
            MinkowskiConvolution(input_channels, 128, kernel_size=3, dimension=3),
            MinkowskiBatchNorm(128),
            MinkowskiReLU(),
            MinkowskiPooling(kernel_size=2, stride=2, dimension=3),
            
            MinkowskiConvolution(128, 256, kernel_size=3, dimension=3),
            MinkowskiBatchNorm(256),
            MinkowskiReLU(),
            MinkowskiPooling(kernel_size=2, stride=2, dimension=3),

            MinkowskiConvolution(256, latent_dim, kernel_size=3, dimension=3),
        )

        # Vector Quantization Layer
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        
        # Decoder: Maps latent space back to 3D Gaussian parameters
        self.decoder = nn.Sequential(
            MinkowskiConvolution(latent_dim, 256, kernel_size=3, dimension=3),
            MinkowskiBatchNorm(256),
            MinkowskiReLU(),
            
            MinkowskiConvolution(256, 128, kernel_size=3, dimension=3),
            MinkowskiBatchNorm(128),
            MinkowskiReLU(),
            
            MinkowskiConvolution(128, input_channels, kernel_size=3, dimension=3)
        )

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantize(z_e)
        return z_e, z_q, indices

    def quantize(self, z_e):
        z_flattened = z_e.F.view(-1, self.codebook.embedding_dim)
        distances = torch.cdist(z_flattened, self.codebook.weight, p=2)
        closest_idxs = distances.argmin(dim=1)
        z_q = self.codebook(closest_idxs).view_as(z_e.F)
        
        # Convert to SparseTensor format
        z_q_sparse = SparseTensor(z_q, coordinates=z_e.C, device=z_e.device)
        return z_q_sparse, closest_idxs

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_e, z_q, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, z_e, z_q, indices

# Example Loss Calculation
def compute_loss(x, x_recon, z_e, z_q, commitment_weight=0.25):
    reconstruction_loss = F.mse_loss(x_recon.F, x.F)
    commitment_loss = F.mse_loss(z_e.F, z_q.F.detach())
    return reconstruction_loss + commitment_weight * commitment_loss

# Instantiate model
model = GaussianCompressionModel(input_channels=6, latent_dim=4, codebook_size=8192)
# Example input with random data for SparseTensor
cube_size = 1.0  # Size of the cube
voxel_size = 0.1  # Larger voxel size
grid_points_per_axis = int(cube_size / voxel_size)
x = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
y = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
z = np.linspace(0, cube_size, grid_points_per_axis, endpoint=False)
coords = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
coords_int = torch.tensor((coords / voxel_size).astype(int))
num_voxels = coords_int.shape[0]
features = torch.randn(num_voxels, 5)
input_tensor = SparseTensor(features, coords)


output, z_e, z_q, indices = model(input_tensor)
loss = compute_loss(input_tensor, output, z_e, z_q, commitment_weight=0.25)
