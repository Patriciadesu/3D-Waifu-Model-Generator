import torch
import glob
import os
import numpy as np
from plyfile import PlyData
import PIL
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import lightning as L
from lightning.fabric import Fabric
from torch.amp import autocast, GradScaler
from functools import partial
from torch.utils.checkpoint import checkpoint

# Set device and configure PyTorch settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

# Directory paths
image_dir = "/content/3D-Waifu-Model-Generator/Processed_Data/360deg_img"
point_cloud_dir = "/content/3D-Waifu-Model-Generator/Processed_Data/3D_PointCloud"

class PointCloudMVSNet(nn.Module):
    def __init__(self):
        super(PointCloudMVSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, 256)

        # Prediction layers
        self.coord_pred = nn.Conv2d(256, 3, kernel_size=3, padding=1)  # Predict x, y, z coordinates
        self.rgb_pred = nn.Conv2d(256, 3, kernel_size=3, padding=1)    # Predict RGB values

    def forward(self, x):
        batch_size, num_images, height, width, channels = x.size()
        x = x.view(batch_size * num_images, channels, height, width)

        # Apply convolutional layers
        x = F.relu(checkpoint(self._forward_block, self.conv1, self.gn1, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv2, self.gn2, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv3, self.gn3, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv4, self.gn4, x), inplace=True)

        # Predict 3D coordinates and RGB values
        coords = self.coord_pred(x)  # [batch_size * num_images, 3, height, width]
        rgb = self.rgb_pred(x)       # [batch_size * num_images, 3, height, width]

        # Reshape to get final shape as [batch_size, num_images, height * width, 6]
        coords = coords.view(batch_size, num_images, 3, height * width)  # Flatten spatial dimensions
        rgb = rgb.view(batch_size, num_images, 3, height * width)        # Flatten spatial dimensions
        
        # Combine coordinates and RGB
        point_cloud = torch.cat([coords, rgb], dim=2)  # [batch_size, num_images, 6, height * width]
        
        return point_cloud.permute(0, 2, 1, 3).contiguous().view(batch_size, 6, -1)  # [batch_size, 6, num_points]
    
    def _forward_block(self, conv, norm, x):
        return norm(conv(x))

def point_cloud_loss(predicted_coords, predicted_rgb, ground_truth_coords, ground_truth_rgb):
    coord_loss = F.mse_loss(predicted_coords, ground_truth_coords, reduction='mean')  # Loss for coordinates
    rgb_loss = F.mse_loss(predicted_rgb, ground_truth_rgb, reduction='mean')  # Loss for RGB values
    return coord_loss + rgb_loss

def train_mvsnet(dataloader, point_cloud_dir, num_epochs=10, lr=0.001):
    fabric = Fabric(accelerator='cuda', devices=1, precision='bf16-true')
    fabric.launch()
    
    with fabric.init_module():
        model = PointCloudMVSNet().to(torch.bfloat16)  # Use bfloat16 for memory efficiency
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)  
    num_steps = num_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images, ground_truth = batch
            ground_truth_coords, ground_truth_rgb = ground_truth[:, :, :3], ground_truth[:, :, 3:]
            images = images.to(device)

            # Mixed-precision forward pass
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted = model(images)
                predicted_coords = predicted[:, :, :3]
                predicted_rgb = predicted[:, :, 3:]
                ground_truth_coords = ground_truth_coords.to(device)
                ground_truth_rgb = ground_truth_rgb.to(device)

                loss = point_cloud_loss(predicted_coords, predicted_rgb, ground_truth_coords, ground_truth_rgb)

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    print("Training complete!")
    torch.save(model.state_dict(), '/content/3D-Waifu-Model-Generator/point_cloud_mvsnet.pth')
    print("Model saved as 'point_cloud_mvsnet.pth'")
    return model

def load_point_cloud_with_rgb(point_cloud_dir, image_shape):
    point_cloud_files = [os.path.join(point_cloud_dir, f).replace("\\", "/") for f in os.listdir(point_cloud_dir) if f.endswith('.ply')]
    point_clouds = []

    for pc_file in point_cloud_files:
        pcd = o3d.io.read_point_cloud(pc_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)  # Extract RGB values

        point_cloud = np.concatenate([points, colors], axis=1)  # Concatenate [x, y, z] with [r, g, b]
        point_clouds.append(point_cloud)

    return torch.tensor(point_clouds, dtype=torch.float32)
def generate_point_cloud_from_model(model, images):
    model.eval()
    with torch.no_grad():
        predicted_point_cloud = model(images.to(device))
        # predicted_point_cloud will have shape [batch_size, num_points, 6]
        return predicted_point_cloud


def create_dataloader(image_dir, point_cloud_dir, batch_size=1):
    images = load_images(image_dir)
    pcds = load_point_cloud_with_rgb(point_cloud_dir, images[0].shape)
    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), ground_truth_depth_maps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_images(image_paths):
    output = []
    path = glob.glob(f"{image_paths}/*")
    for dir in path:
        images = []
        for path in glob.glob(f"{dir}/*"):
            img = PIL.Image.open(path)
            images.append(np.array(img))
        output.append(images)
    return np.array(output)

if __name__ == '__main__':
    dataloader = create_dataloader(image_dir,point_cloud_dir)
    
    model = train_mvsnet(dataloader,point_cloud_dir)
