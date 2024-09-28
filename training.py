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

class OptimizedMVSNet(nn.Module):
    def __init__(self):
        super(OptimizedMVSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, 256)

        # Depth prediction layer
        self.depth_pred = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, num_images, height, width, channels = x.size()
        x = x.view(batch_size * num_images, channels, height, width)

        # Apply convolutional layers with in-place ReLU, GroupNorm, and gradient checkpointing
        x = F.relu(checkpoint(self._forward_block, self.conv1, self.gn1, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv2, self.gn2, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv3, self.gn3, x), inplace=True)
        x = F.relu(checkpoint(self._forward_block, self.conv4, self.gn4, x), inplace=True)

        depth = self.depth_pred(x)
        depth = depth.view(batch_size, num_images, 1, height, width)
        return depth

    def _forward_block(self, conv, norm, x):
        return norm(conv(x))

def depth_loss(predicted_depth, ground_truth_depth):
    return F.mse_loss(predicted_depth, ground_truth_depth, reduction='mean')

def train_mvsnet(dataloader, point_cloud_dir, num_epochs=10, lr=5e-5):
    fabric = Fabric(accelerator='cuda', devices=1, precision='bf16-true')
    fabric.launch()
    
    with fabric.init_module():
        model = OptimizedMVSNet().to(torch.bfloat16)  # Use bfloat16 for memory efficiency
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW is more memory efficient
    num_steps = num_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images, ground_truth_depth = batch
            images = images.to(device)
            ground_truth_depth = ground_truth_depth.to(device)

            # Adjust the ground_truth_depth to match the model output shape
            #ground_truth_depth = ground_truth_depth.unsqueeze(1).expand(-1, 21, -1, -1)  # Shape: [batch_size, 21, 1, height, width]

            # Mixed-precision forward pass
            with autocast(device_type='cuda', dtype=torch.bfloat16):  # Ensure bfloat16 mixed precision
                predicted_depth = model(images)
                loss = depth_loss(predicted_depth, ground_truth_depth)

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    print("Training complete!")
    torch.save(model.state_dict(), '/content/3D-Waifu-Model-Generator/optimized_mvsnet.pth')
    print("Model saved as 'optimized_mvsnet.pth'")
    return model


def create_depth_map_from_point_cloud_orthographic(point_cloud, image_shape):
    points = np.asarray(point_cloud.points)

    # Normalize the points to fit the image space
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Create an empty depth map
    depth_map = np.full(image_shape, np.inf)

    # Convert point cloud coordinates to image coordinates
    height, width = image_shape
    x_image = np.clip(((points[:, 0] - x_min) / (x_max - x_min) * width).astype(int), 0, width - 1)
    y_image = np.clip(((points[:, 1] - y_min) / (y_max - y_min) * height).astype(int), 0, height - 1)

    # Assign Z values to pixels
    for i in range(points.shape[0]):
        depth_value = points[i, 2]
        if depth_map[y_image[i], x_image[i]] > depth_value:
            depth_map[y_image[i], x_image[i]] = depth_value

    depth_map[depth_map == np.inf] = 0
    return torch.tensor(depth_map, dtype=torch.float32)

def load_point_cloud_to_depth_map(point_cloud_dir, image_shape):
    point_cloud_files = [os.path.join(point_cloud_dir, f).replace("\\", "/") for f in os.listdir(point_cloud_dir) if f.endswith('.ply')]
    depth_maps = []

    for pc_file in point_cloud_files:
        pcd = o3d.io.read_point_cloud(pc_file)
        
        # Generate the depth map
        depth_map = create_depth_map_from_point_cloud_orthographic(pcd, image_shape)

        # Resize depth map to match the model output
        depth_map_tensor = depth_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] shape
        resized_depth_map = F.interpolate(depth_map_tensor, size=(576, 576), mode='bilinear', align_corners=False)
        
        depth_maps.append(resized_depth_map.squeeze(0))  # Remove the batch dimension

    return torch.stack(depth_maps)  # Stack all depth maps into a tensor

def create_dataloader(image_dir, point_cloud_dir, batch_size=1):
    images = load_images(image_dir)
    ground_truth_depth_maps = load_point_cloud_to_depth_map(point_cloud_dir, images[0].shape)
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
    images = load_images(image_dir)
    ground_truth_depth_maps = load_point_cloud_to_depth_map(point_cloud_dir, (576, 576))

    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), ground_truth_depth_maps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = train_mvsnet(dataloader,point_cloud_dir)
