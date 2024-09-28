import torch
import glob , os
import torch
import numpy as np
from plyfile import PlyData
import PIL
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import os
import glob
import PIL.Image
import lightning as L
from lightning.fabric import Fabric
from torch.amp import autocast, GradScaler
from functools import partial
from torch.utils.checkpoint import checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
image_dir = "Processed_Data/360deg_img"
point_cloud_dir = "Processed_Data/3D_PointCloud"

class OptimizedMVSNet(nn.Module):
    def __init__(self):
        super(OptimizedMVSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Fewer channels
        self.gn1 = nn.GroupNorm(4, 32)  # GroupNorm uses less memory
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
    scaler = GradScaler()
    num_steps = num_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images, ground_truth_depth = batch
            images = images.to(device)

            # Mixed-precision forward pass
            with autocast(device_type='cuda', dtype=torch.bfloat16):  # Ensure bfloat16 mixed precision
                predicted_depth = model(images)
                ground_truth_depth = load_point_cloud_to_depth_map(ground_truth_depth, img_shape).to(device)
                loss = depth_loss(predicted_depth, ground_truth_depth)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    print("Training complete!")
    torch.save(model.state_dict(), 'optimized_mvsnet.pth')
    print("Model saved as 'optimized_mvsnet.pth'")
    return model

def create_depth_map_from_point_cloud_orthographic(point_cloud, image_shape):
    """
    Creates a depth map by projecting a point cloud onto a 2D image plane (orthographic projection).
    The Z-coordinate of each point is used as the depth value.

    Args:
        point_cloud (o3d.geometry.PointCloud): The point cloud loaded using Open3D.
        image_shape (tuple): The shape of the output depth map (height, width).

    Returns:
        torch.Tensor: The depth map of shape (height, width).
    """
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    
    # Normalize the points to fit the image space (assuming orthographic projection)
    # Map the X and Y coordinates to the image plane (height, width), and Z is used as the depth.
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Create an empty depth map
    depth_map = np.full(image_shape, np.inf)  # Initialize with infinity (representing max depth)

    # Convert point cloud coordinates to image coordinates
    height, width = image_shape
    x_image = np.clip(((points[:, 0] - x_min) / (x_max - x_min) * width).astype(int), 0, width - 1)
    y_image = np.clip(((points[:, 1] - y_min) / (y_max - y_min) * height).astype(int), 0, height - 1)
    
    # Iterate over each point and assign its Z value to the corresponding image pixel
    for i in range(points.shape[0]):
        depth_value = points[i, 2]  # Z value is the depth
        if depth_map[y_image[i], x_image[i]] > depth_value:  # Keep the closest point (min depth)
            depth_map[y_image[i], x_image[i]] = depth_value

    # Replace infinity with zeros (indicating no depth information)
    depth_map[depth_map == np.inf] = 0

    # Convert the depth map to a torch tensor
    return torch.tensor(depth_map, dtype=torch.float32)

def load_point_cloud_to_depth_map(point_cloud_dir, image_shape):
    point_cloud_dir = "Processed_Data/3D_PointCloud"
    point_cloud_files = [os.path.join(point_cloud_dir, f).replace("\\", "/") for f in os.listdir(point_cloud_dir) if f.endswith('.ply')]
    depth_maps = []

    for pc_file in point_cloud_files:
        pcd = o3d.io.read_point_cloud(pc_file)
        
        image_shape = (576, 576)

        # Generate the depth map
        depth_map = create_depth_map_from_point_cloud_orthographic(pcd, image_shape)
        # Rescale depth map to the desired shape
        depth_map_tensor = depth_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] shape
        resized_depth_map = F.interpolate(depth_map_tensor, size=(image_shape[0] // 2, image_shape[1] // 2), mode='bilinear', align_corners=False)
        
        depth_maps.append(resized_depth_map.squeeze(0))  # Remove the batch dimension

    return torch.stack(depth_maps)  # Stack all depth maps into a tensor

def create_dataloader(image_dir, point_cloud_dir, batch_size=1):
    images = load_images(image_dir)
    ground_truth_depth_maps = load_point_cloud_to_depth_map(point_cloud_dir, images[0].shape)
    img_shape = images[0].shape

    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), ground_truth_depth_maps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def normalize_images(images):
    """ Normalize images to the range [0, 1]. """
    return images / 255.0

def load_images(image_paths):
    output = []
    path = glob.glob(f"{image_paths}/*")
    for dir in path:
        images = []
        for path in glob.glob(f"{dir}/*"):
            path = str(path).replace("\\",'/')
            #print(path)
            img = PIL.Image.open(path)
            images.append(img)
        output.append(images)
        print(np.array(output).shape)
    return np.array(output)

if __name__ == '__main__':
    images = load_images(image_dir)
    ground_truth_depth_maps = load_point_cloud_to_depth_map(point_cloud_dir, images[0].shape)

    dataset = torch.utils.data.TensorDataset(torch.tensor(images, dtype=torch.float32), ground_truth_depth_maps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = train_mvsnet(dataloader, point_cloud_dir)
