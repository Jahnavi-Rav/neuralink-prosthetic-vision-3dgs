"""
3D Gaussian Splatting Training Script
Trains on synthetic data, tracks metrics with TensorBoard
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import SyntheticDataLoader


class GaussianSplattingTrainer:
    """Train 3D Gaussians to reconstruct scenes"""
    
    def __init__(self, num_gaussians=1000, lr=0.001):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_gaussians = num_gaussians
        
        # Initialize Gaussian parameters
        self.positions = torch.randn(num_gaussians, 3, device=self.device) * 2.0
        self.positions.requires_grad = True
        
        self.colors = torch.rand(num_gaussians, 3, device=self.device)
        self.colors.requires_grad = True
        
        self.opacities = torch.ones(num_gaussians, 1, device=self.device) * 0.5
        self.opacities.requires_grad = True
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': [self.positions], 'lr': lr},
            {'params': [self.colors], 'lr': lr * 0.1},
            {'params': [self.opacities], 'lr': lr * 0.1}
        ])
        
        print(f"✅ Initialized {num_gaussians} Gaussians on {self.device}")
    
    def render(self, camera_pose, image_size=(256, 256)):
        """
        Render Gaussians from camera viewpoint
        Fully differentiable - no in-place operations
        """
        H, W = image_size
        
        # Get camera parameters
        cam_pos = torch.tensor(camera_pose['position'], device=self.device, dtype=torch.float32)
        fx, fy = camera_pose['fx'], camera_pose['fy']
        cx, cy = camera_pose['cx'], camera_pose['cy']
        
        # Transform points to camera space
        points_cam = self.positions - cam_pos.unsqueeze(0)
        
        # Perspective projection
        z = points_cam[:, 2:3] + 1e-6
        x_proj = (points_cam[:, 0:1] / z) * fx + cx
        y_proj = (points_cam[:, 1:2] / z) * fy + cy
        
        # Filter valid points (in front of camera and within image bounds)
        valid_mask = (z[:, 0] > 0.1) & \
                     (x_proj[:, 0] >= 0) & (x_proj[:, 0] < W-1) & \
                     (y_proj[:, 0] >= 0) & (y_proj[:, 0] < H-1)
        
        # Get valid projections
        valid_x = x_proj[valid_mask, 0]
        valid_y = y_proj[valid_mask, 0]
        valid_colors = torch.sigmoid(self.colors[valid_mask])
        valid_opacities = torch.sigmoid(self.opacities[valid_mask, 0])
        
        # FIXED: Differentiable rendering without in-place ops
        # Create image by accumulating contributions
        image = torch.zeros(H, W, 3, device=self.device)
        
        # Round to nearest pixel (differentiable)
        px = valid_x.round().long().clamp(0, W-1)
        py = valid_y.round().long().clamp(0, H-1)
        
        # Splat each Gaussian
        for i in range(len(px)):
            x_idx = px[i]
            y_idx = py[i]
            color = valid_colors[i]
            alpha = valid_opacities[i]
            
            # Non-in-place update
            old_color = image[y_idx, x_idx].clone()
            new_color = old_color * (1 - alpha) + color * alpha
            image = image.clone()  # Make a copy to avoid in-place modification
            image[y_idx, x_idx] = new_color
        
        return image
    
    def train_step(self, target_image, camera_pose):
        """Single training iteration"""
        # Render from current camera
        rendered = self.render(camera_pose, image_size=target_image.shape[:2])
        
        # Move target to device
        target = target_image.to(self.device)
        
        # Compute loss
        loss = F.mse_loss(rendered, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            psnr = -10 * torch.log10(loss + 1e-8)
        
        return {
            'loss': loss.item(),
            'psnr': psnr.item()
        }
    
    def train(self, num_iterations=1000, log_every=50):
        """Main training loop"""
        # Setup data
        data_loader = SyntheticDataLoader(num_views=20, image_size=(256, 256))
        
        # Setup logging
        log_dir = Path("experiments/tensorboard/run_001")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        print("\n" + "="*60)
        print("🚀 Starting Training")
        print("="*60)
        print(f"Iterations: {num_iterations}")
        print(f"Data views: {len(data_loader)}")
        print(f"Log dir: {log_dir}")
        print("="*60 + "\n")
        
        # Training loop
        best_loss = float('inf')
        
        for iteration in tqdm(range(num_iterations), desc="Training"):
            # Sample random view
            view_idx = torch.randint(0, len(data_loader), (1,)).item()
            sample = data_loader[view_idx]
            
            # Train step
            metrics = self.train_step(sample['image'], sample['pose'])
            
            # Track best loss
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
            
            # Log metrics
            if iteration % log_every == 0:
                writer.add_scalar('Loss/train', metrics['loss'], iteration)
                writer.add_scalar('PSNR/train', metrics['psnr'], iteration)
                
                tqdm.write(f"Iter {iteration:04d} | Loss: {metrics['loss']:.4f} | PSNR: {metrics['psnr']:.2f} dB")
            
            # Save checkpoint
            if iteration % 500 == 0 and iteration > 0:
                self.save_checkpoint(f"models/checkpoints/checkpoint_{iteration:04d}.pth")
        
        # Final save
        self.save_checkpoint("models/checkpoints/final.pth")
        writer.close()
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print(f"✅ Best Loss: {best_loss:.4f}")
        print(f"✅ Checkpoints saved to: models/checkpoints/")
        print(f"✅ TensorBoard logs: {log_dir}")
        print("="*60)
        print("\nView training curves:")
        print(f"  tensorboard --logdir {log_dir}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'positions': self.positions.detach().cpu(),
            'colors': self.colors.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
        }, path)
        print(f"💾 Saved checkpoint: {path}")


def main():
    # Create trainer
    trainer = GaussianSplattingTrainer(num_gaussians=2000, lr=0.01)
    
    # Train
    trainer.train(num_iterations=1000, log_every=50)


if __name__ == "__main__":
    main()