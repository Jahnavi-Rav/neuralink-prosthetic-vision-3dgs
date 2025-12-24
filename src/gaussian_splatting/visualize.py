"""
Visualize trained 3D Gaussians with improved splatting
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import SyntheticDataLoader


class GaussianVisualizer:
    """Load and visualize trained Gaussians with better rendering"""
    
    def __init__(self, checkpoint_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.positions = checkpoint['positions'].to(self.device)
        self.colors = checkpoint['colors'].to(self.device)
        self.opacities = checkpoint['opacities'].to(self.device)
        
        print(f"✅ Loaded {len(self.positions)} Gaussians")
    
    def render(self, camera_pose, image_size=(256, 256)):
        """Render with improved Gaussian splatting (5x5 kernel)"""
        H, W = image_size
        
        cam_pos = torch.tensor(camera_pose['position'], device=self.device, dtype=torch.float32)
        fx, fy = camera_pose['fx'], camera_pose['fy']
        cx, cy = camera_pose['cx'], camera_pose['cy']
        
        # Project to camera
        points_cam = self.positions - cam_pos.unsqueeze(0)
        z = points_cam[:, 2:3] + 1e-6
        x_proj = (points_cam[:, 0:1] / z) * fx + cx
        y_proj = (points_cam[:, 1:2] / z) * fy + cy
        
        # Filter valid
        valid_mask = (z[:, 0] > 0.1) & \
                     (x_proj[:, 0] >= 0) & (x_proj[:, 0] < W) & \
                     (y_proj[:, 0] >= 0) & (y_proj[:, 0] < H)
        
        # Initialize image
        image = torch.zeros(H, W, 4, device=self.device)  # RGBA
        
        valid_x = x_proj[valid_mask, 0]
        valid_y = y_proj[valid_mask, 0]
        valid_colors = torch.sigmoid(self.colors[valid_mask])
        valid_opacities = torch.sigmoid(self.opacities[valid_mask, 0]) * 0.8
        
        # Splat with 3-pixel radius (7x7 kernel)
        splat_radius = 3
        
        for i in range(len(valid_x)):
            cx_pixel = valid_x[i].item()
            cy_pixel = valid_y[i].item()
            color = valid_colors[i].cpu().numpy()
            alpha = valid_opacities[i].item()
            
            # Splat neighborhood
            for dy in range(-splat_radius, splat_radius + 1):
                for dx in range(-splat_radius, splat_radius + 1):
                    px = int(cx_pixel + dx)
                    py = int(cy_pixel + dy)
                    
                    if 0 <= px < W and 0 <= py < H:
                        # Gaussian falloff
                        dist = np.sqrt(dx**2 + dy**2)
                        weight = np.exp(-dist**2 / 2.0) * alpha
                        
                        # Alpha compositing
                        old_rgb = image[py, px, :3]
                        old_alpha = image[py, px, 3]
                        
                        new_alpha = old_alpha + weight * (1 - old_alpha)
                        if new_alpha > 0:
                            for c in range(3):
                                image[py, px, c] = (old_rgb[c] * old_alpha + color[c] * weight) / new_alpha
                        image[py, px, 3] = new_alpha
        
        return image[:, :, :3].cpu().numpy()


def main():
    print("="*60)
    print("🎨 Visualizing Trained 3D Gaussians")
    print("="*60 + "\n")
    
    visualizer = GaussianVisualizer("models/checkpoints/final.pth")
    data_loader = SyntheticDataLoader(num_views=20)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    view_indices = [0, 5, 10, 15]
    
    print("Rendering views...")
    for idx, view_idx in enumerate(view_indices):
        sample = data_loader[view_idx]
        
        # Rendered
        rendered = visualizer.render(sample['pose'])
        
        # Ground truth
        gt = sample['image'].numpy()
        
        axes[idx].imshow(rendered)
        axes[idx].set_title(f"Rendered View {view_idx}")
        axes[idx].axis('off')
        
        axes[idx + 4].imshow(gt)
        axes[idx + 4].set_title(f"Ground Truth {view_idx}")
        axes[idx + 4].axis('off')
    
    plt.suptitle("3D Gaussian Splatting Results\nTop: Rendered | Bottom: Ground Truth", fontsize=14)
    plt.tight_layout()
    
    output_path = Path("demo/images/training_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")
    
    # Novel views
    print("\nRendering novel views...")
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    
    novel_views = [2, 7, 12, 17]
    for idx, view_idx in enumerate(novel_views):
        sample = data_loader[view_idx]
        rendered = visualizer.render(sample['pose'])
        
        axes2[idx].imshow(rendered)
        axes2[idx].set_title(f"Novel View {view_idx}")
        axes2[idx].axis('off')
    
    plt.suptitle("Novel View Synthesis (Unseen During Training)", fontsize=14)
    plt.tight_layout()
    
    output_path2 = Path("demo/images/novel_views.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to: {output_path2}")
    
    print("\n" + "="*60)
    print("✅ Visualization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()