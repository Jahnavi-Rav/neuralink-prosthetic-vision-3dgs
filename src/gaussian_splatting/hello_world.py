"""
Hello World for 3D Gaussian Splatting
Minimal working example to verify the pipeline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimpleGaussianRenderer:
    """Minimal 3DGS implementation for testing"""
    
    def __init__(self, num_points=100):
        self.num_points = num_points
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize random 3D points
        self.positions = torch.randn(num_points, 3, device=self.device) * 2.0
        self.colors = torch.rand(num_points, 3, device=self.device)
        self.sizes = torch.ones(num_points, 1, device=self.device) * 0.1
        
        print(f"✅ Initialized {num_points} Gaussian primitives")
        print(f"✅ Device: {self.device}")
    
    def render_simple(self, camera_pos, image_size=(256, 256)):
        """
        Simple rendering using splatting
        Projects 3D Gaussians onto 2D image plane
        """
        H, W = image_size
        
        # FIXED: Ensure camera_pos is a tensor on the correct device
        camera_pos = torch.tensor(camera_pos, device=self.device, dtype=torch.float32)
        
        # Create empty image
        image = torch.zeros(H, W, 3, device=self.device)
        
        # Project points to 2D (simple perspective projection)
        # Move camera to origin
        points_cam = self.positions - camera_pos.unsqueeze(0)
        
        # Perspective projection (assuming focal length = 1.0)
        focal = 1.0
        z = points_cam[:, 2:3] + 1e-6  # Avoid division by zero
        x_proj = (points_cam[:, 0:1] / z) * focal
        y_proj = (points_cam[:, 1:2] / z) * focal
        
        # Convert to pixel coordinates
        x_pixel = ((x_proj + 1.0) * W / 2.0).long().clamp(0, W-1)
        y_pixel = ((y_proj + 1.0) * H / 2.0).long().clamp(0, H-1)
        
        # Splat Gaussians onto image
        for i in range(self.num_points):
            if z[i] > 0:  # Only render points in front of camera
                px, py = x_pixel[i].item(), y_pixel[i].item()
                color = self.colors[i]
                
                # Simple splatting (just set pixel color)
                image[py, px] = color
        
        return image.cpu().numpy()

def main():
    print("="*60)
    print("🎨 3D GAUSSIAN SPLATTING - HELLO WORLD")
    print("="*60 + "\n")
    
    # Create renderer
    renderer = SimpleGaussianRenderer(num_points=500)
    
    # Define camera positions for multiple views
    camera_positions = [
        [0.0, 0.0, -5.0],  # Front view
        [3.0, 0.0, -5.0],  # Right view
        [0.0, 3.0, -5.0],  # Top view
        [2.0, 2.0, -5.0],  # Diagonal view
    ]
    
    # Render multiple views
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, camera_pos in enumerate(camera_positions):
        print(f"Rendering view {idx+1}/4 from camera at {camera_pos}...")
        image = renderer.render_simple(camera_pos)
        
        axes[idx].imshow(image)
        axes[idx].set_title(f"View {idx+1}: Camera at {camera_pos}")
        axes[idx].axis('off')
    
    plt.suptitle("3D Gaussian Splatting - Multiple Views", fontsize=16)
    plt.tight_layout()
    
    # Save result
    output_dir = Path("demo/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hello_world_3dgs.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Show plot (comment out if running headless)
    # plt.show()
    
    print("\n" + "="*60)
    print("✅ Hello World complete! Basic 3DGS pipeline working.")
    print(f"✅ Output saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()