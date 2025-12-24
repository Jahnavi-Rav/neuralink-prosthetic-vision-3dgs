"""
Synthetic data generator for 3DGS training
No downloads needed - generates data on-the-fly
"""

import torch
import numpy as np
from PIL import Image

class SyntheticDataLoader:
    """Generate synthetic scenes for training"""
    
    def __init__(self, num_views=20, image_size=(256, 256)):
        self.num_views = num_views
        self.image_size = image_size
        
        # Generate camera poses (circular trajectory looking at center)
        self.camera_poses = []
        for i in range(num_views):
            angle = (i / num_views) * 2 * np.pi
            radius = 4.0
            
            self.camera_poses.append({
                'position': [
                    float(radius * np.cos(angle)),
                    float(radius * np.sin(angle)),
                    1.5
                ],
                'fx': 200.0,
                'fy': 200.0,
                'cx': image_size[1] / 2,
                'cy': image_size[0] / 2,
            })
        
        print(f"✅ Generated {num_views} synthetic views")
    
    def __len__(self):
        return self.num_views
    
    def __getitem__(self, idx):
        """Generate one training sample"""
        H, W = self.image_size
        
        # Create image with gradients and shapes
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Get camera position for this view
        cam_x, cam_y, cam_z = self.camera_poses[idx]['position']
        
        # Color gradients
        for y in range(H):
            for x in range(W):
                img[y, x, 0] = int((x / W) * 255)  # Red gradient
                img[y, x, 1] = int((y / H) * 255)  # Green gradient
                img[y, x, 2] = int(((cam_x + 4) / 8) * 255)  # Blue varies by camera
        
        # Add a white circle in center
        center_x, center_y = W // 2, H // 2
        for y in range(H):
            for x in range(W):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 30:
                    img[y, x] = [255, 255, 255]
        
        # Convert to torch tensor
        image_tensor = torch.from_numpy(img).float() / 255.0  # (H, W, 3)
        
        return {
            'image': image_tensor,
            'pose': self.camera_poses[idx]
        }


# Test it
if __name__ == "__main__":
    print("="*60)
    print("🎨 Testing Synthetic Data Loader")
    print("="*60)
    
    loader = SyntheticDataLoader(num_views=10)
    
    # Get first sample
    sample = loader[0]
    print(f"\n✅ Image shape: {sample['image'].shape}")
    print(f"✅ Camera position: {sample['pose']['position']}")
    
    # Save first image to verify
    img_np = (sample['image'].numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save("demo/images/synthetic_sample.png")
    print(f"✅ Saved sample to: demo/images/synthetic_sample.png")
    
    print("\n" + "="*60)
    print("✅ Data loader working!")
    print("="*60)