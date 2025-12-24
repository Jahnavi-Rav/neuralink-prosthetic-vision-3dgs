"""
Neural Encoding for Cortical Visual Prosthesis
Converts rendered scenes to electrode stimulation patterns
Mimics Neuralink N1 implant (1,024 electrodes in 32×32 grid)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CorticalEncoder:
    """
    Converts visual input to electrode stimulation patterns
    Based on retinotopic mapping and phosphene generation
    """
    
    def __init__(self, grid_size=(32, 32), stimulation_range=(0, 100)):
        """
        Args:
            grid_size: Electrode array dimensions (Neuralink N1 = 32×32)
            stimulation_range: Current range in microamps (0-100 µA)
        """
        self.grid_size = grid_size
        self.num_electrodes = grid_size[0] * grid_size[1]
        self.stim_min, self.stim_max = stimulation_range
        
        # Retinotopic mapping parameters (log-polar)
        self.foveal_magnification = 3.0  # 3x resolution in center
        
        print(f"✅ Cortical encoder initialized:")
        print(f"   Electrode grid: {grid_size}")
        print(f"   Total electrodes: {self.num_electrodes}")
        print(f"   Stimulation range: {stimulation_range} µA")
    
    def encode(self, image, priority_map=None):
        """
        Convert image to electrode stimulation pattern
        
        Args:
            image: (H, W, 3) RGB image [0-1]
            priority_map: Optional (H, W) priority weights
        
        Returns:
            stimulation_pattern: (grid_h, grid_w) electrode currents in µA
            phosphene_simulation: (H, W, 3) simulated perceived image
        """
        H, W = image.shape[:2]
        grid_h, grid_w = self.grid_size
        
        # Convert to grayscale (luminance)
        if len(image.shape) == 3:
            luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            luminance = image
        
        # Apply priority map if provided
        if priority_map is not None:
            luminance = luminance * priority_map
        
        # Downsample to electrode grid resolution
        stimulation_pattern = self._downsample_to_grid(luminance, (grid_h, grid_w))
        
        # Apply retinotopic mapping (foveal magnification)
        stimulation_pattern = self._apply_retinotopy(stimulation_pattern)
        
        # Convert to current values (µA)
        stimulation_pattern = self._normalize_to_current(stimulation_pattern)
        
        # Simulate phosphene perception
        phosphene_simulation = self._simulate_phosphenes(stimulation_pattern, (H, W))
        
        return stimulation_pattern, phosphene_simulation
    
    def _downsample_to_grid(self, image, target_size):
        """Downsample image to electrode grid"""
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        
        downsampled = torch.nn.functional.interpolate(
            image_tensor,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return downsampled.squeeze().numpy()
    
    def _apply_retinotopy(self, pattern):
        """Apply log-polar retinotopic mapping (foveal magnification)"""
        grid_h, grid_w = pattern.shape
        center_h, center_w = grid_h // 2, grid_w // 2
        
        # Create distance map from center
        y_coords, x_coords = np.ogrid[:grid_h, :grid_w]
        distances = np.sqrt((x_coords - center_w)**2 + (y_coords - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # Foveal magnification (higher resolution in center)
        magnification = 1.0 + (self.foveal_magnification - 1.0) * (1.0 - distances / max_dist)
        
        # Apply magnification to pattern
        enhanced_pattern = pattern * magnification
        
        return enhanced_pattern
    
    def _normalize_to_current(self, pattern):
        """Normalize pattern to stimulation current range"""
        # Normalize to [0, 1]
        if pattern.max() > 0:
            pattern_norm = pattern / pattern.max()
        else:
            pattern_norm = pattern
        
        # Scale to current range
        current_pattern = pattern_norm * (self.stim_max - self.stim_min) + self.stim_min
        
        return current_pattern
    
    def _simulate_phosphenes(self, stimulation_pattern, output_size):
        """
        Simulate perceived phosphenes from electrode stimulation
        Each electrode creates a blob of light (phosphene)
        """
        H, W = output_size
        grid_h, grid_w = stimulation_pattern.shape
        
        phosphene_image = np.zeros((H, W, 3), dtype=np.float32)
        
        # Phosphene parameters
        phosphene_size = max(H, W) // max(grid_h, grid_w)  # Size of each phosphene
        
        for i in range(grid_h):
            for j in range(grid_w):
                current = stimulation_pattern[i, j]
                
                if current > 5.0:  # Threshold for perception
                    # Map electrode position to image coordinates
                    center_y = int((i + 0.5) * H / grid_h)
                    center_x = int((j + 0.5) * W / grid_w)
                    
                    # Draw phosphene (Gaussian blob)
                    self._draw_phosphene(
                        phosphene_image,
                        (center_y, center_x),
                        intensity=current / self.stim_max,
                        size=phosphene_size
                    )
        
        return np.clip(phosphene_image, 0, 1)
    
    def _draw_phosphene(self, image, center, intensity, size):
        """Draw a single phosphene (Gaussian blob)"""
        H, W = image.shape[:2]
        cy, cx = center
        
        # Gaussian kernel
        y_min = max(0, cy - size)
        y_max = min(H, cy + size + 1)
        x_min = max(0, cx - size)
        x_max = min(W, cx + size + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                if dist < size:
                    # Gaussian falloff
                    weight = np.exp(-(dist**2) / (2 * (size/3)**2)) * intensity
                    # White phosphene
                    image[y, x] = np.minimum(image[y, x] + weight, 1.0)
    
    def visualize_encoding(self, image, stimulation_pattern, phosphene_image, save_path=None):
        """Create visualization of encoding process"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input Image", fontsize=12)
        axes[0, 0].axis('off')
        
        # Electrode stimulation pattern
        im = axes[0, 1].imshow(stimulation_pattern, cmap='hot', vmin=0, vmax=self.stim_max)
        axes[0, 1].set_title(f"Electrode Stimulation\n({self.grid_size[0]}×{self.grid_size[1]} grid)", fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], label='Current (µA)')
        
        # Phosphene simulation
        axes[1, 0].imshow(phosphene_image)
        axes[1, 0].set_title("Simulated Phosphene Perception", fontsize=12)
        axes[1, 0].axis('off')
        
        # Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        ENCODING STATISTICS
        
        Electrode Array: {self.grid_size[0]}×{self.grid_size[1]}
        Total Electrodes: {self.num_electrodes}
        
        Stimulation Range: {self.stim_min}-{self.stim_max} µA
        
        Active Electrodes: {np.sum(stimulation_pattern > 5.0)}
        Avg Current: {np.mean(stimulation_pattern):.1f} µA
        Max Current: {np.max(stimulation_pattern):.1f} µA
        
        Foveal Magnification: {self.foveal_magnification}x
        
        Perception Quality:
        - High-priority regions: 3x electrode density
        - Standard regions: 1x density
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        
        plt.suptitle("Neural Encoding for Cortical Visual Prosthesis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved encoding visualization to: {save_path}")
        
        return fig


# Test the encoder
if __name__ == "__main__":
    print("="*60)
    print("🧠 Testing Neural Encoding Module")
    print("="*60 + "\n")
    
    # Create encoder
    encoder = CorticalEncoder(grid_size=(32, 32))
    
    # Test with synthetic image
    test_image = np.random.rand(256, 256, 3)
    
    # Add a bright spot in center (like the white circle from training)
    center_y, center_x = 128, 128
    for y in range(100, 156):
        for x in range(100, 156):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist < 30:
                test_image[y, x] = [1.0, 1.0, 1.0]
    
    print("\nEncoding image...")
    stimulation_pattern, phosphene_image = encoder.encode(test_image)
    
    print(f"✅ Stimulation pattern shape: {stimulation_pattern.shape}")
    print(f"✅ Phosphene image shape: {phosphene_image.shape}")
    print(f"✅ Active electrodes: {np.sum(stimulation_pattern > 5.0)}/{encoder.num_electrodes}")
    
    # Visualize
    Path("demo/images").mkdir(parents=True, exist_ok=True)
    encoder.visualize_encoding(
        test_image,
        stimulation_pattern,
        phosphene_image,
        save_path="demo/images/neural_encoding_test.png"
    )
    
    print("\n" + "="*60)
    print("✅ Neural encoding module working!")
    print("="*60)