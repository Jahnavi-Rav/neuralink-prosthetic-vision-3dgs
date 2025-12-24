"""
Complete Demo: 3DGS + Semantic Segmentation for Prosthetic Vision
Shows the full pipeline working end-to-end
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import SyntheticDataLoader
from src.gaussian_splatting.visualize import GaussianVisualizer
from src.semantic_segmentation.segmenter import RealtimeSegmenter


def create_complete_demo():
    """Generate final demo showing all components"""
    
    print("="*60)
    print("🎨 COMPLETE PIPELINE DEMO")
    print("3D Gaussian Splatting + Semantic Segmentation")
    print("="*60 + "\n")
    
    # Load trained model
    print("Loading trained 3DGS model...")
    visualizer = GaussianVisualizer("models/checkpoints/final.pth")
    
    # Load segmenter
    print("Loading semantic segmentation model...")
    segmenter = RealtimeSegmenter()
    
    # Load data
    data_loader = SyntheticDataLoader(num_views=20)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    view_indices = [0, 5, 10, 15]
    
    print("\nGenerating visualizations...")
    
    for idx, view_idx in enumerate(view_indices):
        sample = data_loader[view_idx]
        
        # 1. Render from trained Gaussians
        rendered = visualizer.render(sample['pose'])
        
        # 2. Ground truth
        gt = sample['image'].numpy()
        
        # 3. Semantic segmentation
        seg_map, priority_map = segmenter.segment(gt)
        seg_colored = segmenter.visualize(seg_map)
        
        # Row 1: Rendered views
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.imshow(rendered)
        ax1.set_title(f"3DGS Rendered\nView {view_idx}", fontsize=10)
        ax1.axis('off')
        
        # Row 2: Ground truth
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.imshow(gt)
        ax2.set_title(f"Ground Truth\nView {view_idx}", fontsize=10)
        ax2.axis('off')
        
        # Row 3: Semantic segmentation
        ax3 = fig.add_subplot(gs[2, idx])
        ax3.imshow(seg_colored)
        ax3.set_title(f"Semantic Priority\n(Red=High, Green=Medium)", fontsize=10)
        ax3.axis('off')
    
    plt.suptitle(
        "Complete Prosthetic Vision Pipeline Demo\n"
        "Top: 3D Gaussian Splatting | Middle: Input Scene | Bottom: Semantic Priority Map",
        fontsize=16,
        fontweight='bold'
    )
    
    # Save
    output_path = Path("demo/images/complete_demo.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved complete demo to: {output_path}")
    
    # Create architecture diagram
    print("\nGenerating system architecture diagram...")
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    
    # Text-based architecture
    architecture_text = """
    ╔════════════════════════════════════════════════════════════════╗
    ║         PROSTHETIC VISION SYSTEM ARCHITECTURE                  ║
    ╚════════════════════════════════════════════════════════════════╝
    
    INPUT: Smart Glasses Camera Feed (640×480 RGB)
       ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  MODULE 1: 3D Scene Reconstruction (3D Gaussian Splatting)   │
    │  • 2,000 Gaussian primitives                                 │
    │  • Novel view synthesis                                      │
    │  • Real-time rendering (30 FPS target)                       │
    └──────────────────────────────────────────────────────────────┘
       ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  MODULE 2: Semantic Segmentation (SegFormer)                 │
    │  • Object detection (faces, people, obstacles)               │
    │  • Priority assignment (3x for faces, 2x for people)         │
    │  • Real-time inference (<100ms)                              │
    └──────────────────────────────────────────────────────────────┘
       ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  MODULE 3: Neural Encoding [NEXT PHASE]                      │
    │  • 32×32 electrode grid mapping                              │
    │  • Retinotopic projection                                    │
    │  • Phosphene simulation                                      │
    └──────────────────────────────────────────────────────────────┘
       ↓
    OUTPUT: Neuralink N1 Stimulation Pattern (1,024 electrodes)
    
    ╔════════════════════════════════════════════════════════════════╗
    ║  KEY INNOVATION: Semantic-aware Gaussian allocation           ║
    ║  → Faces get 3x more Gaussians/electrodes                     ║
    ║  → People get 2x more Gaussians/electrodes                    ║
    ║  → Enables navigation + social interaction for blind users    ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, architecture_text, 
            fontsize=10, 
            family='monospace',
            ha='center', 
            va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
    
    arch_path = Path("demo/images/system_architecture.png")
    plt.savefig(arch_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved architecture to: {arch_path}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE DEMO GENERATED!")
    print("="*60)
    print("\nFiles created:")
    print(f"  1. {output_path}")
    print(f"  2. {arch_path}")
    print("\nYou now have:")
    print("  ✅ 3D Gaussian Splatting working")
    print("  ✅ Semantic segmentation integrated")
    print("  ✅ Complete visualization")
    print("  ✅ System architecture diagram")
    print("\n🚀 Ready to commit and show to recruiters!")
    print("="*60)


if __name__ == "__main__":
    create_complete_demo()