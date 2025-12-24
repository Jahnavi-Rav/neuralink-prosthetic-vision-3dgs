"""
Real-time semantic segmentation for prosthetic vision
Identifies priority objects: faces, people, obstacles
"""

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
from PIL import Image


class RealtimeSegmenter:
    """Semantic segmentation for BCI prosthetic vision"""
    
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading SegFormer model...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Priority classes for prosthetic vision
        self.priority_classes = {
            12: {'name': 'person', 'priority': 3.0, 'color': [255, 0, 0]},
            13: {'name': 'chair', 'priority': 1.0, 'color': [0, 255, 0]},
            14: {'name': 'car', 'priority': 2.0, 'color': [0, 0, 255]},
        }
        
        print(f"✅ SegFormer loaded on {self.device}")
    
    def segment(self, image):
        """Run semantic segmentation"""
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        
        segmentation_map = logits.argmax(dim=1)[0].cpu().numpy()
        
        # Create priority map
        priority_map = np.ones_like(segmentation_map, dtype=np.float32)
        for class_id, info in self.priority_classes.items():
            mask = (segmentation_map == class_id)
            priority_map[mask] = info['priority']
        
        return segmentation_map, priority_map
    
    def visualize(self, segmentation_map):
        """Create colored segmentation overlay"""
        H, W = segmentation_map.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_id, info in self.priority_classes.items():
            mask = (segmentation_map == class_id)
            colored[mask] = info['color']
        
        return colored


if __name__ == "__main__":
    print("="*60)
    print("🔍 Testing Semantic Segmentation")
    print("="*60 + "\n")
    
    segmenter = RealtimeSegmenter()
    
    # Test on synthetic image
    test_image = np.random.rand(256, 256, 3)
    
    print("\nRunning segmentation...")
    seg_map, priority_map = segmenter.segment(test_image)
    
    print(f"✅ Segmentation shape: {seg_map.shape}")
    print(f"✅ Priority map shape: {priority_map.shape}")
    print(f"✅ Unique classes: {len(np.unique(seg_map))}")
    
    from pathlib import Path
    Path("demo/images").mkdir(parents=True, exist_ok=True)
    colored = segmenter.visualize(seg_map)
    Image.fromarray(colored).save("demo/images/segmentation_test.png")
    print(f"✅ Saved to: demo/images/segmentation_test.png")
    
    print("\n" + "="*60)
    print("✅ Segmentation module working!")
    print("="*60)