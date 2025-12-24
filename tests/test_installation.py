"""
Installation smoke test
Verifies all core dependencies are working
"""

import sys

def test_pytorch():
    print("Testing PyTorch...")
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
    print(f"  ✅ MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
    
    # Test basic tensor ops
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x @ y
    assert z.shape == (3, 3), "Matrix multiplication failed"
    print(f"  ✅ Tensor operations working")

def test_computer_vision():
    print("\nTesting Computer Vision libraries...")
    import cv2
    import PIL
    from PIL import Image
    import numpy as np
    
    print(f"  ✅ OpenCV {cv2.__version__}")
    print(f"  ✅ Pillow {PIL.__version__}")
    
    # Test image creation
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    assert pil_img.size == (100, 100), "Image creation failed"
    print(f"  ✅ Image operations working")

def test_transformers():
    print("\nTesting Transformers...")
    import transformers
    print(f"  ✅ Transformers {transformers.__version__}")

def test_3d_processing():
    print("\nTesting 3D processing libraries...")
    import open3d as o3d
    import trimesh
    print(f"  ✅ Open3D {o3d.__version__}")
    print(f"  ✅ Trimesh {trimesh.__version__}")

def test_scientific():
    print("\nTesting Scientific libraries...")
    import numpy as np
    import scipy
    import matplotlib
    print(f"  ✅ NumPy {np.__version__}")
    print(f"  ✅ SciPy {scipy.__version__}")
    print(f"  ✅ Matplotlib {matplotlib.__version__}")

def test_utilities():
    print("\nTesting Utilities...")
    import yaml
    import tqdm
    from tensorboard import program
    print(f"  ✅ PyYAML working")
    print(f"  ✅ tqdm working")
    print(f"  ✅ TensorBoard working")

def main():
    print("="*60)
    print("🔬 INSTALLATION SMOKE TEST")
    print("="*60)
    
    try:
        test_pytorch()
        test_computer_vision()
        test_transformers()
        test_3d_processing()
        test_scientific()
        test_utilities()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED! Installation verified.")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
