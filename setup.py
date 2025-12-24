from setuptools import setup, find_packages

setup(
    name="neuralink-prosthetic-vision-3dgs",
    version="0.1.0",
    author="Jahnavi Ravi",
    author_email="jahnaviravi1998@gmail.com",
    description="Real-time 3D scene reconstruction for neural prosthetic vision",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)
