"""Setup script for sdf2d and sdf3d packages."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path("README.md")
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="sdf-library",
    version="0.1.0",
    description="2D and 3D Signed Distance Function library with AMReX integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/sdf-library",
    packages=find_packages(),  # Automatically finds sdf2d/ and sdf3d/
    py_modules=["_sdf_common"],  # Shared SDF helpers used by sdf2d and sdf3d
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        # pyAMReX is NOT on PyPI — install via conda:
        #   conda create -n pyamrex -c conda-forge pyamrex
        # This entry is a documentation placeholder only.
        "amrex": [],
        "viz": ["matplotlib>=3.5.0", "scikit-image>=0.19.0"],
        "dev": ["pytest>=7.0", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="sdf signed-distance level-set geometry amrex 2d 3d",
)
