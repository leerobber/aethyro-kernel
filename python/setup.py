"""
Setup for tmg_search Python package.
"""

from setuptools import setup, find_packages

setup(
    name="tmg-search",
    version="0.1.0",
    description="Ternary Memory Graph similarity search engine",
    author="Aethyro",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "numba>=0.56",
    ],
    extras_require={
        "cuda": ["numba[cuda]"],
        "test": ["pytest>=6.0"],
        "bench": ["matplotlib", "scipy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
