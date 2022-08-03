from setuptools import setup, find_packages
import os


def setup_package():
    data = dict(
        name="multiae",
        version="0.0.2",
        packages=find_packages(exclude=["*tests"]),
        author="Ana Lawry Aguila",
        author_email="ana.lawryaguila@outlook.com",
        url="https://github.com/alawryaguila/multiAE",
        install_requires=[
            "pytest",
            "numpy>=1.22",
            "scikit-learn~=0.24.1",
            "scipy~=1.7.1",
            "pandas",
            "matplotlib~=3.4.1",
            "torch>=1.10.2",
            "torchvision~=0.11.3",
            "pytorch-lightning>=1.6.0",
            "umap-learn~=0.5.2",
            "hydra-core~=1.2.0",
        ],
        description="A library for running multiview autoencoder models on medical imaging data",
    )
    setup(**data)


if __name__ == "__main__":
    setup_package()
