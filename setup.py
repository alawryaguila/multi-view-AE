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
            "numpy",
            "scikit-learn",
            "scipy",
            "pandas",
            "matplotlib",
            "torch~=1.10.2",
            "torchvision",
            "pytorch-lightning~=1.5.5",
            "umap-learn",
            "hydra-core==1.2.0",
        ],
        description="A library for running multiview autoencoder models",
    )
    setup(**data)


if __name__ == "__main__":
    setup_package()
