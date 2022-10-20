from setuptools import setup, find_packages

def setup_package():
    data = dict(
        name="multiae",
        version="0.0.2",
        packages=find_packages(exclude=["*tests"]),
        author="Ana Lawry Aguila",
        author_email="ana.lawryaguila@outlook.com",
        url="https://github.com/alawryaguila/multiviewAE",
        install_requires=[
            "scipy>=1.9.0",
            "pytest>=7.1.2",
            "pandas>=1.4.3",
            "numpy>=1.23.1",
            "torchvision>=0.13.0",
            "torch>=1.12.0",
            "pytorch-lightning~=1.5.5",
            "hydra-core",
        ],
        package_data={'': ['*yaml']},
        description="A library for running multiview autoencoder models",
    )
    setup(**data)


if __name__ == "__main__":
    setup_package()
