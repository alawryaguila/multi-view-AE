from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

def setup_package():
    data = dict(
        name="multiae",
        version="0.0.2",
        packages=find_packages(exclude=["*tests"]),
        author="Ana Lawry Aguila",
        author_email="ana.lawryaguila@outlook.com",
        url="https://github.com/alawryaguila/multiviewAE",
        install_requires=REQUIRED_PACKAGES,
        package_data={'': ['*yaml']},
        description="A library for running multiview autoencoder models",
    )
    setup(**data)


if __name__ == "__main__":
    setup_package()
