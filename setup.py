from setuptools import setup, find_packages
with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

def setup_package():
    data = dict(
        name="multiviewae",
        version="1.0.0",
        packages=find_packages(exclude=["*tests"]),
        package_data={'': ['*yaml']},
        author="Ana Lawry Aguila, Alejandra Jayme",
        author_email="ana.lawryaguila@outlook.com, alejandra.jayme@icloud.com",
        url="https://github.com/alawryaguila/multi-view-AE",
        install_requires=REQUIRED_PACKAGES,
        description="A library for running multi-view autoencoder models",
    )
    setup(**data)


if __name__ == "__main__":
    setup_package()
