from setuptools import setup, find_packages
import os

src_dir = os.path.join(os.getcwd(), 'src')
packages = {"" : "src"}
for package in find_packages("src"):
    packages[package] = "src"

setup(
    name='multiview_models',
    version='1.0.0',   
    packages=packages.keys(),
    package_dir={"":"src"},
    author='Ana Lawry Aguila',
    author_email='ana.lawryaguila@outlook.com',
    url='https://github.com/alawryaguila/multiview_models',
    description='A library for running multiview models on medical imaging data'
)