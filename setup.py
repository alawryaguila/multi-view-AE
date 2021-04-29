from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='multiview_models',
    version='1.0.0',   
    packages=find_packages(exclude=['*tests']),
    package_dir={"":"src"},
    author='Ana Lawry Aguila',
    author_email='ana.lawryaguila@outlook.com',
    url='https://github.com/alawryaguila/multiview_models',
    install_requires=[required],
    description='A library for running multiview models on medical imaging data'
)