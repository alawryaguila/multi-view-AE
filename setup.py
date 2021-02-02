from setuptools import setup, find_packages
import os

with open('multiview_models/requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name='multiview_models',
    version='1.0.0',   
    packages=find_packages(exclude=['*tests']),
    author='Ana Lawry Aguila',
    author_email='ana.aguila.18@ucl.ac.uk',
    url='https://github.com/alawryaguila/multiview_models',
    install_requires=[required],
    description='A library for running multiview models on medical imaging data'
)