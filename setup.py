# setup.py

from setuptools import setup, find_packages
from pathlib import Path

name = "nodegamlss"
version = "0.1.0"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=name,
    version=version,
    author="Anton Thielmann",
    author_email="anton.thielmann@tu-clausthal.de",
    description="NodeGAMLSS - an interpretable distributional deep learning GAM model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnFreTh/NodeGAMLSS",
    packages=find_packages(),
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "torch>=1.1.0",
        "numpy>=0.13",
        "scipy>=1.2.0",
        "scikit-learn>=0.17",
        "catboost>=0.12.2",
        "xgboost>=0.81",
        "matplotlib",
        "tqdm",
        "tensorboardX",
        "pandas",
        "prefetch_generator",
        "requests",
        "category_encoders",
        "filelock",
        "qhoptim",
        "mat4py",
        "interpret>=0.2",
        "pygam",
        "seaborn",
    ],
)
