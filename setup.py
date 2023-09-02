from setuptools import setup
import setuptools

setup(
    name='histology-AI',
    description='Histology AI',
    url='https://github.com/frknayk/histology-AI',
    install_requires=["torch", "numpy"],
    version='0.1',
    author="frknayk",
    author_email="furkanayik@outlook.com",
    packages=setuptools.find_packages(),
)
