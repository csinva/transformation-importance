from distutils.core import setup
import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name='trim',
    version='0.0.1',
    author="Chandan Singh, Wooseok Ha",
    description="Transformation importance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/transformation-importance",
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'scikit-image',
        'torch',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)