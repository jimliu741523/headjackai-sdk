import os
import glob
from pathlib import Path
from setuptools import setup, find_packages

NAME = "headjackai-sdk"
REQUIRES_PYTHON = ">=3.6"
DESCRIPTION = "The SDK of Headjack-AI"

def read(fname="README.md"):
    with open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as cfile:
        return cfile.read()
    
    
version_path = os.path.join('headjackai', '__version__.py')    

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())
        
        

packages = find_packages()
print(f"packages: {packages}")


setup(
    name=NAME,
    version='0.1.32',
    author="chunting liu",
    author_email="jim_liu@symphox.net",
    description=DESCRIPTION,
    license="MIT Licence",
    keywords="Headjack ai SDK",
    packages=packages,
    install_requires=requirements,
    python_requires=REQUIRES_PYTHON,
    long_description=read("README.md"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
