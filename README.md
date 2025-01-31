# RewardingVisualDoubt

# Installation 
Run following line at the root dir:
```shell
pip install -e .
```
# Dependencies 
The package uses previously developed and unpackaged **RaDialog** [repo](https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation) by introducing a simple setup.py to its local clone:

```python
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="radialogue",
    packages=find_packages(),
    # Add any dependencies the code needs
    install_requires=requirements,
)
```

Then, the package is ready to be installed at the root directory of the local clone of the repo by running:

```bash
pip install -e .
```

This installation allows ```RewardingVisualDoubt``` to import the package by a simple import: ```ìmport radialogue```