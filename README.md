# RewardingVisualDoubt

# 1. Dependencies 
In a conda managed environment, install with the following 

```bash
conda create -n llava_hf python=3.10
conda activate llava_hf
pip install pip==24.0
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Local Dependency 

The package uses previously developed and unpackaged **RaDialog** [repo](https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation) by introducing a simple setup.py to its local clone:

```python
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="radialog",
    packages=find_packages(),
    # Add any dependencies the code needs
    install_requires=requirements,
)
```

Then, the package is ready to be installed at the root directory of the local clone of the repo by running:

```bash
pip install -e .
```

This installation allows ```RewardingVisualDoubt``` to import the package by a simple import: ```ìmport radialog```

# 2. Installation 
Run following line at the root dir:
```shell
pip install -e . --config-settings editable_mode=compat
```