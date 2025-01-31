# setup.py
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="RewardingVisualDoubt",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    author="Onur Deniz Güler",
    author_email="onurdenizguler@gmail.com",
    description="A PPO-RL library to train visual LLMs in medical domain.",
    python_requires=">=3.10",
)
