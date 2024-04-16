from setuptools import setup

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="retrieval-augmented-generation",
    version="0.0.1",
    packages=["retrieval-augmented-generation"],
    install_requires=reqs,
)
