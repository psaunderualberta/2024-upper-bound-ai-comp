from setuptools import setup

setup(
    name="gym_puddle",
    version="0.0.2",  # compatible with the new gymnasium
    install_requires=[
        "gymnasium==0.29.1",
        "numpy==1.26.4",
        "pygame==2.5.2",
    ],  # And any other dependencies we need
    python_requires=">=3.11", # compatible python version
)
