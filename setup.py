from setuptools import setup

setup(
    name="gym_puddle",
    version="0.0.2",  # compatible with the new gymnasium
    py_modules=["gym_puddle"],  # the module name
    install_requires=[
        "gymnasium==0.29.1",
        "numpy==1.26.4",
        "pygame==2.5.2",
        "numba==0.59.1",
        "opencv-python==4.9.0.80",
        "tqdm==4.66.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "scipy==1.13.0",
    ],  # Any other dependencies would go here
    python_requires=">=3.9",  # compatible python version
)
