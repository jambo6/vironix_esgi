import setuptools


with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()

setuptools.setup(
    name="vironix_esgi",
    version="0.0.1",
    author="James Morrill",
    author_email="james.morrill.6@gmail.com",
    description="Code for the work done for Vironix during the Vermont ESGI study group.",
    url="https://github.com/jambo6/vironix_esgi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=required,
    extras_require={
        "test": ["pytest"]
    }
)