from setuptools import setup, find_packages
from pathlib import Path


REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"

with open(str(REQUIREMENTS_PATH), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="summac",
    packages=find_packages(include=["summac*"]),
    version="0.0.1",
    license="Apache",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=requirements,
    # extras_require={},
    include_package_data=True,
)
