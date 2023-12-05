from pathlib import Path
from setuptools import setup, find_packages

setup(
    name='torchlake',
    version='0.1.0',
    packages=find_packages(),
    install_requires=Path('requirements.txt').read_text().splitlines(),
)