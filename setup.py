from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="torchlake",
    version="0.1.2",
    packages=find_packages(),
    package_data={
        "": [
            "common/datasets/**/*.txt",
        ]
    },
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extra_requires={
        "text": [],
    },
)
