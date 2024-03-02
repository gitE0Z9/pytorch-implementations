from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="torchlake",
    version="0.1.1",
    packages=find_packages(),
    package_data={
        "": [
            "object_detection/configs/**/*.yml",
            "object_detection/datasets/**/*.yml",
            "object_detection/configs/**/*.txt",
            "object_detection/datasets/**/*.txt",
            "object_detection/datasets/**/*.csv",
        ]
    },
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
