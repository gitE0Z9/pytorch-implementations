import json
from pathlib import Path

import scipy.io


def read_json_file(path: Path | str) -> str:
    return json.loads(Path(path).read_bytes())


def write_json_file(path: Path | str, data: dict) -> str:
    return Path(path).write_text(json.dumps(data))


def read_matlab_file(path: Path | str) -> dict:
    return scipy.io.loadmat(Path(path))
