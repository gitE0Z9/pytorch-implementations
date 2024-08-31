import json
from pathlib import Path


def read_json_file(path: Path | str) -> str:
    if isinstance(path, str):
        path = Path(path)
    return json.loads(path.read_bytes())


def write_json_file(path: Path | str, data: dict) -> str:
    if isinstance(path, str):
        path = Path(path)
    return path.write_bytes(json.dumps(data))
