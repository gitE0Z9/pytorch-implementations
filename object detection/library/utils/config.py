import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


def load_classes(path: str) -> list:
    with open(path, "r") as f:
        return f.read().split("\n")
