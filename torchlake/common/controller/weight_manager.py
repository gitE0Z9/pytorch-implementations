from pathlib import Path
from typing import Any, Mapping

import torch

from ..utils.platform import get_file_size


class WeightManager:
    def __init__(self, weights_filename_format: str):
        self.weights_filename_format = weights_filename_format

    def get_filename(self, *args, **kwargs) -> Path:
        filename = self.weights_filename_format.format(*args, **kwargs)

        return Path(filename)

    def save_weight(
        self,
        state_dict: Mapping[str, Any],
        filename: str | Path,
        verbose: bool = True,
    ):
        if isinstance(filename, str):
            filename = Path(filename)

        torch.save(state_dict, filename)

        if verbose:
            print(
                "Save weight to %s, model size is %s"
                % (filename, get_file_size(filename))
            )

    def load_weight(
        self,
        filename: str | Path,
        dest: torch.nn.Module,
        strict: bool = True,
        assign: bool = False,
    ):
        if isinstance(dest, torch.nn.Module):
            dest.load_state_dict(torch.load(filename), strict, assign)
        else:
            dest.load_state_dict(torch.load(filename))
