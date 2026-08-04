"""Torchtext and other functionalities for text pipeline."""

# Torchtext is halted on April 2024
# Torchdata also gave up datapipe and dataloader v2


from . import vocab, transforms, functional, data

__all__ = [
    "data",
    "vocab",
    "transforms",
    "functional",
]
