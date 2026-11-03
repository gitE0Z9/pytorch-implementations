from functools import wraps
from typing import Type, TypeVar

from torch import nn

from .rnn_discriminator import RNNDiscriminator

T = TypeVar("T", bound=RNNDiscriminator)


def register_model_class(model_class: Type[nn.Module]):
    def decorator(cls: Type[T]) -> Type[T]:
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs) -> None:
            original_init(self, *args, rnn_class=model_class, **kwargs)

        cls.__init__ = new_init  # type: ignore[method-assign]
        return cls

    return decorator
