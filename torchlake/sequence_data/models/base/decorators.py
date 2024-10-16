from torch import nn


def register_model_class(model_class: nn.Module):
    def decorator(cls: nn.Module):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            return original_init(self, model_class=model_class, *args, **kwargs)

        cls.__init__ = new_init

        return cls

    return decorator
