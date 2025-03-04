import torch.nn as nn

from .quantization_utils import *


def freeze_model(model):
    """
    freeze the activation range. Resursively invokes layer.fix()
    """
    if type(model) in [QuantAct]:
        model.fix()
    elif isinstance(model, nn.Sequential):
        for n, m in model.named_children():
            freeze_model(m)
    elif isinstance(model, nn.ModuleList):
        for n in model:
            freeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                freeze_model(mod)


def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    if type(model) in [QuantAct]:
        model.unfix()
    elif isinstance(model, nn.Sequential):
        for n, m in model.named_children():
            unfreeze_model(m)
    elif isinstance(model, nn.ModuleList):
        for n in model:
            unfreeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                unfreeze_model(mod)
