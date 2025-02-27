from .HardNegativeNLLLoss import HardNegativeNLLLoss
from .TripletLoss import TripletLoss

def load_loss(loss_class, *args, **kwargs):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
    elif loss_class == 'TripletLoss':
        loss_cls = TripletLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
