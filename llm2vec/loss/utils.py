from .HardNegativeNLLLoss import HardNegativeNLLLoss
from .TripletLoss import TripletLoss
from .HybridLoss import HybridLoss

def load_loss(loss_class, *args, **kwargs):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
    elif loss_class == 'TripletLoss':
        loss_cls = TripletLoss
    elif loss_class == 'HybridLoss':
        loss_cls = HybridLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
