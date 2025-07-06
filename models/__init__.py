from .ClassNet import ClassNet
from .FPNSegUNet import FPNSegUNet


try:
    from .MinkUNet import MinkUNet34A
except ImportError:
    import warnings

    warnings.warn("To use MinkUNet34A install Minkowski Engine.")
