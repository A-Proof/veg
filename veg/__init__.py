"""Veyrian Execution Generator research library."""

__version__ = '0.2.0'


def __getattr__(name):
    if name in ('init', 'run', 'save', 'load', 'generate', 'SIZES'):
        from . import veg3
        return getattr(veg3, name)
    raise AttributeError(name)
