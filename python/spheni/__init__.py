from ._core import *

__version__ = "0.3.1"


def hello() -> None:
    print("hello from spheni!")


def version() -> str:
    return __version__
