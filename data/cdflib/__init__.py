# edited from cdflib https://github.com/MAVENSDC/cdflib for multicore data processing

from data.cdflib.cdfread import CDF
from . import cdfread, cdfwrite
from .epochs import CDFepoch as cdfepoch  # noqa: F401

__all__ = ["CDF"]


try:
    from ._version import version as __version__
except Exception:
    __version__ = "unknown"
