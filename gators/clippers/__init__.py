"""Clipping transformers for outlier handling."""

from gators.clippers.custom_clipper import CustomClipper
from gators.clippers.gaussian_clipper import GaussianClipper
from gators.clippers.iqr_clipper import IQRClipper
from gators.clippers.mad_clipper import MADClipper
from gators.clippers.quantile_clipper import QuantileClipper

__all__ = ["CustomClipper", "GaussianClipper", "IQRClipper", "MADClipper", "QuantileClipper"]
