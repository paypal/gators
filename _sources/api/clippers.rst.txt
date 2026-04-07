Clippers
========

Outlier Clipping Methods
-------------------------

* :class:`~gators.clippers.custom_clipper.CustomClipper` - Custom min/max bounds per column
* :class:`~gators.clippers.gaussian_clipper.GaussianClipper` - Clip based on mean ± n standard deviations
* :class:`~gators.clippers.iqr_clipper.IQRClipper` - Clip based on interquartile range
* :class:`~gators.clippers.mad_clipper.MADClipper` - Clip based on median absolute deviation
* :class:`~gators.clippers.quantile_clipper.QuantileClipper` - Clip based on quantile thresholds
