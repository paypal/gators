Installation
============

Requirements
------------

* Python >= 3.10

Installing from PyPI
--------------------

The easiest way to install Gators is via pip:

.. code-block:: bash

    pip install gators

With ONNX export support:

.. code-block:: bash

    pip install "gators[onnx]"

With tree-based discretizer support (LightGBM):

.. code-block:: bash

    pip install "gators[tree]"

With all optional dependencies:

.. code-block:: bash

    pip install "gators[all]"

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/paypal/gators.git
    cd gators
    pip install -e .

For development (includes testing and documentation dependencies):

.. code-block:: bash

    pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that Gators is installed correctly:

.. code-block:: python

    import gators
    print(gators.__version__)

You should see the version number printed without any errors.

Core Dependencies
-----------------

Gators automatically installs the following core dependencies:

* **polars** >= 1.0 - High-performance DataFrame library
* **pydantic** >= 2.0 - Data validation and settings management
* **pyarrow** >= 12.0 - Efficient data serialization
* **scikit-learn** >= 1.0 - Base estimator classes and sklearn API compatibility
* **holidays** >= 0.30 - Public holiday detection for :class:`~gators.feature_generation_dt.HolidayFeatures`

Optional Dependencies
---------------------

* **onnx** >= 1.14 + **onnxruntime** >= 1.16 — ONNX export (``pip install "gators[onnx]"``)
* **lightgbm** >= 4.0 — :class:`~gators.discretizers.TreeBasedDiscretizer` (``pip install "gators[tree]"``)
