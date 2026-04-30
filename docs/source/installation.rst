Installation
============

Requirements
------------

* Python >= 3.10
* Polars >= 0.18.0

Installing from PyPI
--------------------

The easiest way to install Gators is via pip:

.. code-block:: bash

    pip install gators

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/paypal/gators.git
    cd gators
    pip install -e .

For development (includes testing dependencies):

.. code-block:: bash

    pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that Gators is installed correctly:

.. code-block:: python

    import gators
    print(gators.__version__)

You should see the version number printed without any errors.

Dependencies
------------

Gators automatically installs the following core dependencies:

* **polars** - High-performance DataFrame library
* **pyarrow** - For efficient data serialization
* **pandas** - Data manipulation and analysis
* **pydantic** - Data validation and settings management
* **numpy** - Numerical computing
* **scikit-learn** - For model building and evaluation

Optional dependencies for specific features:

* **lightgbm** - For Tree-based discretization   
* **holidays** - For holiday feature generation
* **pytest** - For running tests (development only)


    "pandas>=2.0.0",
    "polars>=0.19.0",
    "pydantic>=2.12.0",
    "pyarrow>=20.0.0",
    "typing-extensions>=4.15.0",
    "scikit-learn>=1.3.0",
    "lightgbm>=4.0.0",
    "holidays>=0.92",