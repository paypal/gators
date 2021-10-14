************
Installation
************
.. _getting_started:

Prerequisites
=============

**Gators** requires the following dependencies:

* python >=3.6
* numpy == 1.19.5
* cython
* sklearn
* pandas
* pyspark
* koalas
* xgboost
* lightgboost
* treelite
* treelite-runtime

Install
=======

From PyPi or conda-forge repositories
#####################################

Not yet available

From source available on GitHub
###############################

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  >>> git clone git@github.paypal.com:Simility-R/gators.git
  >>> cd gators
  >>> pip3 install  -r requirements.txt 
  >>> python3 setup.py build_ext --inplace
  >>> pip3 install .

Test and coverage
#################

Test
====

  >>> pytest gators -v

Test coverage
=============
  >>> coverage run -m pytest gators -v

Contribute
##########

You can contribute to this code through Pull Request on GitHub. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.paypal.com/Simility-R/gators/pulls

