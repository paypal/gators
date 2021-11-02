rm -rf doc
mkdir doc
sphinx-apidoc -F -M -d 1 --separate -o doc gators `find ../gators -name *.pyx`
cd doc

rm gators*rst


mkdir _static/css
mkdir _static/benchmarks
cp ../doc_data/gators.css _static/css/
cp ../doc_data/GATORS_LOGO.png _static/css/
cp ../doc_data/pandas_logo.png _static/
cp ../doc_data/koalas_logo.png _static/
cp ../doc_data/dask_logo.png _static/
cp ../doc_data/cython_logo.jpeg _static/
cp ../doc_data/numpy_logo.png _static/
cp ../doc_data/sklearn_logo.png _static/
cp ../doc_data/xgboost_logo.png _static/
cp ../doc_data/lightgbm_logo.png _static/
cp ../doc_data/treelite_logo.png _static/
cp -R ../benchmarks/figs/ _static/benchmarks
rm conf.py
cat > conf.py <<EOL
import os
import sys
import gators

sys.path.insert(0, '..')
project = 'gators'
copyright = '2021, Gators development team (Apache-2.0 License).'
# copyright = f"2021 - {datetime.now().year}, scikit-learn developers (BSD License)"

author = 'The gators team'
version = gators.__version__


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.autosummary',
    ]

# autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "inherited-members": True}

templates_path = ['_templates']
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
todo_include_todos = True
master_doc = 'index'
numpydoc_show_class_members = False
autosummary_generate = True
panels_add_bootstrap_css = False
html_use_index = False
html_domain_indices = False
html_theme = 'pydata_sphinx_theme'
html_css_files = ['css/gators.css']
html_static_path = ['_static']
html_logo = '../doc_data/GATORS_LOGO.png'
html_favicon = '../doc_data/gators_logo.ico'
html_theme_options = {
  "logo_link": "index",
  "github_url": "https://github.paypal.com/Simility-R/gators/",
}
man_pages = [
    ('index', 'gators', u'gators Documentation',
     [u'the gators team'], 1)
]
# html_theme_options = {"google_analytics": True}
EOL

rm index.rst
cat > index.rst <<EOL
.. gators documentation

******
Gators
******

**Gators** is a machine learning library initially developed by the **PayPal Risk as a Service** Data Team. While data pre-processing and machine learning models are usually developed in Python, the pre-processing aspect is usually replaced by faster compiled programming languages in the production environment. This change of programming language is an added complexity to the model deployment process but is usually required to cope with the large number of queries per second that can be observed.

The goal of **Gators** is to be able to manage both model building and model serving using only Python, a language that data scientists are generally familiar with. **Gators** is built on top of Pandas, Dask, Koalas, NumPy and Cython. Pandas, Dask, and Koalas are used for model building, while NumPy and Cython are used to speed up the model predictions in real-time. **Gators** was originally built for fraud modelling but can be generalized to many other modelling domains other than binary classification problems.

**Gators** helps to streamline the model building and productionization processes. The model building part can be done using the Pandas library for datasets held in memory, or Dask and Koalas for big data. On the model serving side, the pre-processing is carried out directly with Python, using NumPy and Cython. As a result, the speed-up using both NumPy and Cython for pre-processing is around 100 compared to standard Python code. Additionally, the per-sample response time becomes similar to other compiled languages (microsecond scale).

In summary, **Gators** is a package for handling model building, model deployment, and fast real-time pre-processing for a large number of QPS using only Python.

.. toctree::
    :maxdepth: 2

    about_gators/index
    getting_started/index
    user_guide/index
    examples/index
    reference/index
    benchmarking/index

EOL

mkdir about_gators
cat > about_gators/index.rst <<EOL
************
About Gators
************

**Gators** was created to help data scientists to:

* Perform in-memory and out-of-core memory data pre-processing for model building.
* Fast real-time pre-processing and model scoring.

######################
History of development
######################

In 2018, **Gators** development began at **Simility** and had been open sourced in 2021,

========
Timeline
========
* **2018**: Development of **Gators** started.
* **2020**: Dask, Koalas and Cython packages are added to tackle out-of-core memory datasets and fast real-time pre-processing. 
* **2021**: **Gators** becomes open-sourced.

##################
Library Highlights
##################

* Data pre-processing can be done for both in-memory and out-of-memory datasets using the same interface.
* Using Cython, the real-time data pre-processing is carried out on NumPy arrays with compiled C-code leading to fast response times, similar to compiled languages.

##########
Our Vision
##########

A world where data scientists can develop and push their models in production using only Python, even when there are a large number of queries per second (QPS).

###################################
Python packages leveraged in gators
###################################

.. centered:: "If I have seen further it is by standing on the shoulders of giants."

.. centered:: Sir Isaac Newton

**gators** uses a variety of libraries internally, at each step of the model building process.

Below is the list of libraries used.

===================
Data pre-processing
===================

.. image:: ../_static/pandas_logo.png
    :width: 170 px
    :target: https://pandas.pydata.org/docs/

The well-known package for data analysis is used for data pre-processing during the model building phase. This package should be used as long as the data can fit in memory.

.. image:: ../_static/koalas_logo.png
    :width: 170 px
    :target: https://koalas.readthedocs.io/en/latest/

Koalas is one of the two libraries chosen to handle the preprocessing when the data does not fit in memory. 

.. image:: ../_static/dask_logo.png
    :width: 170 px
    :target: https://docs.dask.org/en/latest/

Dask can also be used to handle the preprocessing when the data does not fit in memory. 

.. image:: ../_static/numpy_logo.png
    :width: 170 px
    :target: https://numpy.org/doc/

NumPy is used in the production environment when the pre-processing needs to be as fast as possible.

.. image:: ../_static/cython_logo.jpeg
    :width: 170 px
    :target: https://cython.readthedocs.io/en/latest/

In the production environment, the pre-processing with be done by pre-compile Cython code on NumPy arrays.

==============
Model building
==============

.. image:: ../_static/sklearn_logo.png
    :width: 170 px
    :target: https://scikit-learn.org/stable/

The most well known package for model building is used for cross-validation and model evaluation.

.. image:: ../_static/xgboost_logo.png
    :width: 170 px
    :target: https://xgboost.readthedocs.io/en/latest/

Decision tree-based package used for model building. XGBoost algorithm applies level-wise tree growth.

.. image:: ../_static/lightgbm_logo.png
    :width: 170 px
    :target: https://lightgbm.readthedocs.io/en/latest/

Decision tree-based package used for model building. LightGBM algorithm applies leaf-wise tree growth.

.. image:: ../_static/treelite_logo.png
    :width: 170 px
    :target: https://treelite.readthedocs.io/

Treelite is used to compile the trained models in C before being deployed in production,
and treelite-runtime is used for real-time model scoring.
EOL

mkdir user_guide
cat > user_guide/index.rst <<EOL
**********
User Guide
**********

.. toctree::
    :maxdepth: 2

    best_practices
EOL

cat > user_guide/best_practices.rst <<EOL
**************
Best Practices
**************

Pandas, Dask or Koalas?
#######################

The choice of using \`Pandas <https://pandas.pydata.org/>\`__, \`Dask <https://docs.dask.org/en/latest/>\`__,  or \`Koalas <https://koalas.readthedocs.io/en/latest/>\`__ will be dictated by your dataset size.
For in-memory datasets it is recommended to use Pandas, Dask or Koalas otherwise.

Does the transformation order matter?
#####################################

Absolutely! While Pandas, Dask and Koalas dataframes hold the datatype of each column,
Numpy does not.

It is then important to group the transformations according to the datatype of the columns
they consider.

    1. datetime column transformations
    2. object column transformations
    3. encoding transformations
    4. numerical transformations

.. Note::

     After an encoding transformation, the data will be only composed of numerical columns,
     any datetime columns should then be removed before this step.

What are the models currently supported by gators?
##################################################

**Gators** mainly focuses on data pre-processing in both offline and in real-time, and model deployment
with the package \`treelite <https://treelite.readthedocs.io/en/latest/>\`__ which compiles in C tree-based
models. Only this type of models is currently supported. Note that for deep learning
models, the \`tvm package <https://tvm.apache.org/>\`__ could be interesting to consider.
 
When using the method \`transform_numpy()\`?
######################################################

The method \`transform_numpy()\`, have been designed to speed-up the pre-processing in a production-like environment, where the response time of the data pre-processing is critical.
It is recommended to use \`transform_numpy()\` off-line to validate the prouction pipeline.



Why the method \`fit_numpy()\` is not defined?
##############################################

The offline model building steps are only done with Pandas, Dask, or Koalas dataframes.
First, the excellent \`Sklearn  <https://scikit#learn.org/stable/>\`__ package already handle NumPy arrays, second,
NumPy is not suitable for large-scale data.


EOL

mkdir getting_started
cp ../examples/10min.ipynb getting_started/

cat > getting_started/index.rst <<EOL

***************
Getting Started
***************
.. toctree::
    :maxdepth: 2

    install
    10min

EOL

cat > getting_started/install.rst <<EOL
************
Installation
************
.. _getting_started:

Prerequisites
=============

**Gators** requires the following dependencies:

* python >=3.6
* numpy
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

From PyPi or conda-forge 
########################


The default installation (in-memory data only):

>>> pip install gators
>>> conda install gators

To handle out-of-core data, you can choose to Dask, Koalas, or both: 

>>> pip install gators "[dask, koalas]"
>>> conda install gators "[dask, koalas]"


From source available on GitHub
###############################

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies:

>>> git clone git@github.paypal.com:Simility-R/gators.git
>>> cd gators
>>> pip install  -r requirements.txt 
>>> python setup.py build_ext --inplace
>>> pip install .

To install the dev gators enironment:
Extra packages

>>> git clone git@github.paypal.com:Simility-R/gators.git
>>> cd gators
>>> pip install  -r requirements.txt 
>>> python setup.py build_ext --inplace
>>> brew install libomp
>>> pip install .[dev]

Test and coverage
#################

Test
====

  >>> pytest gators -v

Test coverage
=============

  >>> pip install pytest-cov
  >>> pytest -v --cov-report html:cov_html --cov=gators gators

Contribute
##########

You can contribute to this code through Pull Request on GitHub. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.paypal.com/Simility-R/gators/pulls

EOL

mkdir examples
cp ../examples/titanic.ipynb examples/
cp ../examples/sf_crime.ipynb examples/
cp ../examples/house_price.ipynb examples/
cp ../examples/templates.ipynb examples/

cat > examples/index.rst <<EOL
********
Examples
********

.. toctree::
    :maxdepth: 2

    titanic
    sf_crime
    house_price
    templates
EOL

mkdir reference
cat > reference/index.rst <<EOL
*************
API Reference
*************

.. toctree::
    :maxdepth: 2

    sampling
    data_cleaning
    binning
    clipping
    scalers
    imputers
    encoders
    feature_generation
    feature_generation_str
    feature_generation_dt
    feature_selection
    pipeline
    model_building
    converter
    transformers

EOL

cat > reference/data_cleaning.rst <<EOL

.. _api.data_cleaning:

*************
Data Cleaning
*************

.. currentmodule:: gators.data_cleaning

These transformers can be used to reduce the number of columns
during the feature selection step.

Base data_cleaning transformer
##############################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst
   
    _BaseDataCleaning 

Off-line data cleaning
###################### 

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    DropHighCardinality    
    DropHighNaNRatio     
    DropLowCardinality

Realtime data cleaning
###################### 

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    ConvertColumnDatatype
    DropColumns 
    DropDatatypeColumns     
    KeepColumns
    Replace    

EOL
cat > reference/binning.rst <<EOL

.. _api.binning:

*******
Binning
*******
.. currentmodule:: gators.binning

Categorical variable binning
############################
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    BinRareCategories
    BinSingleTargetClassCategories

Numerical variable binning
##########################

Base binning
----------------
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    _BaseBinning

Binnings
------------
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    Binning
    QuantileBinning
    TreeBinning
    CustomBinning

EOL

cat > reference/clipping.rst <<EOL

.. _api.clipping:

********
Clipping
********
.. currentmodule:: gators.clipping

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    Clipping
EOL

cat > reference/imputers.rst <<EOL

.. _api.imputers:

********
Imputers
********

Two different types of imputers are available depending on the variable datatype,
namely, numerical and categorical (string or object).
 

Base Imputer
############
.. currentmodule:: gators.imputers

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   _BaseImputer

Numerical Imputers
##################
.. currentmodule:: gators.imputers

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   NumericsImputer
   ObjectImputer

EOL

cat > reference/encoders.rst <<EOL

********
Encoders
********
.. _api.encoders:

The Encoders transform the categorical columns into numerical columns.

.. note::
    The Encoders transform the data inplace. The output of an encoder is then a numerical dataframe or array.
    Before calling an encoder, all the transformations on categorical columns should be done and the datetime columns should be dropped.

.. currentmodule:: gators.encoders

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

BaseEncoder
###########

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   _BaseEncoder

Unsupervised Encoders
#####################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   OrdinalEncoder
   OneHotEncoder
   BinnedColumnsEncoder

Supervised Encoders
###################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    WOEEncoder
    TargetEncoder

.. Note::

   **WOEEncoder** is only valid for binary classification problems, and **TargetEncoder** works for binary and regression problems.
   In the case of a multiclass classification problem, it is recommended to use a one-versus-all approach in order to use these supervised encoders. 

EOL

cat > reference/feature_generation.rst <<EOL

.. _api.feature_generation:

******************
Feature Generation
******************
.. currentmodule:: gators.feature_generation

Base Feature Generation Transformer
###################################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    _BaseFeatureGeneration


Numerical Feature Generation
############################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    ClusterStatistics
    ElementaryArithmetics
    PlaneRotation
    PolynomialFeatures

Categorical Feature Generation
##############################
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    OneHot
    PolynomialObjectFeatures

Feature Generation
##################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    IsEqual
    IsNull

EOL
cat > reference/feature_generation_dt.rst <<EOL

.. _api.feature_generation_dt:

***************************
Feature Generation DateTime
***************************
.. currentmodule:: gators.feature_generation_dt

Base Datetime Feature Generation
################################
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst
  
    _BaseDatetimeFeature

Ordinal Datetime Features
#########################
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst
  
    OrdinalMinuteOfHour
    OrdinalHourOfDay
    OrdinalDayOfWeek
    OrdinalDayOfMonth
    OrdinalMonthOfYear

Cyclic Datetime Features
########################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    CyclicMinuteOfHour
    CyclicHourOfDay
    CyclicDayOfWeek
    CyclicDayOfMonth
    CyclicMonthOfYear

Delta Time Features
###################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    DeltaTime

EOL
cat > reference/feature_generation_str.rst <<EOL

.. _api.feature_generation_str:

*************************
Feature Generation String
*************************
.. currentmodule:: gators.feature_generation_str

Base String Feature Generation
##############################
.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst
  
    _BaseStringFeature

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    SplitExtract
    Extract
    StringContains
    StringLength
    LowerCase
    UpperCase
EOL
cat > reference/feature_selection.rst <<EOL

.. _api.feature_selection:

*****************
Feature Selection
*****************
.. currentmodule:: gators.feature_selection


Base Feature Selection Transformer
##################################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    _BaseFeatureSelection


Unsupervised Feature Selection
##############################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    VarianceFilter
    CorrelationFilter


Supervised Feature Selection
############################

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    InformationValue
    SelectFromModel
    SelectFromModels

EOL
cat > reference/converter.rst <<EOL

.. _api.converter:

*********
Converter
*********
.. currentmodule:: gators.converter

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    ToPandas
    ToNumpy
EOL

cat > reference/pipeline.rst <<EOL

.. _api.pipeline:

********
Pipeline
********
.. currentmodule:: gators.pipeline

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    Pipeline
EOL

cat > reference/transformers.rst <<EOL

.. _api.transformers:

************
Transformers
************
.. currentmodule:: gators.transformers

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   Transformer
   TransformerXY
EOL

cat > reference/scalers.rst <<EOL

.. _api.scalers:

*******
Scalers
*******
.. currentmodule:: gators.scalers

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   MinMaxScaler
   StandardScaler

EOL

cat > reference/sampling.rst <<EOL

.. _api.sampling:

********
Sampling
********
.. currentmodule:: gators.sampling

.. note::

    **UnsupevisedSampling** should be used for regression problems, and
    *SupervisedSampling* can be used for classification problems.

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   UnsupervisedSampling
   SupervisedSampling
EOL

cat > reference/model_building.rst <<EOL

.. _api.model_building:

***************
Model Building
***************
.. currentmodule:: gators.model_building

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

    TrainTestSplit
    XGBBoosterBuilder
    XGBTreeliteDumper
    LGBMTreeliteDumper
EOL

cat > reference/transformers.rst <<EOL

.. _api.transformers:

************
Transformers
************

.. currentmodule:: gators.transformers

.. autosummary::
   :nosignatures:
   :toctree: api/
   :template: class.rst

   Transformer
   TransformerXY
EOL

mkdir _templates
cat > _templates/class.rst <<EOL

{% extends "!autosummary/class.rst" %}

{% block methods %} {% if methods %}

{% endif %} {% endblock %}

{% block attributes %} {% if attributes %}

{% endif %} {% endblock %}
EOL

mkdir benchmarking
cat > benchmarking/index.rst <<EOL
************
Benchmarking
************

Per-sample *transform* benchmarking
===================================

The benchmarking is done using the jupyter notebook *%timeit* magic command.

.. toctree::
    :maxdepth: 2

    transform_numpy
    fit_transform
EOL

cat > benchmarking/transform_numpy.rst <<EOL

Per-sample *transform* benchmarking
====================================

Data Cleaning
-------------

.. image:: ../../_static/benchmarks/DropColumns_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/DropDatatypeColumns_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/KeepColumns_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Replace_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Binning
-------

.. image:: ../../_static/benchmarks/BinRareCategories_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CustomBinning_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Binning_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/QuantileBinning_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Clipping
--------

.. image:: ../../_static/benchmarks/Clipping_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Scalers
-------

.. image:: ../../_static/benchmarks/MinMaxScaler_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StandardScaler_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Imputers
--------
.. image:: ../../_static/benchmarks/NumericsImputer_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IntImputer_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/FloatImputer_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/ObjectImputer_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Encoders
--------

.. image:: ../../_static/benchmarks/OneHotEncoder_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalEncoder_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/TargetEncoder_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/WOEEncoder_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation
------------------

.. image:: ../../_static/benchmarks/ElementaryArithmetics_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IsEqual_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IsNull_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/ClusterStatistics_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OneHot_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left


.. image:: ../../_static/benchmarks/PlaneRotation_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/PolynomialFeatures_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation String
-------------------------
.. image:: ../../_static/benchmarks/LowerCase_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/UpperCase_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Extract_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StringContains_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StringLength_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/SplitExtract_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation DateTime
---------------------------

.. image:: ../../_static/benchmarks/OrdinalMinuteOfHour_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalHourOfDay_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalDayOfWeek_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalDayOfMonth_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalMonthOfYear_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicMinuteOfHour_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicHourOfDay_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicDayOfWeek_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicDayOfMonth_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicMonthOfYear_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/DeltaTime_transform_numpy.jpg
   :width: 800px
   :alt: alternate text
   :align: left
EOL



cat > benchmarking/fit_transform.rst <<EOL

*fit_transform* benchmarking
=============================



Data Cleaning
-------------

.. image:: ../../_static/benchmarks/DropColumns_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/DropDatatypeColumns_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/KeepColumns_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Replace_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Binning
-------

.. image:: ../../_static/benchmarks/BinRareCategories_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CustomBinning_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Binning_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/QuantileBinning_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Clipping
--------

.. image:: ../../_static/benchmarks/Clipping_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Scalers
-------

.. image:: ../../_static/benchmarks/MinMaxScaler_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StandardScaler_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Imputers
--------
.. image:: ../../_static/benchmarks/NumericsImputer_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IntImputer_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/FloatImputer_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/ObjectImputer_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Encoders
--------

.. image:: ../../_static/benchmarks/OneHotEncoder_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalEncoder_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/TargetEncoder_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/WOEEncoder_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Feature Generation
------------------

.. image:: ../../_static/benchmarks/ElementaryArithmetics_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IsEqual_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/IsNull_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/ClusterStatistics_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OneHot_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left


.. image:: ../../_static/benchmarks/PlaneRotation_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/PolynomialFeatures_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Feature Generation String
-------------------------
.. image:: ../../_static/benchmarks/LowerCase_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/UpperCase_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/Extract_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StringContains_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/StringLength_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/SplitExtract_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

Feature Generation DateTime
---------------------------

.. image:: ../../_static/benchmarks/OrdinalMinuteOfHour_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalHourOfDay_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalDayOfWeek_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalDayOfMonth_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/OrdinalMonthOfYear_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicMinuteOfHour_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicHourOfDay_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicDayOfWeek_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicDayOfMonth_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/CyclicMonthOfYear_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarks/DeltaTime_fit_transform.jpg
   :width: 400px
   :alt: alternate text
   :align: left
EOL


make clean
make html
