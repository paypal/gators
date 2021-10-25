rm -rf doc
mkdir doc
sphinx-apidoc -F -M -d 1 --separate -o doc gators `find ../gators -name *.pyx`
cd doc

rm gators*rst


mkdir _static/css
cp ../doc_data/gators.css _static/css/
cp ../doc_data/GATORS_LOGO.png _static/css/
cp ../doc_data/pandas_logo.png _static/
cp ../doc_data/koalas_logo.png _static/
cp ../doc_data/cython_logo.jpeg _static/
cp ../doc_data/numpy_logo.png _static/
cp ../doc_data/sklearn_logo.png _static/
cp ../doc_data/hyperopt_logo.png _static/
cp ../doc_data/xgboost_logo.png _static/
cp ../doc_data/lightgbm_logo.png _static/
cp ../doc_data/treelite_logo.png _static/
cp -R ../doc_data/benchmarking_pandas_numpy _static/
rm conf.py
cat > conf.py <<EOL
import os
import sys
import gators

sys.path.insert(0, '..')
project = 'gators'
copyright = '2021, the gators development team.'
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

autoclass_content = "class"
autodoc_member_order = "bysource"
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
EOL

rm index.rst
cat > index.rst <<EOL
.. gators documentation

******
Gators
******

**Gators** is a machine learning library initially developed by the Simility a PayPal service Data Team. While data pre-processing and machine learning models are usually developed in Python, the pre-processing aspect is usually replaced by faster compiled programming languages in the production environment. This change of programming language is an added complexity to the model deployment process but is usually required to cope with the large number of queries per second that can be observed.

The goal of **Gators** is to be able to manage both model building and model serving using only Python, a language that data scientists are generally familiar with. **Gators** is built on top of Pandas, Koalas, NumPy and Cython. Pandas and Koalas are used for model building, while NumPy and Cython are used to speed up the model predictions in real-time. **Gators** was originally built for fraud modelling but can be generalized to many other modelling domains other than binary classification problems.

**Gators** helps to streamline the model building and productionization processes. The model building part can be done using the Pandas library for datasets held in memory, or Koalas for big data. On the model serving side, the pre-processing is carried out directly with Python, using NumPy and Cython. As a result, the speed-up using both NumPy and Cython for pre-processing is around 100 compared to standard Python code. Additionally, the per-sample response time becomes similar to other compiled languages (microsecond scale).

In summary, **Gators** is a package for handling model building, model deployment, and fast real-time pre-processing for a large number of QPS using only Python.

.. toctree::
    :maxdepth: 2

    about_gators/index
    getting_started/index
    user_guide/index
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

In 2018, **Gators** development began at Simility and had been open sourced in 2021,

========
Timeline
========
* **2018**: Development of **Gators** started.
* **2020**: Koalas and Cython packages are added to tackle out-of-core memory datasets and fast real-time pre-processing. 
* **2021**: **Gators** becomes open source.

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

.. centered:: "If I have seen further it is by standing on the shoulders of Giants."

.. centered:: Sir Isaac Newton

**gators** uses a variety of libraries internally, at each step of the model building process.

Below is the list of libraries used.

===================
Data pre-processing
===================

.. image:: ../_static/pandas_logo.png
    :width: 170 px
    :target: https://pandas.pydata.org/docs/

The most well known package for data analysis is used for data pre-processing during the model building phase. This package should be used as long as the data can fit in memory.

.. image:: ../_static/koalas_logo.png
    :width: 170 px
    :target: https://koalas.readthedocs.io/en/latest/

koalas has been chosen to replace pandas if the data does not fit in memory. The main advantage of koalas compared to other big data packages such as PySpark and Dask, is the fact that the syntax is close to pandas.

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

.. image:: ../_static/hyperopt_logo.png
    :width: 170 px
    :target: http://hyperopt.github.io/hyperopt/

This package is used for hyperparameter tuning. The three algorithms currently available to perform hyperparameter tuning are:

* Random Search
* Tree of Parzen Estimators (TPE)
* Adaptative TPE

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
    titanic
    sf_crime
    house_price
EOL

cat > user_guide/best_practices.rst <<EOL
**************
Best Practices
**************

Pandas or Koalas?
#################

The choice of using \`Pandas <https://pandas.pydata.org/>\`__ or \`Koalas <https://koalas.readthedocs.io/en/latest/>\`__ will be dictated by your data set.
For in-memory datasets it is recommended to use pandas, koalas otherwise.

Does the transformation order matter?
#####################################

Absolutely! While Pandas and Koalas dataframes hold the datatype of each column,
Numpy does not.

It is then important to group the transformations according to the datatype of the columns
they consider.

    1. datetime column transformations
    2. object column transformations
    3. encoding transformation
    4. numerical transformations

.. Note::

     After an encoding transformation, all the column datatypes with be set to *np.float32* or *np.float64*,
     any datetime columns should then be removed before this step.

What are the models currently supported by gators?
##################################################

**Gators** mainly focuses on data pre-processing in both offline and in real-time but
a submodule uses the package \`treelite <https://treelite.readthedocs.io/en/latest/>\`__ which compiles in C tree-based
models. Only this type of models is currently supported. Note that for deep learning
models, the \`tvm package <https://tvm.apache.org/>\`__ could be interesting to consider.
 
When using NumPy?
##################

In **gators**, NumPy, by means of the method \`transform_numpy()\` , should only be used in the production environment where the response time of the data pre-processing is critical.

Why the method \`fit_numpy()\` is not defined?
##############################################

The offline model building steps are only done with pandas or koalas dataframes.
First, the excellent \`Sklearn  <https://scikit#learn.org/stable/>\`__ package already handle NumPy arrays, second,
NumPy is not suitable for large-scale data.


EOL

mkdir getting_started
ln ../examples/titanic.ipynb user_guide/
ln ../examples/sf_crime.ipynb user_guide/
ln ../examples/house_price.ipynb user_guide/
ln ../examples/10min.ipynb getting_started/

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
commands to get a copy from Github and install all dependencies:

  >>> git clone git@github.paypal.com:Simility-R/gators.git
  >>> cd gators
  >>> pip3 install  -r requirements.txt 
  >>> python3 setup.py build_ext --inplace
  >>> pip3 install .

To install the dev gators enironment:
Extra packages
  >>> git clone git@github.paypal.com:Simility-R/gators.git
  >>> cd gators
  >>> pip3 install  -r requirements.txt 
  >>> python3 setup.py build_ext --inplace
  >>> brew install libomp
  >>> brew install pandoc
  >>> pip3 install .[dev]

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
   :toctree: api/
   
    _BaseDataCleaning 

Off-line data cleaning
###################### 

.. autosummary::
   :toctree: api/

    DropHighCardinality    
    DropHighNaNRatio     
    DropLowCardinality

Realtime data cleaning
###################### 

.. autosummary::
   :toctree: api/

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
   :toctree: api/

    BinRareEvents

Numerical variable binning
##########################

Base discretizer
----------------
.. autosummary::
   :toctree: api/

    _BaseDiscretizer

Discretizers
------------
.. autosummary::
   :toctree: api/

    Discretizer
    QuantileDiscretizer
    CustomDiscretizer

EOL

cat > reference/clipping.rst <<EOL

.. _api.clipping:

********
Clipping
********
.. currentmodule:: gators.clipping

.. autosummary::
   :toctree: api/

    Clipping
EOL

cat > reference/imputers.rst <<EOL

.. _api.imputers:

********
Imputers
********

Four different types of imputers are available depending on the variable datatype,
namely: numerical, integer, float, and categorical (string or object).
 

.. note::

    * *NumericsImputer* imputes numerical variables.

    * *FloatImputer* imputes only numerical variables satisfying the condition:
      
         x != x.round().

    * *IntImputer* imputes only numerical variables satisfying the condition:
    
         x == x.round()

    * *ObjectImputer* imputes only categorical variables.


Base Imputer
############
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   _BaseImputer

Numerical Imputers
##################
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   NumericsImputer
   IntImputer
   FloatImputer

Categorical Imputer
###################

.. autosummary::
   :toctree: api/

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
   :toctree: api/

BaseEncoder
###########

.. autosummary::
   :toctree: api/

   _BaseEncoder

Unsupervised Encoders
#####################

.. autosummary::
   :toctree: api/

   OrdinalEncoder
   OneHotEncoder

Binary Encoders
###############

.. autosummary::
   :toctree: api/

    WOEEncoder
    TargetEncoder

Multi-Class Encoder
###################

.. note::
    The *MultiClassEncoder* takes as input a binary encoder and apply it
    to each class. For example, if a n-class classification is composed of
    *c* categorical columns, the number encoded columns with be *n x c*.

.. autosummary::
   :toctree: api/

   MultiClassEncoder

Regression Encoder
##################

.. note::

    The *RegressionEncoder* takes as input a binary encoder and a discretizer.
    First, the discretizer transform the continuous target values into categorical values.
    Second, the MultiClassEncoder is applied to the data using the transformed target values.

.. autosummary::
   :toctree: api/

   RegressionEncoder

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
   :toctree: api/

    _BaseFeatureGeneration


Numerical Feature Generation
############################

.. autosummary::
   :toctree: api/

    ClusterStatistics
    ElementaryArithmetics
    PlaneRotation
    PolynomialFeatures

Categorical Feature Generation
##############################
.. autosummary::
   :toctree: api/

    OneHot

Feature Generation
##################

.. autosummary::
   :toctree: api/

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
   :toctree: api/
  
    _BaseDatetimeFeature

Ordinal Datetime Features
#########################
.. autosummary::
   :toctree: api/
  
    OrdinalMinuteOfHour
    OrdinalHourOfDay
    OrdinalDayOfWeek
    OrdinalDayOfMonth
    OrdinalMonthOfYear

Cyclic Datetime Features
########################

.. autosummary::
   :toctree: api/

    CyclicMinuteOfHour
    CyclicHourOfDay
    CyclicDayOfWeek
    CyclicDayOfMonth
    CyclicMonthOfYear

Delta Time Features
###################

.. autosummary::
   :toctree: api/

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
   :toctree: api/
  
    _BaseStringFeature

.. autosummary::
   :toctree: api/

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
   :toctree: api/

    _BaseFeatureSelection


Unsupervised Feature Selection
##############################

.. autosummary::
   :toctree: api/

    VarianceFilter
    CorrelationFilter


Supervised Feature Selection
############################

.. autosummary::
   :toctree: api/

    InformationValue
    MultiClassInformationValue
    RegressionInformationValue
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
   :toctree: api/

    ConvertColumnDatatype
    KoalasToPandas
    ToNumpy
EOL

cat > reference/pipeline.rst <<EOL

.. _api.pipeline:

********
Pipeline
********
.. currentmodule:: gators.pipeline

.. autosummary::
   :toctree: api/

    Pipeline
EOL

cat > reference/transformers.rst <<EOL

.. _api.transformers:

************
Transformers
************
.. currentmodule:: gators.transformers

.. autosummary::
   :toctree: api/

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
   :toctree: api/

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
    *SupervisedSampling* should be used for classification problems.

.. autosummary::
   :toctree: api/

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
   :toctree: api/

    TrainTestSplit
    HyperOpt
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
   :toctree: api/

   Transformer
   TransformerXY
EOL

mkdir benchmarking
cat > benchmarking/index.rst <<EOL
************
Benchmarking
************

Per-sample *transform* benchmarking
===================================

Benchmarking done using the jupyter notebook *%timeit* magic command.


Data Cleaning
-------------

.. image:: ../../_static/benchmarking_pandas_numpy/DropColumns.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/DropDatatypeColumns.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/KeepColumns.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/Replace.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Binning
-------

.. image:: ../../_static/benchmarking_pandas_numpy/BinRareEvents.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CustomDiscretizer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/Discretizer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/QuantileDiscretizer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Clipping
--------

.. image:: ../../_static/benchmarking_pandas_numpy/Clipping.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Scalers
-------

.. image:: ../../_static/benchmarking_pandas_numpy/MinMaxScaler.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/StandardScaler.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Imputers
--------
.. image:: ../../_static/benchmarking_pandas_numpy/NumericsImputer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/IntImputer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/FloatImputer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/ObjectImputer.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Encoders
--------

.. image:: ../../_static/benchmarking_pandas_numpy/OneHotEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/TargetEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/WOEEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/MultiClassEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/RegressionEncoder.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation
------------------

.. image:: ../../_static/benchmarking_pandas_numpy/ElementaryArithmetics.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/IsEqual.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/IsNull.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/ClusterStatistics.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OneHot.jpg
   :width: 800px
   :alt: alternate text
   :align: left


.. image:: ../../_static/benchmarking_pandas_numpy/PlaneRotation.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/PolynomialFeatures.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation String
-------------------------
.. image:: ../../_static/benchmarking_pandas_numpy/LowerCase.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/UpperCase.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/Extract.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/StringContains.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/StringLength.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/SplitExtract.jpg
   :width: 800px
   :alt: alternate text
   :align: left

Feature Generation DateTime
---------------------------

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalMinuteOfHour.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalHourOfDay.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalDayOfWeek.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalDayOfMonth.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/OrdinalMonthOfYear.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CyclicMinuteOfHour.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CyclicHourOfDay.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CyclicDayOfWeek.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CyclicDayOfMonth.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/CyclicMonthOfYear.jpg
   :width: 800px
   :alt: alternate text
   :align: left

.. image:: ../../_static/benchmarking_pandas_numpy/DeltaTime.jpg
   :width: 800px
   :alt: alternate text
   :align: left
EOL

make clean
make html
