**************
Best Practices
**************

Pandas or Koalas?
#################

The choice of using `Pandas <https://pandas.pydata.org/>`__ or `Koalas <https://koalas.readthedocs.io/en/latest/>`__ will be dictated by your data set.
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
a submodule uses the package `treelite <https://treelite.readthedocs.io/en/latest/>`__ which compiles in C tree-based
models. Only this type of models is currently supported. Note that for deep learning
models, the `tvm package <https://tvm.apache.org/>`__ could be interesting to consider.
 
When using NumPy?
##################

In **gators**, NumPy, by means of the method `transform_numpy()` , should only be used in the production environment where the response time of the data pre-processing is critical.

Why the method `fit_numpy()` is not defined?
##############################################

The offline model building steps are only done with pandas or koalas dataframes.
First, the excellent `Sklearn  <https://scikit#learn.org/stable/>`__ package already handle NumPy arrays, second,
NumPy is not suitable for large-scale data.


