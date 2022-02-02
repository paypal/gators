# License: Apache-2.0
from abc import ABC
from typing import List

import numpy as np
import pandas as pd

from gators import DataFrame


class FunctionFactory(ABC):
    def fit(self):
        """fit."""

    def predict(self):
        """predict."""

    def predict_proba(self):
        """predict_proba."""

    def head(self):
        """head."""

    def tail(self):
        """tail."""

    def feature_importances_(self):
        """feature_importances_."""

    def shape(self):
        """shape."""

    def to_pandas(self):
        """to_pandas."""

    def to_numpy(self):
        """to_numpy."""

    def concat(self):
        """concat."""

    def nunique(self):
        """nunique."""

    def get_dummies(self):
        """get_dummies."""

    def to_dict(self):
        """to_dict."""

    def most_frequent(self):
        """most_frequent."""

    def fillna(self):
        """fillna."""

    def random_split(self):
        """random_split."""

    def delta_time(self):
        """delta_time."""

    def join(self):
        """join."""

    def replace(self):
        """replace."""


class FunctionPandas(FunctionFactory):
    def set_option(self, option, value):
        pass

    def head(self, X, n, compute=False):
        return X.head(n)

    def tail(self, X, n):
        return X.tail(n)

    def fit(self, model, X, y):
        return model.fit(X.to_numpy(), y.to_numpy())

    def predict(self, model, X):
        return model.predict(X.to_numpy())

    def predict_proba(self, model, X):
        return model.predict_proba(X.to_numpy())

    def feature_importances_(self, model, X, y):
        model.fit(X.to_numpy(), y.to_numpy())
        feature_importances_ = pd.Series(model.feature_importances_, index=X.columns)
        return feature_importances_

    def shape(self, X):
        return X.shape

    def to_pandas(self, X):
        return X

    def to_numpy(self, X):
        return X.to_numpy()

    def concat(self, objs, **kwargs):
        return pd.concat(objs, **kwargs)

    def melt(self, frame, **kwargs):
        return pd.melt(frame, **kwargs)

    def nunique(self, X):
        return X.nunique()

    def get_dummies(self, X, columns, **kwargs):
        return pd.get_dummies(X, columns=columns, **kwargs)

    def to_dict(self, X, **kwargs):
        return X.to_dict(**kwargs)

    def most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().index.to_numpy()[0] for c in columns]
        return dict(zip(columns, values))

    def fillna(self, X, **kwargs):
        return X.fillna(**kwargs)

    def random_split(self, X, frac, random_state=None):
        X_new = X.sample(frac=1.0 - frac, random_state=random_state)
        return X_new, X.drop(X_new.index, axis=0)

    def delta_time(self, X, column_names, columns_a, columns_b, deltatime_dtype):
        for name, c_a, c_b in zip(column_names, columns_a, columns_b):
            X[name] = (X[c_a] - X[c_b]).astype(deltatime_dtype)
        return X

    def join(self, X, other):
        return X.join(other)

    def replace(self, X, replace_dict):
        return X.replace(replace_dict)

    def raise_y_dtype_error(self, y):
        if not isinstance(y, pd.Series):
            raise TypeError("`y` should be a pandas series.")


class FunctionKoalas(FunctionFactory):
    def set_option(self, option, value):
        import databricks.koalas as ks

        ks.set_option(option, value)

    def head(self, X, n, compute=False):
        return X.head(n)

    def tail(self, X, n):
        return X.tail(n)

    def predict(self, model, X):
        from pyspark.ml.feature import VectorAssembler

        columns = X.columns.tolist()
        vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
        X_spark = vector_assembler.transform(X.to_spark())
        return model.transform(X_spark).to_koalas()["prediction"]

    def predict_proba(self, model, X):
        from pyspark.ml.feature import VectorAssembler

        columns = X.columns.tolist()
        vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
        X_spark = vector_assembler.transform(X.to_spark())
        return model.transform(X_spark).to_koalas()["probability"]

    def shape(self, X):
        return X.shape

    def fit(self, model, X, y):
        from pyspark.ml.feature import VectorAssembler

        columns = X.columns.tolist()
        vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
        spark_df = vector_assembler.transform(X.join(y).sort_index().to_spark())
        return model.fit(spark_df)

    def feature_importances_(self, model, X, y):
        from pyspark.ml.feature import VectorAssembler

        columns = list(X.columns)
        vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
        spark_df = vector_assembler.transform(X.join(y).to_spark())
        trained_model = model.fit(spark_df)
        feature_importances_ = pd.Series(
            trained_model.featureImportances.toArray(), index=columns
        )
        return feature_importances_

    def to_pandas(self, X):
        return X.to_pandas()

    def to_numpy(self, X):
        return X.to_numpy()

    def concat(self, objs, **kwargs):
        import databricks.koalas as ks

        return ks.concat(objs, **kwargs)

    def melt(self, frame, **kwargs):
        import databricks.koalas as ks

        return ks.melt(frame, **kwargs)

    def nunique(self, X):
        return X.nunique()

    def get_dummies(self, X, columns, **kwargs):
        import databricks.koalas as ks

        return ks.get_dummies(X, columns=columns, **kwargs)

    def to_dict(self, X, **kwargs):
        return X.to_dict(**kwargs)

    def most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().index.to_numpy()[0] for c in columns]
        return dict(zip(columns, values))

    def fillna(self, X, **kwargs):
        for col, val in kwargs["value"].items():
            if "int" in str(X[col].dtype):
                continue
            X[col] = X[col].fillna(val)
        return X

    def random_split(self, X, frac, random_state=None):
        weights = [1 - frac, frac]
        main_index_name = X.index.name
        index_name = "index" if main_index_name is None else main_index_name
        X[index_name] = X.index
        train, test = X.to_spark().randomSplit(weights, seed=random_state)
        train, test = train.to_koalas().set_index(
            index_name
        ), test.to_koalas().set_index(index_name)
        train.index.name, test.index.name = main_index_name, main_index_name
        return train, test

    def delta_time(self, X, column_names, columns_a, columns_b, deltatime_dtype=None):
        for name, c_a, c_b in zip(column_names, columns_a, columns_b):
            X = X.assign(dummy=(X[c_a].astype(float) - X[c_b].astype(float))).rename(
                columns={"dummy": name}
            )
        return X

    def join(self, X, other):
        import databricks.koalas as ks

        return X.join(other).sort_index()

    def replace(self, X, replace_dict):
        return X.replace(replace_dict)

    def raise_y_dtype_error(self, y):
        import databricks.koalas as ks

        if not isinstance(y, ks.Series):
            raise TypeError("`y` should be a koalas series.")


class FunctionDask(FunctionFactory):
    def set_option(self, option, value):
        pass

    def head(self, X, n, compute=False):
        return X.head(n, compute=compute, npartitions=X.npartitions)

    def tail(self, X, n):
        return X.tail(n, compute=False)

    def fit(self, model, X, y):
        return model.fit(X.values, y.values)

    def predict(self, model, X):
        return model.predict(X.values)

    def predict_proba(self, model, X):
        return model.predict_proba(X.values)

    def shape(self, X):
        X_shape = X.shape
        if len(X_shape) == 1:
            return (
                X_shape[0] if isinstance(X_shape[0], int) else X_shape[0].compute(),
            )
        return (
            X_shape[0] if isinstance(X_shape[0], int) else X_shape[0].compute(),
            X_shape[1] if isinstance(X_shape[1], int) else X_shape[1].compute(),
        )

    def feature_importances_(self, model, X, y):
        model.fit(X, y)
        feature_importances_ = pd.Series(model.feature_importances_, index=X.columns)
        return feature_importances_

    def to_pandas(self, X):
        return X.compute()

    def to_numpy(self, X):
        return X.compute().to_numpy()

    def concat(self, objs, **kwargs):
        import dask.dataframe as dd

        return dd.concat(objs, **kwargs)

    def melt(self, frame, **kwargs):
        import dask.dataframe as dd

        return dd.melt(frame, **kwargs)

    def nunique(self, X):
        if "Series" in str(type(X)):
            return X.nunique().compute()
        return pd.Series([X[c].nunique().compute() for c in X], index=list(X.columns))

    def get_dummies(self, X, columns, **kwargs):
        import dask.dataframe as dd

        X[columns] = X[columns].astype(object).categorize()
        return dd.get_dummies(X, **kwargs)

    def to_dict(self, X, **kwargs):
        return X.compute().to_dict(**kwargs)

    def most_frequent(self, X):
        columns = list(X.columns)
        values = [X[c].value_counts().compute().index[0] for c in columns]
        return dict(zip(columns, values))

    def fillna(self, X, **kwargs):
        return X.fillna(**kwargs)

    def random_split(self, X, frac, random_state=None):
        return X.random_split(frac=[1 - frac, frac], random_state=random_state)

    def delta_time(self, X, column_names, columns_a, columns_b, deltatime_dtype):
        for name, c_a, c_b in zip(column_names, columns_a, columns_b):
            X[name] = (X[c_a] - X[c_b]).astype(deltatime_dtype)
        return X

    def join(self, X, other):
        return X.join(other)

    def replace(self, X, replace_dict):
        def replace_(X, replace_dict):
            return X.replace(replace_dict)

        return X.map_partitions(replace_, replace_dict)

    def raise_y_dtype_error(self, y):
        import dask.dataframe as dd

        if not isinstance(y, dd.Series):
            raise TypeError("`y` should be a dask series.")


def get_function(X):
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": FunctionPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": FunctionKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": FunctionDask(),
        "<class 'pandas.core.series.Series'>": FunctionPandas(),
        "<class 'databricks.koalas.series.Series'>": FunctionKoalas(),
        "<class 'dask.dataframe.core.Series'>": FunctionDask(),
    }
    if str(type(X)) not in factories:
        raise TypeError(
            """`X` should be a pandas, koalas, or dask dataframe and
                        `y` should be a pandas, koalas, or dask series"""
        )
    return factories[str(type(X))]


def get_bounds(X_dtype: type) -> List:
    """Return upper and lower of the input numerical NumPy datatype.

    Parameters
    ----------
    X_dtype : type, default np.float64
        Numerical NumPy datatype.

    Returns
    -------
    List
        Lower ad upper bounds.
    """
    if "float" in str(X_dtype):
        info = np.finfo(X_dtype)
        return info.min, info.max
    elif "int" in str(X_dtype):
        info = np.iinfo(X_dtype)
        return info.min, info.max


def get_datatype_columns(X: DataFrame, datatype: type) -> List[str]:
    """Return the columns of the specified datatype.

    Parameters
    ----------
    X : DataFrame
            Input dataframe.
    datatype : type
        Datatype.

    Returns
    -------
    List[str]
        List of columns
    """
    X_dtypes = X.dtypes
    if datatype != object:
        mask = X_dtypes == datatype
    else:
        mask = ((X_dtypes.astype(str).str.startswith("<U")) | (X_dtypes == object)) | (
            X_dtypes == bool
        )
    datatype_columns = [c for c, m in zip(X_dtypes.index, mask) if m]
    return datatype_columns


def exclude_columns(columns: List[str], excluded_columns: List[str]) -> List[str]:
    """Return the columns in `columns` not in `selected_columns`.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    excluded_theta_vec : List[float]
        List of columns.

    Returns
    -------
    List[str]
        List of columns.
    """
    return [c for c in columns if c not in excluded_columns]


def get_idx_columns(columns: List[str], selected_columns: List[str]) -> np.ndarray:
    """Return the indices of the columns in `columns`
      and `selected_columns`.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    selected_theta_vec : List[float]
        List of columns.

    Returns
    -------
    np.ndarray
        Array of indices.
    """
    selected_idx_columns = []
    for selected_column in selected_columns:
        for i, column in enumerate(columns):
            if column == selected_column:
                selected_idx_columns.append(i)
                break
    return np.array(selected_idx_columns)


def exclude_idx_columns(columns: List[str], excluded_columns: List[str]) -> np.ndarray:
    """Return the indices of the columns in `columns`
        and not in `excluded_columns`.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    excluded_theta_vec : List[float]
        List of columns.

    Returns
    -------
    np.ndarray
        Array of indices.
    """

    selected_idx_columns = [
        i for i, c in enumerate(columns) if c not in excluded_columns
    ]
    return np.array(selected_idx_columns)


def get_numerical_columns(X: DataFrame) -> List[str]:
    """Return the float columns.

    Parameters
    ----------
    X : DataFrame
        Input dataframe.

    Returns
    -------
    List[str]
        List of columns.
    """
    X_dtypes = X.dtypes
    mask = (
        (X_dtypes == np.float64)
        | (X_dtypes == np.int64)
        | (X_dtypes == np.float32)
        | (X_dtypes == np.int32)
        | (X_dtypes == np.float16)
        | (X_dtypes == np.int16)
    )
    numerical_columns = [c for c, m in zip(X_dtypes.index, mask) if m]
    return numerical_columns


def flatten_list(list_to_flatten: List):
    """Flatten list.

    Parameters
    ----------
    list_to_flatten : List
        List to flatten

    Returns
    -------
    List
        Flatten list
    """
    list_flatten = []
    for i, l in enumerate(list_to_flatten):
        if not isinstance(l, list):
            list_flatten.append(l)
        else:
            list_flatten += l
    return list_flatten


def prettify_number(x, precision, infinity):
    if abs(x) >= infinity:
        return x
    if (abs(x) > 1) or (x == 0):
        return round(x, precision)
    exponent = np.floor(np.log10(abs(x)))
    return round(x, int(precision - 1 - exponent))
