# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler


def get_bounds(X_dtype: type) -> List:
    """Return upper and lower of the input numerical NumPy datatype.

    Parameters
    ----------
    X_dtype : type, default to np.float64
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


def concat(
    objs: List[Union[pd.DataFrame, ks.DataFrame]], axis: int
) -> Union[pd.DataFrame, ks.DataFrame]:
    """Concatenate the `objs` along an axis.

    Parameters
    ----------
    objs : List[Union[pd.DataFrame, ks.DataFrame]]
        List of dataframes to concatenate.
    axis : int
        The axis to concatenate along.

    Returns
    -------
    Union[pd.DataFrame, ks.DataFrame]
        Concatenated dataframe.
    """
    if isinstance(objs[0], (pd.DataFrame, pd.Series)):
        return pd.concat(objs, axis=axis)
    return ks.concat(objs, axis=axis)


def get_datatype_columns(
    X: Union[pd.DataFrame, ks.DataFrame], datatype: type
) -> List[str]:
    """Return the columns of the specified datatype.

    Parameters
    ----------
    X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
    datatype : type
        Datatype.

    Returns
    -------
    List[str]
        List of columns
    """
    X_dtypes = X.dtypes
    mask = X_dtypes == datatype
    if isinstance(X, ks.DataFrame) and (datatype == object):
        mask = (X_dtypes == object) | (X_dtypes.astype(str).str.startswith("<U"))

    datatype_columns = [c for c, m in zip(X_dtypes.index, mask) if m]
    return datatype_columns


def exclude_columns(columns: List[str], excluded_columns: List[str]) -> List[str]:
    """Return the columns in `columns` not in `selected_columns`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    excluded_columns : List[str]
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
    columns : List[str]
        List of columns.
    selected_columns : List[str]
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
    return np.array(selected_idx_columns)


def exclude_idx_columns(columns: List[str], excluded_columns: List[str]) -> np.ndarray:
    """Return the indices of the columns in `columns`
        and not in `excluded_columns`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    excluded_columns : List[str]
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


def get_numerical_columns(X: Union[pd.DataFrame, ks.DataFrame]) -> List[str]:
    """Return the float columns.

    Parameters
    ----------
    X : Union[pd.DataFrame, ks.DataFrame]
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


def get_float_only_columns(X: Union[pd.DataFrame, ks.DataFrame]) -> List[str]:
    """Return the float columns.

    Parameters
    ----------
    X : Union[pd.DataFrame, ks.DataFrame]
        Input dataframe.

    Returns
    -------
    List[str]
        List of columns.
    """
    numerical_columns = get_numerical_columns(X)
    if not numerical_columns:
        return []
    i_max = int(np.min(np.array([X.shape[0], 50000])))
    X_dummy = X[numerical_columns].head(i_max).to_numpy()
    float_columns = [
        col
        for i, col in enumerate(numerical_columns)
        if not np.allclose(X_dummy[:, i].round(), X_dummy[:, i], equal_nan=True)
        or (X_dummy[:, i] != X_dummy[:, i]).mean() == 1
    ]
    return float_columns


def get_int_only_columns(X: Union[pd.DataFrame, ks.DataFrame]) -> List[str]:
    """Return the list of integer columns.

    Parameters
    ----------
    X : Union[pd.DataFrame, ks.DataFrame]
        Input dataframe.

    Returns
    -------
    List[str]
        List of columns.
    """
    numerical_columns = get_numerical_columns(X)
    if not numerical_columns:
        return []
    i_max = int(np.min(np.array([X.shape[0], 50000])))
    X_dummy = X[numerical_columns].head(i_max).to_numpy()
    int_columns = [
        col
        for i, col in enumerate(numerical_columns)
        if np.allclose(X_dummy[:, i].round(), X_dummy[:, i], equal_nan=True)
        and (X_dummy[:, i] != X_dummy[:, i]).mean() != 1
    ]
    return int_columns


def generate_spark_dataframe(X: ks.DataFrame, y=None):
    """
    Generates a Spark dataframe and transforms the features
    to one column, ready for training in a SparkML model

    Parameters
    ----------
    X : ks.DataFrame
        Feature set.
    y : ks.Series, default to None.
        Target column. Defaults to None.

    Returns
    -------
    pyspark.DataFrame
        Contains the features transformed into one column.
    """
    columns = list(X.columns)
    if y is None:
        spark_df = X.to_spark()
    else:
        spark_df = X.join(y).to_spark()
    vector_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    transformed_spark_df = vector_assembler.transform(spark_df)
    return transformed_spark_df


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
