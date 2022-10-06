import numpy as np
from gators import DataFrame, Series
from . import util


def compute_iv(X: DataFrame, y: Series, regularization=0.1):
    y_name = y.name
    object_columns = util.get_datatype_columns(X, object)
    stats = (
        util.get_function(X)
        .melt(X[object_columns].join(y), id_vars=y_name)
        .groupby(["variable", "value"])
        .agg(["sum", "count"])[y_name]
    )
    stats = util.get_function(X).to_pandas(stats)
    stats = stats.rename(columns={"sum": "1"})
    stats["0"] = stats["count"] - stats["1"]
    stats = stats.drop("count", axis=1)
    stats = stats[["0", "1"]]
    stats[["distrib_0", "distrib_1"]] = (stats[["0", "1"]] + regularization) / (
        stats[["0", "1"]].groupby("variable").sum() + 2 * regularization
    )
    stats["woe"] = np.log(stats["distrib_1"] / stats["distrib_0"])
    iv = (stats["distrib_1"] - stats["distrib_0"]) * stats["woe"]
    iv = iv.groupby("variable").sum()
    iv.name = "iv"
    iv.index.name = None
    return iv, stats
