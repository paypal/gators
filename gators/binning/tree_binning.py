# License: Apache-2.0
from typing import Union

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..util import util
from ._base_binning import _BaseBinning


from gators import DataFrame


class TreeBinning(_BaseBinning):
    """Bin the columns using decision tree based splits.

    The binning can be done inplace or by adding the binned
    columns to the existing data.

    Parameters
    ----------
    tree : 'DecisionTree'
        Decision tree model used to create the bin intervals.
    inplace : bool, default False
        If False, return the dataframe with the new binned columns
        with the names "column_name__bin"). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    ---------
    >>> from gators.binning import TreeBinning
    >>> from sklearn.tree import DecisionTreeClassifier

    The binning can be done inplace by modifying the existing columns

    >>> obj = TreeBinning(tree=DecisionTreeClassifier(max_depth=2, random_state=0), inplace=True)

    or by adding new binned columns

    >>> obj = TreeBinning(tree=DecisionTreeClassifier(max_depth=2, random_state=0), inplace=True)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([0, 1, 0, 1], name="TARGET"), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> y = ks.Series([0, 1, 0, 1], name="TARGET")

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> y = pd.Series([0, 1, 0, 1], name="TARGET")

    The result is a transformed dataframe belonging to the same dataframe library.

    * with `inplace=True`

    >>> obj = TreeBinning(tree=DecisionTreeClassifier(max_depth=2, random_state=0), inplace=True)
    >>> obj.fit_transform(X, y)
                   A              B              C
    0   [-2.06, 1.4)  (-inf, -0.25)  (-inf, -1.05)
    1  (-inf, -2.06)   [-0.25, inf)    [0.07, inf)
    2   [-2.06, 1.4)  (-inf, -0.25)    [0.07, inf)
    3     [1.4, inf)   [-0.25, inf)  [-1.05, 0.07)

    * with `inplace=False`

    >>> X = pd.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> obj = TreeBinning(tree=DecisionTreeClassifier(max_depth=2, random_state=0), inplace=False)
    >>> obj.fit_transform(X, y)
          A     B     C         A__bin         B__bin         C__bin
    0  1.07 -1.19 -1.15   [-2.06, 1.4)  (-inf, -0.25)  (-inf, -1.05)
    1 -2.59 -0.22  1.92  (-inf, -2.06)   [-0.25, inf)    [0.07, inf)
    2 -1.54 -0.28  1.09   [-2.06, 1.4)  (-inf, -0.25)    [0.07, inf)
    3  1.72  1.28 -0.95     [1.4, inf)   [-0.25, inf)  [-1.05, 0.07)

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({
    ... 'A': [1.07, -2.59, -1.54, 1.72],
    ... 'B': [-1.19, -0.22, -0.28, 1.28],
    ... 'C': [-1.15, 1.92, 1.09, -0.95]})
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.07, -1.19, -1.15, '[-2.06, 1.4)', '(-inf, -0.25)',
            '(-inf, -1.05)'],
           [-2.59, -0.22, 1.92, '(-inf, -2.06)', '[-0.25, inf)',
            '[0.07, inf)'],
           [-1.54, -0.28, 1.09, '[-2.06, 1.4)', '(-inf, -0.25)',
            '[0.07, inf)'],
           [1.72, 1.28, -0.95, '[1.4, inf)', '[-0.25, inf)', '[-1.05, 0.07)']],
          dtype=object)

    See Also
    --------
    gators.binning.CustomBinning
        Bin using user input splits.
    gators.binning.Binning
        Bin using equal splits.
    gators.binning.CustomBinning
        Bin using the variable quantiles.

    """

    def __init__(
        self, tree: Union[DecisionTreeClassifier, DecisionTreeRegressor], inplace=False
    ):
        _BaseBinning.__init__(self, n_bins=1, inplace=inplace)
        self.tree = tree

    def fit(self, X: DataFrame, y) -> "TreeBinning":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        "TreeBinning"
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        self.base_columns = list(X.columns)
        self.columns = util.get_numerical_columns(X)
        self.column_names = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        n_cols = self.idx_columns.size
        if n_cols == 0:
            return self
        self.bins = {}
        for c in self.columns:
            self.tree.fit(
                util.get_function(X).to_numpy(X[[c]]).astype(np.float32),
                util.get_function(X).to_numpy(y).astype(np.int32),
            )

            splits = sorted(
                [
                    float(node.split("<=")[1])
                    for node in tree.export_text(self.tree, decimals=6).split("|   ")
                    if "<=" in node
                ]
            )
            self.bins[c] = (
                np.unique(
                    [-np.inf]
                    + [util.prettify_number(x, precision=2) for x in splits]
                    + [np.inf]
                )
                if len(splits)
                else np.array([-np.inf, np.inf])
            )
        max_bins = max([len(v) for v in self.bins.values()])
        self.bins_np = np.inf * np.ones((max_bins, n_cols))
        for i, b in enumerate(self.bins.values()):
            self.bins_np[: len(b), i] = b
        self.bins_dict = {col: np.unique(self.bins[col]) for col in self.bins.keys()}
        self.pretty_bins_dict = {
            k: [util.prettify_number(x, precision=2) for x in v]
            for k, v in self.bins_dict.items()
        }
        self.labels, self.labels_np = self.get_labels(self.pretty_bins_dict)
        return self
