# License: Apache-2.0
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline as SciKitPipeline
from ..transformers.transformer import Transformer

from gators import Series


class Pipeline(SciKitPipeline, Transformer):
    """SciKitPileline Chain the **gators** transformers together.

    Parameters
    ----------
    steps : List[Tuple[Transformer]]
        List of transformers ending, or not, by an estimator.

    memory : str, default None.
        Scilit-learn pipeline `memory` parameter.

    verbose : bool, default False.
        Verbosity.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.imputers import NumericsImputer
    >>> from gators.imputers import ObjectImputer
    >>> from gators.pipeline import Pipeline
    >>> steps = [
    ... ('ObjectImputer', ObjectImputer(strategy='constant', value='MISSING')),
    ... ('NumericsImputerMedian', NumericsImputer(strategy='median', columns=['A'])),
    ... ('NumericsImputerFrequent', NumericsImputer(strategy='most_frequent', columns=['B']))]
    >>> obj = Pipeline(steps=steps)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [0.1, 0.2, 0.3, np.nan],
    ... 'B': [1, 2, 2, np.nan],
    ... 'C': ['a', 'b', 'c', None]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [0.1, 0.2, 0.3, np.nan],
    ... 'B': [1, 2, 2, np.nan],
    ... 'C': ['a', 'b', 'c', None]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [0.1, 0.2, 0.3, np.nan],
    ... 'B': [1, 2, 2, np.nan],
    ... 'C': ['a', 'b', 'c', None]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B        C
    0  0.1  1.0        a
    1  0.2  2.0        b
    2  0.3  2.0        c
    3  0.2  2.0  MISSING

    >>> X = pd.DataFrame({
    ... 'A': [0.1, 0.2, 0.3, np.nan],
    ... 'B': [1, 2, 2, np.nan],
    ... 'C': ['a', 'b', 'c', None]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.1, 1.0, 'a'],
           [0.2, 2.0, 'b'],
           [0.3, 2.0, 'c'],
           [0.2, 2.0, 'MISSING']], dtype=object)
    """

    def __init__(self, steps: List[Tuple[Transformer]], memory=None, verbose=False):
        if not isinstance(steps, (list, np.ndarray)):
            raise TypeError("`steps` should be a list.")
        if not steps:
            raise TypeError("`steps` should not be an empty list.")
        self.steps = steps
        self.is_model = (hasattr(self.steps[-1][1], "predict")) or (
            "pyspark.ml" in str(type(self.steps[-1][1]))
        )
        SciKitPipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)
        self.n_steps = len(self.steps)
        self.n_transformations = self.n_steps - 1 if self.is_model else self.n_steps

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        for step in self.steps[: self.n_transformations]:
            X = step[1].transform_numpy(X)
        return X

    def display_encoder_mapping(
        self, cmap: Union[str, "colormap"], k=5, decimals=2, describe: str = ""
    ):
        """Display the encoder mapping in a jupyter notebook.

        Parameters
        ----------
        cmap : Union[str, 'colormap']
            Matplotlib colormap.
        k : int, default 5.
            Number of mappings displayed.
        decimals : int, default 2.
            Number of decimal places to use.
        describe : str, default ''.
            Name of the encoded values.
        """
        encoder_list = [p[1] for p in self.steps if "Encoder" in str(p[1])]
        figs = []
        if not encoder_list:
            return
        encoder = encoder_list[0]
        binning_list = [p[1] for p in self.steps if "Binning" in str(p[1])]
        if binning_list:
            binning_mapping = binning_list[0].mapping
        else:
            binning_mapping = {}
        mapping = pd.DataFrame(encoder.mapping)
        columns = list(
            (mapping.max() - mapping.min()).sort_values(ascending=False).index
        )
        for c in columns[:k]:
            values = mapping[[c]]
            values["bins"] = values.index
            if c.replace("__bin", "") in binning_mapping:
                splits = binning_mapping[c.replace("__bin", "")]
                new_splits = {}
                for k, v in splits.items():
                    s1, s2 = v.split(",")
                    if "-inf" not in s1:
                        v = v.replace(s1[1:], str(round(float(s1[1:]), 2)))
                    if "inf" not in s2:
                        v = v.replace(s2[:-1], str(round(float(s2[:-1]), 2)))
                    new_splits[k] = v.replace(",", ", ")
                values["bins"] = values["bins"].replace(new_splits)
            else:
                values = values.sort_values(c, ascending=False)
            values = values.set_index("bins").dropna()
            if decimals:
                values = values.round(decimals)
            else:
                values = values.astype(int)
            values.columns = [""]
            x, y = 0.6 * len(values) / 1.62, 0.6 * len(values)
            fig, ax = plt.subplots(figsize=(x, y))
            _ = sns.heatmap(values, ax=ax, cmap=cmap, annot=True, cbar=False)
            _ = ax.set_title(describe)
            _ = ax.set_ylabel(None)
            _ = ax.set_ylabel(c)
            figs.append(fig)
        return figs
