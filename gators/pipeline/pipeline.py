# License: Apache-2.0
from typing import List, Tuple

import numpy as np
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

    >>> from gators.imputers import NumericImputer
    >>> from gators.imputers import ObjectImputer
    >>> from gators.pipeline import Pipeline
    >>> steps = [
    ... ('ObjectImputer', ObjectImputer(strategy='constant', value='MISSING')),
    ... ('NumericImputerMedian', NumericImputer(strategy='median', columns=['A'])),
    ... ('NumericImputerFrequent', NumericImputer(strategy='most_frequent', columns=['B']))]
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
