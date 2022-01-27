# License: Apache-2.0
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SciKitPipeline
from ..transformers.transformer import Transformer

from gators import Series


class Pipeline(SciKitPipeline, Transformer):
    """SciKitPileline Chain the **gators** transformers together.

    Parameters
    ----------
    steps : List[Transformer]
        List of transformations.

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

    def __init__(self, steps: List[Transformer], memory=None, verbose=False):
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

    def predict_numpy(self, X: np.ndarray, y: Series = None) -> np.ndarray:
        """Predict on X, and predict.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : np.ndarray
            Model predictions.
        """
        for step in self.steps[:-1]:
            X = step[1].transform_numpy(X)
        return self.steps[-1][1].predict(X)

    def predict_proba_numpy(self, X: np.ndarray) -> np.ndarray:
        """Predict on X, and return the probability of success.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Model probabilities of success.
        """
        for step in self.steps[:-1]:
            X = step[1].transform_numpy(X)
        return self.steps[-1][1].predict_proba(X)

    def display_encoder_mapping_nb(self, cmap: Union[str, "colormap"], k=5, decimals=2):
        """Display the encoder mapping in a jupyter notebook.

        Parameters
        ----------
        cmap : Union[str, 'colormap']
            Matplotlib colormap.
        k : int, default to 5.
            Number of mappings displayed.  
        decimals : int
            Number of decimal places to use.
        """    
        from IPython.display import display
        encoder_list = [p[1] for p in self.steps if 'Encoder' in str(p[1])]
        if not encoder_list:
            return
        encoder = encoder_list[0]
        describe = encoder.__class__.__name__.replace(
            'Encoder', '').upper().replace('TARGET', 'MEAN TARGET')
        binning_list = [p[1] for p in self.steps if 'Binning' in str(p[1])]
        if binning_list:
            binning_mapping = binning_list[0].mapping
        else:
            binning_mapping = {}
        mapping = pd.DataFrame(encoder.mapping)
        vmin = mapping.min().min()
        vmax = mapping.max().max()
        columns = list((mapping.max()-mapping.min()).sort_values(ascending=False).index)
        for c in columns[:k]:
            values = mapping[[c]]
            values['bins'] = values.index
            if c.replace('__bin', '') in binning_mapping:
                splits = binning_mapping[c.replace('__bin', '')]
                values['bins'] = values['bins'].replace(splits)
            else:
                values = values.sort_values(c)
            values = values.set_index('bins').dropna()
            if decimals:
                values = values.round(decimals)
            else:
                values = values.astype(int)
            values.index.name = f'{c}'
            values.columns = [f'{describe} values']
            display(values.style.background_gradient(cmap=cmap, vmin=vmin, vmax=vmax))
