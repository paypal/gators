# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util


class _BaseFeatureGeneration(Transformer):
    """Base feature generation transformer class.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    column_names : List[str], default to None.
        List of generated columns.
    patterns : List[str]
        List of patterns.
    column_mapping: Dict[str, List[str]]
        Mapping between generated features and base features.

    """

    def __init__(
        self,
        columns: List[str],
        column_names: List[str],
        column_mapping: List[str],
        dtype: type = None,
    ):
        Transformer.__init__(self)
        self.column_names = column_names
        self.columns = columns
        self.column_mapping = column_mapping
        self.idx_columns: np.ndarray = np.array([])
        self.dtype = dtype
