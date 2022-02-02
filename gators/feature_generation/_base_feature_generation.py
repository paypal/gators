# License: Apache-2.0
from typing import List

import numpy as np

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class _BaseFeatureGeneration(Transformer):
    """Base feature generation transformer class.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    column_names : List[str], default None.
        List of generated columns.
    patterns : List[str]
        List of patterns.
    """

    def __init__(
        self,
        columns: List[str],
        column_names: List[str],
    ):
        Transformer.__init__(self)
        self.column_names = column_names
        self.columns = columns
        self.idx_columns: np.ndarray = np.array([])
