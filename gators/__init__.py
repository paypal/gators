import os
from typing import TypeVar

__version__ = "0.3.1"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.Series, ks.Series, dd.Series]")
