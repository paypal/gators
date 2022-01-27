import os
from typing import TypeVar

__version__ = "0.2.0"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.Series, ks.Series, dd.Series]")
