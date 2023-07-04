import contextlib
import os
from typing import TypeVar

__version__ = "0.3.4"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

DataFrame = TypeVar("Union[pd.DataFrame, ks.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.Series, ks.Series, dd.Series]")

with contextlib.suppress(ImportError):
    import spark
    import koalas as ks
    import warnings

    prev = spark.conf.get("spark.sql.execution.arrow.enabled")
    ks.set_option("compute.default_index_type", "distributed")
    warnings.filterwarnings("ignore")
