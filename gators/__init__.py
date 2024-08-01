import contextlib
import os
from typing import TypeVar

__version__ = "0.3.4"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

DataFrame = TypeVar("Union[pd.DataFrame, ps.DataFrame, dd.DataFrame]")
Series = TypeVar("Union[pd.Series, ps.Series, dd.Series]")

with contextlib.suppress(ImportError):
    import spark
    import pyspark.pandas as ps
    import warnings

    prev = spark.conf.get("spark.sql.execution.arrow.enabled")
    ps.set_option("compute.default_index_type", "distributed")
    warnings.filterwarnings("ignore")
