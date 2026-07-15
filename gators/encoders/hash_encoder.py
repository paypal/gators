from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class HashEncoder(_BaseTransformer):
    """Encode categorical features via feature hashing (hashing trick).

    Maps each category value to an integer bucket in ``[0, n_features)`` using
    a deterministic hash function.  Because no vocabulary is learnt during
    ``fit()``, the encoder handles **unknown categories at inference time
    naturally** — any unseen value simply hashes to some bucket without
    raising an error.

    Each input column produces one output column containing integers in
    ``[0, n_features)``.  The output column name is ``{col}__hash``.

    Parameters
    ----------
    n_features : int, default=16
        Number of hash buckets.  Controls the trade-off between collision
        rate (lower → more collisions) and the number of distinct encodings
        (higher → fewer collisions but larger range).  Must be ≥ 2.
    subset : list[str] or None, default=None
        Categorical (String, Categorical, Enum, Boolean) columns to encode.
        If ``None``, all such columns are selected automatically.
    inplace : bool, default=True
        If ``True`` the original columns are overwritten with hash values
        (cast to ``Float64`` for consistency with other encoders).
        If ``False`` new columns suffixed ``__hash`` are added alongside the
        originals (subject to ``drop_columns``).
    drop_columns : bool, default=True
        When ``inplace=False``, whether to drop the original columns after
        adding the hashed columns.  Ignored when ``inplace=True``.

    Notes
    -----
    The hash is computed with :meth:`polars.Expr.hash` using ``seed=0`` for
    full determinism across processes (unlike Python's built-in ``hash()``
    which is randomised by ``PYTHONHASHSEED``).  Boolean columns are cast to
    ``String`` before hashing so that ``True`` / ``False`` map to stable
    buckets.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.encoders import HashEncoder

    >>> X = pl.DataFrame({
    ...     "color":  ["red", "blue", "green", "red", "blue"],
    ...     "size":   ["S", "M", "L", "XL", "S"],
    ...     "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
    ... })
    >>> encoder = HashEncoder(n_features=8, inplace=False)
    >>> encoder.fit(X)
    >>> X_enc = encoder.transform(X)
    """

    n_features: Annotated[int, Field(ge=2)] = 16
    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True

    _CAT_DTYPES = {pl.String, pl.Categorical, pl.Enum}
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "HashEncoder":
        """Detect the subset of categorical columns (no statistics are learnt).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Not used; present for sklearn compatibility.

        Returns
        -------
        HashEncoder
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype.base_type() in self._CAT_DTYPES
            ]

        self._column_mapping = {col: f"{col}__hash" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical columns using the hashing trick.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with hash-encoded columns.
        """
        dtypes = dict(zip(X.columns, X.dtypes))

        def _hash_expr(col: str) -> pl.Expr:
            base = (
                pl.col(col).cast(pl.String)
                if dtypes.get(col) == pl.Boolean
                else pl.col(col).cast(pl.String)
            )
            hashed = (base.hash(seed=0) % self.n_features).cast(pl.Float64)
            return hashed

        if self.inplace:
            return X.with_columns([_hash_expr(col) for col in self.subset])

        new_col_exprs = [_hash_expr(col).alias(self._column_mapping[col]) for col in self.subset]
        X = X.with_columns(new_col_exprs)

        if self.drop_columns and self.subset:
            return X.drop(self.subset)
        return X
