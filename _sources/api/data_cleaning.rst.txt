Data Cleaning
=============

Column Operations
-----------------

* :class:`~gators.data_cleaning.cast_columns.CastColumns` - Cast columns to different data types
* :class:`~gators.data_cleaning.drop_columns.DropColumns` - Drop specified columns
* :class:`~gators.data_cleaning.rename_columns.RenameColumns` - Rename columns

Quality Filters
---------------

* :class:`~gators.data_cleaning.correlation_filter.CorrelationFilter` - Remove highly correlated features
* :class:`~gators.data_cleaning.drop_constant_columns.DropConstantColumns` - Remove constant columns
* :class:`~gators.data_cleaning.drop_duplicate_columns.DropDuplicateColumns` - Remove duplicate columns
* :class:`~gators.data_cleaning.drop_duplicate_rows.DropDuplicateRows` - Remove duplicate rows
* :class:`~gators.data_cleaning.drop_high_nan_ratio.DropHighNaNRatio` - Drop columns with high missing ratio
* :class:`~gators.data_cleaning.drop_low_cardinality.DropLowCardinality` - Drop low cardinality columns
* :class:`~gators.data_cleaning.high_cardinality_filter.HighCardinalityFilter` - Filter high cardinality features
* :class:`~gators.data_cleaning.replace.Replace` - Replace values in columns
* :class:`~gators.data_cleaning.variance_filter.VarianceFilter` - Remove low variance features
