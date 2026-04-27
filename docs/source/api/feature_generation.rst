Feature Generation - Numeric
=============================

Mathematical Operations
-----------------------

* :class:`~gators.feature_generation.comparison_features.ComparisonFeatures` - Compare columns
* :class:`~gators.feature_generation.distance_features.DistanceFeatures` - Calculate distance between columns
* :class:`~gators.feature_generation.math_features.MathFeatures` - Mathematical operations between columns
* :class:`~gators.feature_generation.ratio_features.RatioFeatures` - Calculate ratio between columns
* :class:`~gators.feature_generation.plan_rotation_features.PlanRotationFeatures` - Rotate features in space
* :class:`~gators.feature_generation.polynomial_features.PolynomialFeatures` - Generate polynomial features
* :class:`~gators.feature_generation.scalar_math_features.ScalarMathFeatures` - Scalar mathematical operations

Conditions and Rules
--------------------

* :class:`~gators.feature_generation.condition_features.ConditionFeatures` - Condition-based features
* :class:`~gators.feature_generation.is_null.IsNull` - Generate null indicator features
* :class:`~gators.feature_generation.rule_features.RuleFeatures` - Apply business rules

Group Features
--------------

* :class:`~gators.feature_generation.group_statistics_features.GroupStatisticsFeatures` - Generate group-based statistics
* :class:`~gators.feature_generation.group_scaling_features.GroupScalingFeatures` - Generate group-based scaling features
* :class:`~gators.feature_generation.group_lag_features.GroupLagFeatures` - Generate group-based lag features
* :class:`~gators.feature_generation.row_group_features.RowStatisticsFeatures` - Generate group-based statistics per row
