Feature Generation - String
============================

Text Properties
---------------

* :class:`~gators.feature_generation_str.character_statistics.CharacterStatistics` - Character-level statistics
* :class:`~gators.feature_generation_str.length.Length` - String length
* :class:`~gators.feature_generation_str.occurrences.Occurrences` - Count substring occurrences

Text Patterns
-------------

* :class:`~gators.feature_generation_str.contains.Contains` - Check if string contains substring
* :class:`~gators.feature_generation_str.endswith.Endswith` - Check if string ends with pattern
* :class:`~gators.feature_generation_str.pattern_detector.PatternDetector` - Detect regex patterns
* :class:`~gators.feature_generation_str.startswith.Startswith` - Check if string starts with pattern

Text Transformation
-------------------

* :class:`~gators.feature_generation_str.extract_substring.ExtractSubstring` - Extract substring
* :class:`~gators.feature_generation_str.lower.Lower` - Convert to lowercase
* :class:`~gators.feature_generation_str.ngram.NGram` - Generate n-gram
* :class:`~gators.feature_generation_str.split.Split` - Split based on a delimiter
* :class:`~gators.feature_generation_str.split_extract.SplitExtract` - Split based on a delimiter and extract
* :class:`~gators.feature_generation_str.upper.Upper` - Convert to uppercase

Text Interaction
----------------

* :class:`~gators.feature_generation_str.interaction_features.InteractionFeatures` - Exhaustive string interactions
* :class:`~gators.feature_generation_str.combine_features.CombineFeatures` - Selected string interactions
