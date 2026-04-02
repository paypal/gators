"""Test time_window_features module."""

import polars as pl
import pytest
from datetime import datetime, timedelta
from gators.feature_generation_dt.time_window_features import TimeWindowFeatures


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    base = datetime(2024, 1, 1, 10, 0, 0)
    return pl.DataFrame({
        'TransactionDT': [
            base,
            base + timedelta(minutes=30),
            base + timedelta(hours=1),
            base + timedelta(hours=2),
            base + timedelta(hours=25),
        ],
        'TransactionAmt': [100.0, 150.0, 200.0, 120.0, 300.0],
        'card1': ['C1', 'C1', 'C2', 'C1', 'C1'],
        'card2': ['A', 'A', 'B', 'A', 'A'],
    }).sort('TransactionDT')


@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with null values."""
    base = datetime(2024, 1, 1, 10, 0, 0)
    return pl.DataFrame({
        'TransactionDT': [
            base,
            base + timedelta(minutes=30),
            base + timedelta(hours=1),
            base + timedelta(hours=2),
        ],
        'TransactionAmt': [100.0, None, 200.0, None],
        'card1': ['C1', 'C1', 'C2', 'C1'],
    }).sort('TransactionDT')


class TestTimeWindowFeaturesInit:
    """Test TimeWindowFeatures initialization."""

    def test_valid_initialization(self):
        """Test valid initialization."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h', '24h'],
            func=['count', 'mean'],
        )
        assert transformer.subset == ['TransactionAmt']
        assert transformer.time_column == 'TransactionDT'
        assert transformer.windows == ['1h', '24h']
        assert transformer.func == ['count', 'mean']
        assert transformer.by is None

    def test_with_groupby(self):
        """Test initialization with groupby columns."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            by=['card1', 'card2'],
        )
        assert transformer.by == ['card1', 'card2']

    def test_invalid_aggregation(self):
        """Test that invalid aggregation raises error."""
        with pytest.raises(ValueError, match="is not in the predefined list"):
            TimeWindowFeatures(
                subset=['TransactionAmt'],
                time_column='TransactionDT',
                windows=['1h'],
                func=['invalid'],
            )

    def test_invalid_window_format(self):
        """Test that invalid window format raises error."""
        with pytest.raises(ValueError, match="Invalid window format"):
            TimeWindowFeatures(
                subset=['TransactionAmt'],
                time_column='TransactionDT',
                windows=['1hour'],  # Invalid format
                func=['count'],
            )


class TestTimeWindowFeaturesFit:
    """Test TimeWindowFeatures fit method."""

    def test_fit_creates_column_mapping(self, sample_data):
        """Test that fit creates proper column mapping."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
        )
        transformer.fit(sample_data)
        
        # Check column names are generated
        assert transformer.new_column_names is not None
        assert len(transformer.new_column_names) == 1
        assert 'count_TransactionAmt_1h' in transformer.new_column_names

    def test_fit_multiple_windows_and_aggregations(self, sample_data):
        """Test fit with multiple windows and func."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h', '24h'],
            func=['count', 'sum'],
        )
        transformer.fit(sample_data)
        
        assert len(transformer.new_column_names) == 4

    def test_fit_with_groupby(self, sample_data):
        """Test fit with groupby columns."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            by=['card1'],
        )
        transformer.fit(sample_data)
        
        assert len(transformer.new_column_names) == 1
        assert 'count_TransactionAmt_1h_card1' in transformer.new_column_names

    def test_fit_with_multiple_groupby(self, sample_data):
        """Test fit with multiple groupby columns."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            by=['card1', 'card2'],
        )
        transformer.fit(sample_data)
        
        expected_col = 'count_TransactionAmt_1h_card1_card2'
        assert expected_col in transformer.new_column_names

    def test_fit_with_custom_column_names(self, sample_data):
        """Test fit with custom column names."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            new_column_names=['custom_count'],
        )
        transformer.fit(sample_data)
        
        assert transformer.new_column_names == ['custom_count']

    def test_fit_with_wrong_number_of_custom_names(self, sample_data):
        """Test fit with wrong number of custom names raises error."""
        with pytest.raises(ValueError, match="Length of new_column_names"):
            TimeWindowFeatures(
                subset=['TransactionAmt'],
                time_column='TransactionDT',
                windows=['1h', '24h'],
                func=['count'],
                new_column_names=['only_one'],  # Should be 2
            )


class TestTimeWindowFeaturesTransform:
    """Test TimeWindowFeatures transform method."""

    def test_transform_count_single_window(self, sample_data):
        """Test transform with count aggregation and single window."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'count_TransactionAmt_1h' in result.columns
        # Rows at times: 10:00, 10:30, 11:00, 12:00, 11:00+24h
        # Row 0 (10:00): [09:00, 10:00) → 0
        # Row 1 (10:30): [09:30, 10:30) includes 10:00 → 1
        # Row 2 (11:00): [10:00, 11:00) includes 10:00, 10:30 → 2
        # Row 3 (12:00): [11:00, 12:00) includes 11:00 → 1
        # Row 4 (11:00+24h): [10:00+24h, 11:00+24h) → 0
        expected_counts = [0, 1, 2, 1, 0]
        assert result['count_TransactionAmt_1h'].to_list() == expected_counts

    def test_transform_sum_single_window(self, sample_data):
        """Test transform with sum aggregation."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['sum'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'sum_TransactionAmt_1h' in result.columns
        # Row 0: 0
        # Row 1: 100 (row 0)
        # Row 2: 100+150=250 (rows 0,1)
        # Row 3: 200 (row 2)
        # Row 4: 0
        expected_sums = [0.0, 100.0, 250.0, 200.0, 0.0]
        assert result['sum_TransactionAmt_1h'].to_list() == expected_sums

    def test_transform_mean_single_window(self, sample_data):
        """Test transform with mean aggregation."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['mean'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'mean_TransactionAmt_1h' in result.columns
        # Row 0: null (no history)
        # Row 1: 100.0 (row 0)
        # Row 2: 125.0 (mean of 100, 150)
        # Row 3: 200.0 (row 2)
        # Row 4: null
        expected_means = [None, 100.0, 125.0, 200.0, None]
        assert result['mean_TransactionAmt_1h'].to_list() == expected_means

    def test_transform_multiple_aggregations(self, sample_data):
        """Test transform with multiple func."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count', 'sum', 'mean'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'count_TransactionAmt_1h' in result.columns
        assert 'sum_TransactionAmt_1h' in result.columns
        assert 'mean_TransactionAmt_1h' in result.columns

    def test_transform_multiple_windows(self, sample_data):
        """Test transform with multiple windows."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h', '24h'],
            func=['count'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'count_TransactionAmt_1h' in result.columns
        assert 'count_TransactionAmt_24h' in result.columns
        
        # 24h window captures more transactions
        # Row 0 (day1 10:00): 0
        # Row 1 (day1 10:30): 1 (row0)
        # Row 2 (day1 11:00): 2 (rows 0,1)
        # Row 3 (day1 12:00): 3 (rows 0,1,2)
        # Row 4 (day2 11:00): 2 (rows 2,3 from previous day)
        expected_24h = [0, 1, 2, 3, 2]
        assert result['count_TransactionAmt_24h'].to_list() == expected_24h

    def test_transform_with_groupby(self, sample_data):
        """Test transform with groupby columns."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            by=['card1'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'count_TransactionAmt_1h_card1' in result.columns
        # Group C1: rows 0, 1, 3, 4 (times: 10:00, 10:30, 12:00, 11:00+24h)
        # Group C2: row 2 (time: 11:00)
        # Row 0 (C1, 10:00): 0
        # Row 1 (C1, 10:30): 1 (row 0 in window)
        # Row 2 (C2, 11:00): 0 (different group)
        # Row 3 (C1, 12:00): 0 (no C1 transactions in last hour)
        # Row 4 (C1, 11:00+24h): 0 (no C1 transactions in last hour)
        expected = [0, 1, 0, 0, 0]
        assert result['count_TransactionAmt_1h_card1'].to_list() == expected

    def test_transform_groupby_all_aggregations(self, sample_data):
        """Test transform with groupby and all func to ensure coverage."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count', 'sum', 'mean', 'std', 'median', 'min', 'max'],
            by=['card1'],
        )
        result = transformer.fit_transform(sample_data)
        
        # Verify all grouped aggregation columns exist
        assert 'count_TransactionAmt_1h_card1' in result.columns
        assert 'sum_TransactionAmt_1h_card1' in result.columns
        assert 'mean_TransactionAmt_1h_card1' in result.columns
        assert 'std_TransactionAmt_1h_card1' in result.columns
        assert 'median_TransactionAmt_1h_card1' in result.columns
        assert 'min_TransactionAmt_1h_card1' in result.columns
        assert 'max_TransactionAmt_1h_card1' in result.columns

    def test_transform_multiple_numerical_columns(self, sample_data):
        """Test transform with multiple numerical columns."""
        # Add another numerical column
        X = sample_data.with_columns((pl.col('TransactionAmt') * 2).alias('DoubleAmt'))
        
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt', 'DoubleAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        assert 'count_TransactionAmt_1h' in result.columns
        assert 'count_DoubleAmt_1h' in result.columns
        # Counts should be the same for both columns
        assert result['count_TransactionAmt_1h'].to_list() == result['count_DoubleAmt_1h'].to_list()

    def test_transform_with_nulls(self, sample_data_with_nulls):
        """Test transform handles null values correctly."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count', 'sum'],
        )
        result = transformer.fit_transform(sample_data_with_nulls)
        
        # Data: 100, null, 200, null at times 10:00, 10:30, 11:00, 12:00
        # Count counts ALL rows in window (including ones with null values)
        # This is correct for velocity features: "how many transactions in last hour"
        # Row 0 (10:00): 0
        # Row 1 (10:30): 1 (row0)
        # Row 2 (11:00): 2 (row0, row1 - counts both even though row1 is null)
        # Row 3 (12:00): 1 (row2)
        expected_counts = [0, 1, 2, 1]
        assert result['count_TransactionAmt_1h'].to_list() == expected_counts
        
        # Sum aggregates only non-null values
        expected_sums = [0.0, 100.0, 100.0, 200.0]
        assert result['sum_TransactionAmt_1h'].to_list() == expected_sums

    def test_transform_std_aggregation(self, sample_data):
        """Test transform with std aggregation."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['24h'],
            func=['std'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'std_TransactionAmt_24h' in result.columns
        # First row: null (no history)
        # Second row: null (only 1 value, std undefined)
        # Third row and beyond: should have values
        assert result['std_TransactionAmt_24h'][0] is None
        assert result['std_TransactionAmt_24h'][1] is None

    def test_transform_median_aggregation(self, sample_data):
        """Test transform with median aggregation."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['24h'],
            func=['median'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'median_TransactionAmt_24h' in result.columns

    def test_transform_min_max_aggregations(self, sample_data):
        """Test transform with min and max func."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['24h'],
            func=['min', 'max'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert 'min_TransactionAmt_24h' in result.columns
        assert 'max_TransactionAmt_24h' in result.columns

    def test_transform_all_aggregations(self, sample_data):
        """Test transform with all supported func."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['24h'],
            func=['count', 'sum', 'mean', 'std', 'median', 'min', 'max'],
        )
        result = transformer.fit_transform(sample_data)
        
        assert len(result.columns) == 11  # Original 4 + 7 new features
        assert 'count_TransactionAmt_24h' in result.columns
        assert 'sum_TransactionAmt_24h' in result.columns
        assert 'mean_TransactionAmt_24h' in result.columns
        assert 'std_TransactionAmt_24h' in result.columns
        assert 'median_TransactionAmt_24h' in result.columns
        assert 'min_TransactionAmt_24h' in result.columns
        assert 'max_TransactionAmt_24h' in result.columns

    def test_transform_30_minute_window(self):
        """Test transform with 30-minute window."""
        base = datetime(2024, 1, 1, 10, 0, 0)
        X = pl.DataFrame({
            'dt': [
                base,
                base + timedelta(minutes=20),
                base + timedelta(minutes=40),
            ],
            'amt': [100, 200, 300],
        }).sort('dt')
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['30m'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        # Row 0 (10:00): [09:30, 10:00) → 0
        # Row 1 (10:20): [09:50, 10:20) includes 10:00 → 1
        # Row 2 (10:40): [10:10, 10:40) includes 10:20 → 1
        expected = [0, 1, 1]
        assert result['count_amt_30m'].to_list() == expected

    def test_transform_7_day_window(self):
        """Test transform with 7-day window."""
        base = datetime(2024, 1, 1)
        X = pl.DataFrame({
            'dt': [
                base,
                base + timedelta(days=3),
                base + timedelta(days=6),
                base + timedelta(days=8),
            ],
            'amt': [100, 200, 300, 400],
        }).sort('dt')
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['7d'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        # Row 0 (day 0): [day -7, day 0) → 0
        # Row 1 (day 3): [day -4, day 3) includes day0 → 1
        # Row 2 (day 6): [day -1, day 6) includes day0, day3 → 2
        # Row 3 (day 8): [day 1, day 8) includes day3, day6 (day0 outside) → 2
        expected = [0, 1, 2, 2]
        assert result['count_amt_7d'].to_list() == expected

    def test_transform_1_month_window(self):
        """Test transform with 1-month window (converted to 30 days)."""
        base = datetime(2024, 1, 1)
        X = pl.DataFrame({
            'dt': [
                base,
                base + timedelta(days=15),
                base + timedelta(days=29),
                base + timedelta(days=31),
            ],
            'amt': [100, 200, 300, 400],
        }).sort('dt')
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['1M'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        # Row 0 (day 0): 0
        # Row 1 (day 15): 1
        # Row 2 (day 29): 2
        # Row 3 (day 31): 2 (day 0 is outside 30-day window, days 15 and 29 are in)
        expected = [0, 1, 2, 2]
        assert result['count_amt_1M'].to_list() == expected

    def test_transform_1_year_window(self):
        """Test transform with 1-year window (converted to 365 days)."""
        base = datetime(2024, 1, 1)
        X = pl.DataFrame({
            'dt': [
                base,
                base + timedelta(days=180),
                base + timedelta(days=364),
                base + timedelta(days=366),
            ],
            'amt': [100, 200, 300, 400],
        }).sort('dt')
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['1Y'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        # Row 0 (day 0): 0
        # Row 1 (day 180): 1
        # Row 2 (day 364): 2
        # Row 3 (day 366): 2 (day 0 is outside 365-day window, days 180 and 364 are in)
        expected = [0, 1, 2, 2]
        assert result['count_amt_1Y'].to_list() == expected

    def test_transform_drop_columns(self, sample_data):
        """Test transform with drop_columns option."""
        transformer = TimeWindowFeatures(
            subset=['TransactionAmt'],
            time_column='TransactionDT',
            windows=['1h'],
            func=['count'],
            drop_columns=True,
        )
        result = transformer.fit_transform(sample_data)
        
        # TransactionAmt should be dropped
        assert 'TransactionAmt' not in result.columns
        assert 'count_TransactionAmt_1h' in result.columns
        # Other columns should remain
        assert 'TransactionDT' in result.columns
        assert 'card1' in result.columns

    def test_transform_single_row(self):
        """Test transform with single row returns zero counts."""
        X = pl.DataFrame({
            'dt': [datetime(2024, 1, 1)],
            'amt': [100],
        })
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['1h'],
            func=['count', 'sum'],
        )
        result = transformer.fit_transform(X)
        
        assert result['count_amt_1h'][0] == 0
        assert result['sum_amt_1h'][0] == 0.0

    def test_transform_empty_dataframe(self):
        """Test transform with empty dataframe."""
        X = pl.DataFrame({
            'dt': [],
            'amt': [],
        }).with_columns([
            pl.col('dt').cast(pl.Datetime),
            pl.col('amt').cast(pl.Float64),
        ])
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['1h'],
            func=['count'],
        )
        result = transformer.fit_transform(X)
        
        assert result.height == 0
        assert 'count_amt_1h' in result.columns

    def test_transform_maintains_original_order(self):
        """Test that transform requires sorted time column."""
        base = datetime(2024, 1, 1, 10, 0, 0)
        # Create unsorted data
        X = pl.DataFrame({
            'dt': [
                base + timedelta(hours=2),
                base,
                base + timedelta(hours=1),
            ],
            'amt': [300, 100, 200],
            'id': [3, 1, 2],
        })
        
        transformer = TimeWindowFeatures(
            subset=['amt'],
            time_column='dt',
            windows=['1h'],
            func=['count'],
        )
        
        # Should raise error on unsorted data
        with pytest.raises(Exception, match="not sorted"):
            result = transformer.fit_transform(X)
