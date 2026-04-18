"""Tests for data_pipeline.base_preprocessor.BasePreProcessor."""
import logging

import pandas as pd
import pytest

from data_pipeline.base_preprocessor import BasePreProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(df: pd.DataFrame, config: dict) -> BasePreProcessor:
    return BasePreProcessor(df, config)


def _simple_df(**cols) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    return pd.DataFrame(cols)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Provide a simple default DataFrame fixture."""
    return _simple_df()


@pytest.fixture
def make_processor():
    """Provide a factory fixture to build BasePreProcessor instances."""

    def _factory(df: pd.DataFrame, config: dict) -> BasePreProcessor:
        return _make(df, config)

    return _factory


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for BasePreProcessor initialization behavior."""

    def test_stores_df_and_config(self, simple_df, make_processor):
        """Store incoming DataFrame and config on construction."""
        cfg = {"drop_duplicates": False}
        p = make_processor(simple_df, cfg)
        assert p.df is simple_df
        assert p.config is cfg


# ---------------------------------------------------------------------------
# requires_config decorator
# ---------------------------------------------------------------------------


class TestRequiresConfig:
    """Tests for methods guarded by requires_config."""

    def test_skips_method_when_key_absent(self, simple_df, make_processor):
        """Skip decorated methods when config key is missing."""
        p = make_processor(simple_df, {})
        original_cols = list(simple_df.columns)
        p._drop_unused_columns()  # skipped – no 'use_cols' key
        assert list(p.df.columns) == original_cols

    def test_executes_method_when_key_present(self):
        """Execute decorated methods when config key exists."""
        df = _simple_df()
        p = _make(df, {"use_cols": ["a"]})
        # _lowercase_col_names turns A/B into a/b, so use_cols keeps "a".
        p._lowercase_col_names()
        p._drop_unused_columns()
        assert list(p.df.columns) == ["a"]


# ---------------------------------------------------------------------------
# preprocess / _basic_preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for the high-level preprocess workflow."""

    def test_preprocess_returns_dataframe(self, simple_df, make_processor):
        """Return a DataFrame after running preprocess."""
        p = make_processor(simple_df, {})
        result = p.preprocess()
        assert isinstance(result, pd.DataFrame)

    def test_preprocess_calls_custom(self):
        """Ensure _custom_preprocessing is called (override to verify)."""
        called = []

        class Custom(BasePreProcessor):
            def _custom_preprocessing(self):
                called.append(True)

        p = Custom(_simple_df(), {})
        p.preprocess()
        assert called == [True]


# ---------------------------------------------------------------------------
# _lowercase_col_names
# ---------------------------------------------------------------------------


class TestLowercaseColNames:
    """Tests for lowercase column normalization."""

    def test_columns_lowercased(self):
        """Convert mixed-case column names to lowercase."""
        df = pd.DataFrame({"FOO": [1], "Bar": [2]})
        p = _make(df, {})
        p._lowercase_col_names()
        assert list(p.df.columns) == ["foo", "bar"]


# ---------------------------------------------------------------------------
# _drop_unused_columns
# ---------------------------------------------------------------------------


class TestDropUnusedColumns:
    """Tests for configured column selection."""

    def test_keeps_only_use_cols(self):
        """Keep only columns listed in use_cols."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        p = _make(df, {"use_cols": ["a", "b"]})
        p._drop_unused_columns()
        assert set(p.df.columns) == {"a", "b"}

    def test_missing_cols_logged_as_warning(self, caplog):
        """Log a warning when configured columns are missing."""
        df = pd.DataFrame({"a": [1]})
        p = _make(df, {"use_cols": ["a", "missing"]})
        with caplog.at_level(logging.WARNING):
            p._drop_unused_columns()
        assert "use_cols" in caplog.text

    def test_case_insensitive_match(self):
        """Match use_cols values case-insensitively."""
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        p = _make(df, {"use_cols": ["FOO"]})
        p._drop_unused_columns()
        assert "foo" in p.df.columns


# ---------------------------------------------------------------------------
# _fill_missing_values
# ---------------------------------------------------------------------------


class TestFillMissingValues:
    """Tests for fillna-based missing value handling."""

    def test_fills_nan_with_value(self):
        """Replace NaN values using configured fillna mapping."""
        df = pd.DataFrame({"summary": [None, "ok"]})
        p = _make(df, {"fillna": {"summary": ""}})
        p._fill_missing_values()
        assert p.df["summary"].iloc[0] == ""

    def test_missing_col_warns(self, caplog):
        """Log a warning when fillna references a missing column."""
        df = pd.DataFrame({"a": [1]})
        p = _make(df, {"fillna": {"nonexistent": 0}})
        with caplog.at_level(logging.WARNING):
            p._fill_missing_values()
        assert "fillna" in caplog.text


# ---------------------------------------------------------------------------
# _preprocess_string_cols
# ---------------------------------------------------------------------------


class TestPreprocessStringCols:
    """Tests for regex-driven string preprocessing behavior."""

    def _cfg(self, pattern, col, pattern_name):
        """Build a minimal string_cols config payload."""
        return {
            "string_cols": {
                "patterns": {pattern_name: pattern},
                "columns": {col: [pattern_name]},
            }
        }

    def test_applies_regex_to_column(self):
        """Apply configured regex cleanup to target text columns."""
        df = pd.DataFrame({"text": ["hello <b>world</b>"]})
        p = _make(df, self._cfg("<[^>]+>", "text", "strip_html"))
        p._preprocess_string_cols()
        assert p.df["text"].iloc[0] == "hello world"

    def test_nan_values_preserved(self):
        """Preserve NaN values when cleaning string columns."""
        df = pd.DataFrame({"text": [None, "ok"]})
        p = _make(df, self._cfg("<[^>]+>", "text", "strip_html"))
        p._preprocess_string_cols()
        assert pd.isna(p.df["text"].iloc[0])

    def test_missing_column_warns(self, caplog):
        """Log a warning when string_cols references a missing column."""
        df = pd.DataFrame({"other": ["x"]})
        p = _make(df, self._cfg("<b>", "text", "tag"))
        with caplog.at_level(logging.WARNING):
            p._preprocess_string_cols()
        assert "string_cols" in caplog.text

    def test_missing_pattern_warns(self, caplog):
        """Log a warning when a referenced pattern name is not defined."""
        df = pd.DataFrame({"text": ["hello"]})
        cfg = {
            "string_cols": {
                "patterns": {"known": "x"},
                "columns": {"text": ["nonexistent_pattern"]},
            }
        }
        p = _make(df, cfg)
        with caplog.at_level(logging.WARNING):
            p._preprocess_string_cols()
        assert "nonexistent_pattern" in caplog.text

    def test_missing_patterns_config_returns_early(self, caplog):
        """Return early when string pattern config is missing."""
        df = pd.DataFrame({"text": ["hello"]})
        cfg = {"string_cols": {"columns": {"text": ["p"]}}}  # no 'patterns'
        p = _make(df, cfg)
        with caplog.at_level(logging.DEBUG):
            p._preprocess_string_cols()  # should return early, no error

    def test_missing_columns_config_returns_early(self, caplog):
        """Return early when string column config is missing."""
        df = pd.DataFrame({"text": ["hello"]})
        cfg = {"string_cols": {"patterns": {"p": "x"}}}  # no 'columns'
        p = _make(df, cfg)
        with caplog.at_level(logging.DEBUG):
            p._preprocess_string_cols()  # should return early, no error

    def test_empty_pattern_string_skipped(self):
        """Patterns with empty string values are compiled but result is falsy-ish check."""
        df = pd.DataFrame({"text": ["hello"]})
        cfg = {
            "string_cols": {
                "patterns": {"empty": ""},  # empty string – compiled but falsy regex
                "columns": {"text": ["empty"]},
            }
        }
        p = _make(df, cfg)
        # Empty pattern compiles but re.compile("") matches everything between chars
        # The key point: no error is raised
        p._preprocess_string_cols()


# ---------------------------------------------------------------------------
# _convert_datetime_columns
# ---------------------------------------------------------------------------


class TestConvertDatetimeCols:
    """Tests for datetime conversion logic."""

    @pytest.mark.parametrize(
        "values",
        [
            [1_000_000],
            [1_000_000_000_001],
            ["2024-01-01", "2024-06-15"],
        ],
    )
    def test_converts_supported_datetime_inputs(self, values):
        """Convert numeric and string datetime inputs to datetime dtype."""
        df = pd.DataFrame({"ts": values})
        p = _make(df, {"datetime_cols": ["ts"]})
        p._convert_datetime_columns()
        assert pd.api.types.is_datetime64_any_dtype(p.df["ts"])

    def test_missing_col_warns(self, caplog):
        """Log a warning when datetime_cols references a missing column."""
        df = pd.DataFrame({"a": [1]})
        p = _make(df, {"datetime_cols": ["missing"]})
        with caplog.at_level(logging.WARNING):
            p._convert_datetime_columns()
        assert "datetime_cols" in caplog.text


# ---------------------------------------------------------------------------
# _sort_by_time
# ---------------------------------------------------------------------------


class TestSortByTime:
    """Tests for optional sort behavior."""

    def test_sorts_ascending(self):
        """Sort rows ascending on the configured column."""
        df = pd.DataFrame({"ts": [3, 1, 2]})
        p = _make(df, {"sort_col": "ts"})
        p._sort_by_time()
        assert list(p.df["ts"]) == [1, 2, 3]

    def test_missing_col_warns(self, caplog):
        """Log a warning when sort_col is not present."""
        df = pd.DataFrame({"a": [1]})
        p = _make(df, {"sort_col": "ts"})
        with caplog.at_level(logging.WARNING):
            p._sort_by_time()
        assert "sort_col" in caplog.text

    def test_none_sort_col_returns_early(self):
        """Leave data unchanged when sort_col is empty or None."""
        df = pd.DataFrame({"a": [3, 1, 2]})
        p = _make(df, {"sort_col": None})
        p._sort_by_time()
        assert list(p.df["a"]) == [3, 1, 2]  # unchanged


# ---------------------------------------------------------------------------
# _drop_duplicates
# ---------------------------------------------------------------------------


class TestDropDuplicates:
    """Tests for configurable duplicate row removal."""

    @pytest.mark.parametrize(
        "drop_duplicates, expected_len",
        [(True, 2), (False, 3)],
    )
    def test_drop_duplicates_flag(self, drop_duplicates, expected_len):
        """Apply duplicate removal only when enabled."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        p = _make(df, {"drop_duplicates": drop_duplicates})
        p._drop_duplicates()
        assert len(p.df) == expected_len


# ---------------------------------------------------------------------------
# _col_mismatched_warning
# ---------------------------------------------------------------------------


class TestColMismatchedWarning:
    """Tests for warning formatting helper."""

    @pytest.mark.parametrize(
        "col_input, expected_text",
        [(["col1", "col2"], "col1, col2"), ("col1", "col1")],
    )
    def test_warning_message_formats_columns(self, caplog, col_input, expected_text):
        """Format and log missing column warnings for str and list inputs."""
        p = _make(_simple_df(), {})
        with caplog.at_level(logging.WARNING):
            p._col_mismatched_warning("test_func", col_input)
        assert expected_text in caplog.text


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


class TestGetData:
    """Tests for access to processed DataFrame data."""

    def test_returns_dataframe(self):
        """Return the underlying DataFrame from get_data."""
        df = _simple_df()
        p = _make(df, {})
        assert p.get_data() is df


# ---------------------------------------------------------------------------
# _custom_preprocessing (default pass)
# ---------------------------------------------------------------------------


class TestCustomPreprocessing:
    """Tests for default custom preprocessing hook behavior."""

    def test_default_is_noop(self):
        """Do nothing by default in _custom_preprocessing."""
        df = _simple_df()
        p = _make(df, {})
        p._custom_preprocessing()  # should not raise


# ---------------------------------------------------------------------------
# Full preprocess pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Integration-style test for end-to-end preprocessing."""

    def test_full_pipeline(self):
        """Run full configured pipeline and validate key outcomes."""
        df = pd.DataFrame(
            {
                "ID": [1, 2, 2],
                "DATE": ["2024-01-01", "2024-06-01", "2024-06-01"],
                "TEXT": ["Hello <b>World</b>", None, None],
            }
        )
        config = {
            "use_cols": ["ID", "DATE", "TEXT"],
            "fillna": {"TEXT": ""},
            "string_cols": {
                "patterns": {"strip_html": "<[^>]+>"},
                "columns": {"TEXT": ["strip_html"]},
            },
            "datetime_cols": ["DATE"],
            "sort_col": "DATE",
            "drop_duplicates": True,
        }
        p = BasePreProcessor(df, config)
        result = p.preprocess()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["text"].iloc[0] == "Hello World"
