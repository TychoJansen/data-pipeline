r"""Base data processor for tabular data cleaning and preprocessing.

Provides a configurable preprocessing pipeline for data cleaning, missing value
handling, column selection, string/datetime processing, and deduplication.

Config example:
{
    "use_cols": ["Id", "ProductId"],
    "string_cols": {
        "patterns": {"remove_tags": "<[^>]+>"},
        "columns": {"summary": ["remove_tags"]}
    },
    "fillna": {"Summary": ""},
    "drop_duplicates": true
}

Usage:
    from data_pipeline.base_preprocessor import BasePreProcessor

    processor = BasePreProcessor(df, config)
    clean_df = processor.preprocess()
"""

import functools
import logging
import re
from typing import Any, Dict, List, Union

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BasePreProcessor:
    """Process tabular data using a configuration dictionary."""

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """Initialize the data processor.

        Args:
            df: The pandas DataFrame to preprocess.
            config: Configuration dictionary containing all required keys.
        """
        self.config: Dict[str, Any] = config
        self.df: pd.DataFrame = df

    @staticmethod
    def requires_config(key: str):
        """Skip the decorated method when ``key`` is absent from config.

        Args:
            key: The config key that must be present for the method to execute.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if key not in self.config:
                    return
                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def preprocess(self) -> pd.DataFrame:
        """Run the full preprocessing pipeline and return the cleaned DataFrame.

        Returns:
            The preprocessed DataFrame.
        """
        self._basic_preprocessing()
        self._custom_preprocessing()
        return self.df

    def _basic_preprocessing(self) -> None:
        """Apply the standard preprocessing steps in a fixed order."""
        self._lowercase_col_names()
        self._drop_unused_columns()
        self._fill_missing_values()
        self._preprocess_string_cols()
        self._convert_datetime_columns()
        self._sort_by_time()
        self._drop_duplicates()

    def get_data(self) -> pd.DataFrame:
        """Return the processed pandas DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        return self.df

    def _lowercase_col_names(self) -> None:
        """Convert all DataFrame column names to lowercase."""
        self.df.columns = [col.lower() for col in self.df.columns]

    @requires_config("use_cols")
    def _drop_unused_columns(self) -> None:
        """Drop columns not specified in the configuration's use_cols."""
        use_cols = [c.lower() for c in self.config.get("use_cols")] or []

        missing = [col for col in use_cols if col not in self.df.columns]
        if missing:
            self._col_mismatched_warning("use_cols", missing)

        self.df = self.df[self.df.columns.intersection(use_cols)]

    @requires_config("fillna")
    def _fill_missing_values(self) -> None:
        """Fill missing values based on the configuration's fillna mapping."""
        col_dict = self.config.get("fillna") or {}
        for col, value in col_dict.items():
            col = col.lower()
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(value)
            else:
                self._col_mismatched_warning("fillna", col)

    @requires_config("string_cols")
    def _preprocess_string_cols(self) -> None:
        """Clean string columns using compiled regex patterns defined in config.

        Reads ``string_cols.patterns`` and ``string_cols.columns`` from the
        config to determine which patterns to apply to which columns.
        """
        settings = self.config.get("string_cols") or {}
        pattern_defs = settings.get("patterns", {})
        col_settings = settings.get("columns", {})

        if not pattern_defs or not col_settings:
            logger.debug("patterns config or columns config is missing under string_cols")
            return

        compiled_patterns = {name: re.compile(p) for name, p in pattern_defs.items() if p}

        def clean_text(x: Any, pattern_names: List[str]):
            if pd.isna(x):
                return x

            value = str(x).strip()

            for name in pattern_names:
                regex = compiled_patterns.get(name)
                if not regex:
                    logger.warning(f"Pattern '{name}' not found in config")
                    continue
                value = regex.sub("", value)

            return value

        for col, pattern_names in col_settings.items():
            col = col.lower()

            if col not in self.df.columns:
                self._col_mismatched_warning("string_cols", col)
                continue

            self.df[col] = self.df[col].apply(lambda x: clean_text(x, pattern_names))

    @requires_config("datetime_cols")
    def _convert_datetime_columns(self) -> None:
        """Convert configured columns to pandas datetime format.

        Supports:
        - Unix timestamps (seconds or milliseconds)
        - ISO date strings
        - Mixed safe conversion

        Config:
            datetime_cols: list of column names to convert
        """
        cols = self.config.get("datetime_cols") or []
        cols = [col.lower() for col in cols]

        for col in cols:
            if col not in self.df.columns:
                self._col_mismatched_warning("datetime_cols", col)
                continue

            series = self.df[col]

            # Numeric (Unix timestamp)
            if pd.api.types.is_numeric_dtype(series):
                # Auto-detect seconds vs milliseconds
                if series.max() > 1e12:
                    self.df[col] = pd.to_datetime(series, unit="ms", errors="coerce")
                else:
                    self.df[col] = pd.to_datetime(series, unit="s", errors="coerce")
            # String / object
            else:
                self.df[col] = pd.to_datetime(series, errors="coerce")

    @requires_config("sort_col")
    def _sort_by_time(self) -> None:
        """Sort the DataFrame by the column specified in ``sort_col`` config key."""
        sort_col = self.config.get("sort_col")

        if not sort_col:
            return
        sort_col = sort_col.lower()

        if sort_col in self.df.columns:
            self.df = self.df.sort_values(sort_col)
        else:
            self._col_mismatched_warning("sort_col", sort_col)

    @requires_config("drop_duplicates")
    def _drop_duplicates(self) -> None:
        """Drop duplicate rows when ``drop_duplicates`` is True in config."""
        if self.config["drop_duplicates"]:
            self.df = self.df.drop_duplicates()

    def _col_mismatched_warning(self, func: str, col: Union[str, List[str]]) -> None:
        """Log a warning for columns not found in the DataFrame.

        Args:
            func: Name of the calling method (used in the log message).
            col: Column name or list of column names that were not found.
        """
        if isinstance(col, list):
            col = ", ".join(col)
        logger.warning(f"{func}: columns not found -> {col}")

    def _custom_preprocessing(self) -> None:
        """Override this in subclasses to add custom preprocessing steps."""
        pass
