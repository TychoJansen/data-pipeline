"""DataLoader: A configurable class for loading and preprocessing tabular data.

This class loads tabular data (CSV, JSON, Parquet, XLSX, or tab-delimited TXT)
based on a config dictionary.
Designed for flexible ETL workflows, with all options controlled via the config
dictionary or JSON file.

Config example:
{
    "data_path": "your_file.csv",
    "extensions": ["csv"],
    "merge_files": true
}

Usage:
    from data_pipeline.dataloader import DataLoader

    loader = DataLoader({"data_path": "data/myfile.csv"})
    loader.load()
    df = loader.get_data()
"""
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataLoader:
    """Load and preprocess tabular data using a configuration dictionary."""

    REQUIRED_KEYS = ["data_path"]
    READERS = {
        ".csv": lambda p, **kwargs: pd.read_csv(p, **kwargs),
        ".json": lambda p, **kwargs: pd.read_json(p, lines=True, **kwargs),
        ".parquet": lambda p, **kwargs: pd.read_parquet(p, **kwargs),
        ".xlsx": lambda p, **kwargs: pd.read_excel(p, **kwargs),
        ".txt": lambda p, **kwargs: pd.read_csv(p, **{"sep": "\t", **kwargs}),
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataLoader.

        Args:
            config: Configuration dictionary containing all required keys.
        """
        self.config: Dict[str, Any] = config
        self._validate_config()

        self.path: Path = Path(config["data_path"])
        self.read_kwargs = self.config.get("read_kwargs", {})

        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        self.extensions = self.config.get("extensions")
        if self.extensions:
            self.extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in self.extensions]

    def _validate_config(self) -> None:
        """Validate that the config contains all required keys."""
        for key in self.REQUIRED_KEYS:
            if key not in self.config:
                raise KeyError(f"Missing key in config: {key}")

    def __repr__(self) -> str:
        """Return a string representation of the DataLoader instance."""
        return f"DataLoader(path={self.path}, extensions={self.extensions})"

    def load(self) -> "DataLoader":
        """Load data from the configured path (file or directory).

        Returns:
            self: The DataLoader instance with data loaded into ``self.data``.

        Raises:
            ValueError: If the path is neither a file nor a directory.
        """
        if self.path.is_file():
            logger.info(f"Loading single file: {self.path}")
            self.data = self._load_single_file(self.path)
        elif self.path.is_dir():
            logger.info(f"Loading files from directory: {self.path}")
            self.data = self._load_directory()
        else:
            raise ValueError(f"Path is neither file nor directory: {self.path}")
        return self

    def _load_directory(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load all matching files from the configured directory.

        Returns:
            A single merged DataFrame when ``merge_files=True``, or a dict
            mapping relative file paths to DataFrames when ``merge_files=False``.

        Raises:
            FileNotFoundError: If no matching files are found in the directory.
            ValueError: If a schema mismatch occurs during merge.
            RuntimeError: If duplicate file keys are encountered.
        """
        files = [f for f in sorted(self.path.glob("*")) if f.is_file()]

        if self.extensions:
            files = [f for f in files if f.suffix.lower() in self.extensions]

        if not files:
            raise FileNotFoundError(f"No files found in {self.path} with extensions: {self.extensions}")

        # Case 1: merge all files into a single DataFrame
        if self.config.get("merge_files", False):
            logger.info(f"Merging {len(files)} files into a single DataFrame.")
            dfs = [self._load_single_file(f) for f in files]

            # enforce schema and raise error if column mismatch
            cols = dfs[0].columns
            for i, df in enumerate(dfs[1:], start=1):
                if not df.columns.equals(cols):
                    raise ValueError(f"Schema mismatch in file {files[i]} compared to first file")
            return pd.concat(dfs, ignore_index=True)

        # Case 2: return dict with relative paths
        else:
            logger.info(f"Returning {len(files)} files as a dictionary.")
            data: Dict[str, pd.DataFrame] = {}

            for f in files:
                key = f.relative_to(self.path).as_posix()

                # Check for duplicate keys
                if key in data:
                    raise RuntimeError(f"Unexpected duplicate key: {key}")

                data[key] = self._load_single_file(f)
            return data

    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single file into a DataFrame based on its extension.

        Args:
            file_path: Path to the file to load.

        Returns:
            DataFrame loaded from the file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        file_type = file_path.suffix.lower()

        reader = self.READERS.get(file_type)
        if not reader:
            raise ValueError(f"Unsupported file type: {file_type}")

        return reader(file_path, **self.read_kwargs)

    def get_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return the data loaded by :meth:`load`.

        Returns:
            DataFrame or dict of DataFrames previously loaded.

        Raises:
            ValueError: If :meth:`load` has not been called yet.
        """
        if not hasattr(self, "data"):
            raise ValueError("Data not loaded. Call load() first.")
        return self.data
