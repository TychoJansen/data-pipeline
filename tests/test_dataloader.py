"""Tests for data_pipeline.dataloader.DataLoader."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from data_pipeline.dataloader import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csv(path: Path, name: str = "a.csv", content: str = "col1,col2\n1,2\n3,4\n") -> Path:
    f = path / name
    f.write_text(content, encoding="utf-8")
    return f


def _json_lines(path: Path, name: str = "a.json") -> Path:
    f = path / name
    f.write_text('{"col1": 1, "col2": 2}\n{"col1": 3, "col2": 4}\n', encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# __init__ / _validate_config
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for DataLoader initialization and config validation."""

    def test_valid_file_config(self, tmp_path):
        """Create a loader from a valid file path."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        assert loader.path == f

    def test_missing_required_key_raises(self):
        """Raise when required data_path is missing."""
        with pytest.raises(KeyError, match="data_path"):
            DataLoader({})

    def test_path_not_exists_raises(self):
        """Raise when configured path does not exist."""
        with pytest.raises(FileNotFoundError):
            DataLoader({"data_path": "/nonexistent/path/file.csv"})

    def test_extensions_normalized(self, tmp_path):
        """Normalize configured extensions to lowercase dotted form."""
        _csv(tmp_path)
        loader = DataLoader({"data_path": str(tmp_path), "extensions": ["CSV", ".txt"]})
        assert ".csv" in loader.extensions
        assert ".txt" in loader.extensions

    def test_no_extensions(self, tmp_path):
        """Keep extensions unset when not provided."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        assert loader.extensions is None

    def test_read_kwargs_stored(self, tmp_path):
        """Store read_kwargs for downstream pandas readers."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f), "read_kwargs": {"nrows": 1}})
        assert loader.read_kwargs == {"nrows": 1}


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """Tests for DataLoader string representation."""

    def test_repr(self, tmp_path):
        """Include key fields in __repr__."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        r = repr(loader)
        assert "DataLoader" in r
        assert "extensions" in r


# ---------------------------------------------------------------------------
# load – single file
# ---------------------------------------------------------------------------


class TestLoadSingleFile:
    """Tests for loading individual supported file types."""

    def test_load_csv(self, tmp_path):
        """Load CSV files into a DataFrame."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert isinstance(loader.data, pd.DataFrame)
        assert list(loader.data.columns) == ["col1", "col2"]

    def test_load_json_lines(self, tmp_path):
        """Load JSON lines files into a DataFrame."""
        f = _json_lines(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert isinstance(loader.data, pd.DataFrame)

    def test_load_parquet(self, tmp_path):
        """Load Parquet files into a DataFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        f = tmp_path / "data.parquet"
        df.to_parquet(f, index=False)
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert loader.data.shape == (2, 1)

    def test_load_xlsx(self, tmp_path):
        """Load XLSX files into a DataFrame."""
        df = pd.DataFrame({"a": [1]})
        f = tmp_path / "data.xlsx"
        df.to_excel(f, index=False)
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert loader.data.shape == (1, 1)

    def test_load_txt(self, tmp_path):
        """Load tab-delimited TXT files into a DataFrame."""
        f = tmp_path / "data.txt"
        f.write_text("col1\tcol2\n1\t2\n", encoding="utf-8")
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert "col1" in loader.data.columns

    def test_unsupported_extension_raises(self, tmp_path):
        """Raise for unsupported file extensions."""
        f = tmp_path / "data.xyz"
        f.write_text("data")
        loader = DataLoader({"data_path": str(f)})
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load()

    def test_load_returns_self(self, tmp_path):
        """Return self to allow call chaining."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        result = loader.load()
        assert result is loader


# ---------------------------------------------------------------------------
# load – invalid path (neither file nor dir)
# ---------------------------------------------------------------------------


class TestLoadInvalidPath:
    """Tests for invalid load path type handling."""

    def test_neither_file_nor_dir_raises(self, tmp_path):
        """Raise when path behaves as neither file nor directory."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        # Make path point to something that exists as a Path object but mock is_file/is_dir
        with patch("pathlib.Path.is_file", return_value=False), patch("pathlib.Path.is_dir", return_value=False):
            with pytest.raises(ValueError, match="neither file nor directory"):
                loader.load()


# ---------------------------------------------------------------------------
# load – directory
# ---------------------------------------------------------------------------


class TestLoadDirectory:
    """Tests for loading and merging directory contents."""

    def test_load_dir_returns_dict(self, tmp_path):
        """Return a dict of DataFrames when merge_files is false."""
        _csv(tmp_path, "a.csv")
        _csv(tmp_path, "b.csv")
        loader = DataLoader({"data_path": str(tmp_path)})
        loader.load()
        assert isinstance(loader.data, dict)
        assert len(loader.data) == 2

    def test_load_dir_merge(self, tmp_path):
        """Merge files into one DataFrame when merge_files is true."""
        _csv(tmp_path, "a.csv")
        _csv(tmp_path, "b.csv")
        loader = DataLoader({"data_path": str(tmp_path), "merge_files": True})
        loader.load()
        assert isinstance(loader.data, pd.DataFrame)
        assert len(loader.data) == 4

    def test_load_dir_extension_filter(self, tmp_path):
        """Apply extension filtering while loading a directory."""
        _csv(tmp_path, "a.csv")
        (tmp_path / "b.txt").write_text("x\ty\n1\t2\n", encoding="utf-8")
        loader = DataLoader({"data_path": str(tmp_path), "extensions": ["csv"]})
        loader.load()
        assert len(loader.data) == 1

    @pytest.mark.parametrize(
        "config",
        [
            {"extensions": ["csv"]},
            {},
        ],
    )
    def test_load_dir_no_files_raises(self, tmp_path, config):
        """Raise when no matching files are found in the directory."""
        loader = DataLoader({"data_path": str(tmp_path), **config})
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_merge_schema_mismatch_raises(self, tmp_path):
        """Raise when merged files have mismatched schemas."""
        _csv(tmp_path, "a.csv", "x,y\n1,2\n")
        _csv(tmp_path, "b.csv", "x,z\n3,4\n")
        loader = DataLoader({"data_path": str(tmp_path), "merge_files": True})
        with pytest.raises(ValueError, match="Schema mismatch"):
            loader.load()

    def test_duplicate_key_raises(self, tmp_path):
        """Force duplicate keys by mapping all relative paths to the same value."""
        _csv(tmp_path, "a.csv")
        _csv(tmp_path, "b.csv")
        loader = DataLoader({"data_path": str(tmp_path)})
        with patch("pathlib.Path.relative_to", return_value=Path("duplicate.csv")):
            with pytest.raises(RuntimeError, match="Unexpected duplicate key"):
                loader._load_directory()


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


class TestGetData:
    """Tests for retrieving loaded data from the loader."""

    def test_get_data_before_load_raises(self, tmp_path):
        """Raise when get_data is called before load."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        with pytest.raises(ValueError, match="load\\(\\)"):
            loader.get_data()

    def test_get_data_after_load(self, tmp_path):
        """Return loaded data after successful load call."""
        f = _csv(tmp_path)
        loader = DataLoader({"data_path": str(f)})
        loader.load()
        assert isinstance(loader.get_data(), pd.DataFrame)
