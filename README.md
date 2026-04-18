# data-pipeline

Small, configurable utilities for loading tabular datasets and running reusable preprocessing steps.

## What this project provides

The package includes three core parts:

- `DataLoader` (`data_pipeline/dataloader.py`)
	- Loads one file or a directory of files.
	- Supports `.csv`, `.json` (json lines), `.parquet`, `.xlsx`, `.txt` (tab-delimited).
	- Can return a dictionary of DataFrames or merge files into one DataFrame.

- `BasePreProcessor` (`data_pipeline/base_preprocessor.py`)
	- Applies configurable preprocessing steps for tabular data.
	- Steps include:
		- lower-casing column names
		- selecting `use_cols`
		- `fillna`
		- regex-based string cleaning (`string_cols`)
		- datetime conversion (`datetime_cols`)
		- optional sorting (`sort_col`)
		- optional deduplication (`drop_duplicates`)

- `Config` (`data_pipeline/config.py`)
	- Loads JSON config files from `configs/`.
	- Normalizes all config keys to lowercase.
	- Validates config structure with a schema.

## Requirements

- Python `>=3.11`
- Poetry (recommended package/environment manager)

Project dependencies are defined in `pyproject.toml`.

Important format notes:

- Reading `.parquet` files requires `pyarrow`.
- Reading `.xlsx` files requires `openpyxl`.

Those are included in the current dev dependency group.

## Installation

From the project root:

```bash
poetry install
```

## Quick How To Run

Minimal end-to-end example (load + preprocess):

```python
from data_pipeline.config import Config
from data_pipeline.dataloader import DataLoader
from data_pipeline.base_preprocessor import BasePreProcessor

# Load config files from ./configs
loader_cfg = Config("dataloader.json").data
processor_cfg = Config("dataprocessor.json").data

# Load data
loader = DataLoader(loader_cfg).load()
data = loader.get_data()

# If merge_files=false in loader config, data is a dict[str, DataFrame]
# This example preprocesses either the single DataFrame or each DataFrame in a dict.
if isinstance(data, dict):
		processed = {k: BasePreProcessor(df, processor_cfg).preprocess() for k, df in data.items()}
else:
		processed = BasePreProcessor(data, processor_cfg).preprocess()

print(type(processed))
```

Run it with Poetry:

```bash
poetry run python your_script.py
```

## Config Files

### `configs/dataloader.json`

Used by `DataLoader`.

Current keys:

- `data_path` (required): file or directory path
- `merge_files` (optional): `true|false`
- `extensions` (optional): list of extensions, for example `[".csv"]`
- `read_kwargs` (optional): forwarded to pandas reader functions

Example:

```json
{
	"data_path": "...\\Projects\\data\\",
	"merge_files": true,
	"extensions": [".csv"]
}
```

### `configs/dataprocessor.json`

Used by `BasePreProcessor`.

Current keys:

- `use_cols`: list of columns to keep
- `datetime_cols`: list of columns to parse as datetime
- `sort_col`: column to sort by
- `string_cols.patterns`: regex patterns
- `string_cols.columns`: mapping of column to list of pattern names
- `fillna`: mapping of column to replacement value
- `drop_duplicates`: `true|false`

Example:

```json
{
	"use_cols": [],
	"datetime_cols": [],
	"sort_col": "",
	"string_cols": {
		"patterns": {
			"remove_characters": "",
			"remove_tags": "",
			"remove_extra_whitespace": ""
		},
		"columns": {}
	},
	"fillna": {
		"Summary": ""
	},
	"drop_duplicates": true
}
```

## Running Tests

Run all tests:

```bash
poetry run python -m pytest tests -q
```

Coverage is enforced from `pyproject.toml`:

- target package: `data_pipeline`
- fail-under: `100`

So the command above also checks coverage automatically.

## Project Structure

```text
data-pipeline/
	configs/
		dataloader.json
		dataprocessor.json
	data_pipeline/
		__init__.py
		base_preprocessor.py
		config.py
		dataloader.py
	tests/
		test_base_preprocessor.py
		test_config.py
		test_dataloader.py
	pyproject.toml
```
