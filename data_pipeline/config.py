"""Configuration helpers for loading and validating JSON config files.

Provides the Config class for loading JSON configuration files, searching
upward through parent directories when needed, and exposing config values
with dict-like access and dot-notation support.

Usage:
    from data_pipeline.config import Config

    cfg = Config("dataloader.json")
    data_path = cfg["data_path"]
    use_cols = cfg.get("use_cols", [])
"""

import json
import os
from typing import Any, Dict


class Config:
    """Load and validate application configuration from JSON files."""

    SCHEMA = {
        "data_path": {"type": str, "required": True},
        "use_cols": {"type": list, "items": str, "required": False},
        "fillna": {"type": dict, "required": False},
        "datetime_cols": {"type": list, "items": str, "required": False},
        "sort_col": {"type": str, "required": False},
        "drop_duplicates": {"type": bool, "required": False},
        "string_cols": {
            "type": dict,
            "required": False,
            "schema": {
                "patterns": {"type": dict, "required": True},
                "columns": {"type": dict, "required": True},
            },
        },
    }

    def __init__(
        self,
        filename: str,
        config_dir: str = "configs",
        search_levels: int = 5,
        schema: Dict[str, Any] = None,
        validate_on_load: bool = True,
        strict: bool = True,
    ) -> None:
        """Initialize Config by locating, loading, and validating the JSON file.

        Args:
            filename: JSON config file name (e.g. ``"dataloader.json"``).
            config_dir: Directory name to search for the config file.
            search_levels: Number of parent directories to traverse while searching.
            schema: Custom validation schema; defaults to :attr:`SCHEMA`.
            validate_on_load: Whether to validate the config immediately after load.
            strict: When ``True``, unknown keys raise :class:`KeyError`.

        Raises:
            FileNotFoundError: If the config file is not found within ``search_levels``.
            KeyError: If required keys are missing or unknown keys exist in strict mode.
            TypeError: If a config value has the wrong type.
        """
        self.path: str = self._find_config_path(filename, config_dir, search_levels)
        self._schema = schema or self.SCHEMA

        raw = self._load()
        self.config: Dict[str, Any] = self._normalize_config(raw)

        if validate_on_load:
            self.validate(strict=strict)

    @staticmethod
    def _normalize_config(obj):
        """Recursively lowercase all dict keys in a config object.

        Args:
            obj: The config value to normalize (dict, list, or scalar).

        Returns:
            The normalized config with all dict keys lowercased.
        """
        if isinstance(obj, dict):
            return {str(k).lower(): Config._normalize_config(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._normalize_config(item) for item in obj]
        else:
            return obj

    def _find_config_path(self, filename: str, config_dir: str, search_levels: int) -> str:
        """Search upward through parent directories for the config file.

        Args:
            filename: Config file name to locate.
            config_dir: Sub-directory name where config files are expected.
            search_levels: Maximum number of directory levels to traverse upward.

        Returns:
            Absolute path to the found config file.

        Raises:
            FileNotFoundError: If the file is not found within the given levels.
        """
        start_dir = os.getcwd()

        dir_to_search = start_dir
        for _ in range(search_levels):
            potential_path = os.path.join(dir_to_search, config_dir, filename)
            if os.path.exists(potential_path):
                return potential_path
            dir_to_search = os.path.dirname(dir_to_search)

        raise FileNotFoundError(
            f"Config file '{filename}' not found after searching {search_levels} levels up from {start_dir}"
        )

    def _load(self) -> Dict[str, Any]:
        """Read and parse the JSON config file from :attr:`path`.

        Returns:
            Parsed JSON content as a dictionary.
        """
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate(self, schema: Dict[str, Any] = None, strict: bool = False) -> None:
        """Validate the loaded config against a schema.

        Args:
            schema: Schema dict to validate against; defaults to :attr:`_schema`.
            strict: When ``True``, unknown keys raise :class:`KeyError`.

        Raises:
            KeyError: If required keys are missing or unknown keys exist (strict).
            TypeError: If a value does not match its expected type.
        """
        schema = schema or self._schema
        self._validate_dict(self.config, schema, path="config", strict=strict)

    def _validate_dict(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        strict: bool,
    ) -> None:
        """Recursively validate a dict against a schema definition.

        Args:
            data: The config dict to validate.
            schema: Schema defining expected types and requirements.
            path: Dot-separated path string used in error messages.
            strict: When ``True``, keys not present in the schema raise :class:`KeyError`.

        Raises:
            KeyError: For unknown keys (strict mode) or missing required keys.
            TypeError: If a value does not match its expected type.
        """
        # Check for unknown keys in strict mode
        if strict:
            unknown_keys = set(data) - set(schema)
            if unknown_keys:
                raise KeyError(f"{path}: unknown keys -> {', '.join(unknown_keys)}")

        for key, rule in schema.items():
            current_path = f"{path}.{key}"

            # Normalize rule
            if isinstance(rule, dict) and "type" in rule:
                expected_type = rule["type"]
                is_required = rule.get("required", False)
                nested_schema = rule.get("schema")
                item_type = rule.get("items")
            else:
                expected_type = rule
                is_required = False
                nested_schema = None
                item_type = None

            # Check missing keys
            if key not in data:
                if is_required:
                    raise KeyError(f"Missing required key: {current_path}")
                continue

            value = data[key]

            # Check type
            if not isinstance(value, expected_type):
                type_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"{current_path} must be {type_name}, got {type(value).__name__}")

            # Nested dict validation
            if expected_type is dict and nested_schema:
                self._validate_dict(value, nested_schema, current_path, strict)

            # List item type validation
            if expected_type is list and item_type:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        type_name = getattr(item_type, "__name__", str(item_type))
                        raise TypeError(f"{current_path}[{i}] must be {type_name}, got {type(item).__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        """Supports dot notation: 'string_cols.patterns'."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k)
            if value is None:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Return the config value for ``key``.

        Raises:
            KeyError: If ``key`` is not present in the config.
        """
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if ``key`` is present in the top-level config dict."""
        return key in self.config

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw config dictionary."""
        return self.config
