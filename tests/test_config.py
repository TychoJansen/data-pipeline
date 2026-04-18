"""Tests for data_pipeline.config.Config."""
import json

import pytest

from data_pipeline.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path, data):
    """Write a JSON config file inside a configs/ sub-directory."""
    cfg_dir = path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / "test.json"
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    return cfg_file


# ---------------------------------------------------------------------------
# _normalize_config
# ---------------------------------------------------------------------------


class TestNormalizeConfig:
    """Tests for recursive config normalization behavior."""

    def test_dict_keys_lowercased(self):
        """Lowercase top-level dictionary keys."""
        result = Config._normalize_config({"FOO": 1, "Bar": "baz"})
        assert result == {"foo": 1, "bar": "baz"}

    def test_nested_dict(self):
        """Lowercase keys in nested dictionaries."""
        result = Config._normalize_config({"OUTER": {"INNER": 42}})
        assert result == {"outer": {"inner": 42}}

    def test_list_items_normalized(self):
        """Normalize dictionary keys for items inside lists."""
        result = Config._normalize_config([{"KEY": "val"}])
        assert result == [{"key": "val"}]

    def test_scalar_passthrough(self):
        """Return scalar values unchanged."""
        assert Config._normalize_config("hello") == "hello"
        assert Config._normalize_config(123) == 123
        assert Config._normalize_config(None) is None


# ---------------------------------------------------------------------------
# _find_config_path
# ---------------------------------------------------------------------------


class TestFindConfigPath:
    """Tests for config path discovery behavior."""

    def test_finds_config_in_configs_dir(self, tmp_path, monkeypatch):
        """Find a config file in a provided configs directory."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"data_path": "x.csv"})
        cfg = Config.__new__(Config)
        found = cfg._find_config_path("test.json", str(tmp_path / "configs"), 5)
        assert found.endswith("test.json")

    def test_raises_when_not_found(self, tmp_path, monkeypatch):
        """Raise when config file cannot be found."""
        monkeypatch.chdir(tmp_path)
        cfg = Config.__new__(Config)
        with pytest.raises(FileNotFoundError, match="Config file"):
            cfg._find_config_path("missing.json", "configs", 1)

    def test_falls_back_to_cwd_when_abspath_raises_nameerror(self, tmp_path, monkeypatch):
        """Fall back to cwd when __file__ resolution fails."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"data_path": "x.csv"})
        cfg = Config.__new__(Config)

        with monkeypatch.context() as m:
            m.setattr("data_pipeline.config.os.path.abspath", lambda _: (_ for _ in ()).throw(NameError("no __file__")))
            found = cfg._find_config_path("test.json", str(tmp_path / "configs"), 1)

        assert found.endswith("test.json")


# ---------------------------------------------------------------------------
# __init__ / _load
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for Config initialization and load behavior."""

    def test_loads_valid_config(self, tmp_path, monkeypatch):
        """Load valid JSON config content."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"data_path": "x.csv"})
        cfg = Config("test.json", config_dir=str(tmp_path / "configs"), validate_on_load=False)
        assert cfg["data_path"] == "x.csv"

    def test_validate_on_load_runs(self, tmp_path, monkeypatch):
        """Validate on load when enabled."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"data_path": "x.csv"})
        # Should not raise with valid required key
        Config("test.json", config_dir=str(tmp_path / "configs"), validate_on_load=True, strict=False)

    def test_validate_on_load_raises_on_missing_required(self, tmp_path, monkeypatch):
        """Raise on load when required keys are missing."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"use_cols": ["a"]})
        with pytest.raises(KeyError, match="data_path"):
            Config("test.json", config_dir=str(tmp_path / "configs"), validate_on_load=True, strict=False)


# ---------------------------------------------------------------------------
# validate / _validate_dict
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for schema validation logic and error paths."""

    def _make_cfg(self, tmp_path, monkeypatch, data, validate_on_load=False):
        """Create a Config instance from temporary test data."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, data)
        return Config(
            "test.json",
            config_dir=str(tmp_path / "configs"),
            validate_on_load=validate_on_load,
            strict=False,
        )

    def test_strict_unknown_keys_raises(self, tmp_path, monkeypatch):
        """Raise on unknown keys when strict mode is enabled."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv", "unknown": 1})
        with pytest.raises(KeyError, match="unknown"):
            cfg.validate(strict=True)

    def test_missing_required_raises(self, tmp_path, monkeypatch):
        """Raise when required schema keys are missing."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"use_cols": ["a"]})
        with pytest.raises(KeyError, match="data_path"):
            cfg.validate(strict=False)

    def test_wrong_type_raises(self, tmp_path, monkeypatch):
        """Raise when a config value has the wrong type."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": 123})
        with pytest.raises(TypeError, match="data_path"):
            cfg.validate(strict=False)

    def test_valid_config_passes(self, tmp_path, monkeypatch):
        """Pass validation for a minimal valid config."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv"})
        cfg.validate(strict=False)  # No raise

    def test_list_item_type_wrong_raises(self, tmp_path, monkeypatch):
        """Raise when list items do not match expected item type."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv", "use_cols": [1, 2]})
        with pytest.raises(TypeError, match="use_cols"):
            cfg.validate(strict=False)

    def test_list_item_type_valid(self, tmp_path, monkeypatch):
        """Pass when list item types match schema requirements."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv", "use_cols": ["a"]})
        cfg.validate(strict=False)  # No raise

    def test_nested_schema_validated(self, tmp_path, monkeypatch):
        """string_cols has a nested schema requiring 'patterns' and 'columns'."""
        data = {
            "data_path": "x.csv",
            "string_cols": {
                "patterns": {"strip_html": "<[^>]+>"},
                "columns": {"text": ["strip_html"]},
            },
        }
        cfg = self._make_cfg(tmp_path, monkeypatch, data)
        cfg.validate(strict=False)

    def test_nested_schema_missing_key_raises(self, tmp_path, monkeypatch):
        """string_cols.patterns is required - missing it should raise."""
        data = {
            "data_path": "x.csv",
            "string_cols": {"columns": {"text": []}},
        }
        cfg = self._make_cfg(tmp_path, monkeypatch, data)
        with pytest.raises(KeyError):
            cfg.validate(strict=False)

    def test_rule_without_type_key(self, tmp_path, monkeypatch):
        """A rule that is just a bare type (not a dict with 'type') uses the else branch."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv"})
        # Use a custom schema where a rule is a bare type, not a dict
        custom_schema = {"data_path": str}
        cfg.validate(schema=custom_schema, strict=False)

    def test_custom_schema_passed_to_validate(self, tmp_path, monkeypatch):
        """Validate against a caller-provided custom schema."""
        cfg = self._make_cfg(tmp_path, monkeypatch, {"data_path": "x.csv"})
        cfg.validate(schema={"data_path": {"type": str, "required": True}}, strict=False)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    """Tests for dot-notation and default behavior in get."""

    def _make(self, tmp_path, monkeypatch, data):
        """Create a Config instance for get method tests."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, data)
        return Config("test.json", config_dir=str(tmp_path / "configs"), validate_on_load=False)

    def test_simple_key(self, tmp_path, monkeypatch):
        """Return values for top-level keys."""
        cfg = self._make(tmp_path, monkeypatch, {"data_path": "x.csv"})
        assert cfg.get("data_path") == "x.csv"

    def test_dot_notation(self, tmp_path, monkeypatch):
        """Return values for nested keys via dot notation."""
        cfg = self._make(tmp_path, monkeypatch, {"string_cols": {"patterns": {"p": "v"}}})
        assert cfg.get("string_cols.patterns.p") == "v"

    def test_missing_key_returns_default(self, tmp_path, monkeypatch):
        """Return provided default for missing keys."""
        cfg = self._make(tmp_path, monkeypatch, {"data_path": "x.csv"})
        assert cfg.get("nonexistent", "default_val") == "default_val"

    def test_none_value_returns_default(self, tmp_path, monkeypatch):
        """When get() yields None, the default is returned."""
        cfg = self._make(tmp_path, monkeypatch, {"data_path": "x.csv"})
        assert cfg.get("use_cols") is None

    def test_non_dict_mid_path_returns_default(self, tmp_path, monkeypatch):
        """When a mid-path value is not a dict, return default."""
        cfg = self._make(tmp_path, monkeypatch, {"data_path": "x.csv"})
        # data_path is a string, so traversing further is not a dict
        assert cfg.get("data_path.nested", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# __getitem__ / __contains__ / data property
# ---------------------------------------------------------------------------


class TestDictLikeAccess:
    """Tests for dict-like behavior of Config objects."""

    def _make(self, tmp_path, monkeypatch):
        """Create a Config instance for dict-like access tests."""
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path, {"data_path": "x.csv"})
        return Config("test.json", config_dir=str(tmp_path / "configs"), validate_on_load=False)

    def test_getitem_existing_key(self, tmp_path, monkeypatch):
        """Return existing values through __getitem__."""
        cfg = self._make(tmp_path, monkeypatch)
        assert cfg["data_path"] == "x.csv"

    def test_getitem_missing_key_raises(self, tmp_path, monkeypatch):
        """Raise KeyError through __getitem__ for missing keys."""
        cfg = self._make(tmp_path, monkeypatch)
        with pytest.raises(KeyError):
            _ = cfg["nonexistent"]

    def test_contains_true(self, tmp_path, monkeypatch):
        """Report True from __contains__ for existing keys."""
        cfg = self._make(tmp_path, monkeypatch)
        assert "data_path" in cfg

    def test_contains_false(self, tmp_path, monkeypatch):
        """Report False from __contains__ for missing keys."""
        cfg = self._make(tmp_path, monkeypatch)
        assert "missing_key" not in cfg

    def test_data_property(self, tmp_path, monkeypatch):
        """Expose normalized config through data property."""
        cfg = self._make(tmp_path, monkeypatch)
        assert isinstance(cfg.data, dict)
        assert cfg.data["data_path"] == "x.csv"
