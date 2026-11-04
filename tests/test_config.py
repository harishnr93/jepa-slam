import json
import logging

import pytest

from latent_loop.utils.config import (
    REPO_ROOT,
    deep_merge,
    get_by_dotted_key,
    load_config,
    parse_overrides,
    resolve_path,
)
from latent_loop.utils.logging import get_logger, setup_logging


def test_deep_merge_is_recursive_and_non_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"c": 20}, "e": 5})
    assert merged == {"a": {"b": 1, "c": 20}, "d": 3, "e": 5}
    assert base["a"]["c"] == 2


def test_default_config_loads():
    config = load_config()
    assert config["dataset"]["name"] == "synthetic"
    assert config["dataset"]["sampling"]["stride"] == 1


def test_kitti_config_extends_default():
    config = load_config("configs/kitti.yaml")
    assert config["dataset"]["name"] == "kitti"
    assert config["dataset"]["root"] == "data/raw/kitti/dataset"
    assert config["dataset"]["sampling"]["start"] == 0  # inherited
    assert config["visualization"]["gif_fps"] == 10  # inherited
    assert "extends" not in config


def test_overrides_are_yaml_typed():
    config = load_config(
        overrides=["dataset.sampling.stride=5", "dataset.sequence='03'", "new.key=[1, 2]"]
    )
    assert config["dataset"]["sampling"]["stride"] == 5
    assert config["dataset"]["sequence"] == "03"
    assert get_by_dotted_key(config, "new.key") == [1, 2]
    assert get_by_dotted_key(config, "missing.key", "fallback") == "fallback"
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_circular_extends_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_resolve_path():
    assert resolve_path("data/raw") == REPO_ROOT / "data/raw"
    assert resolve_path("/tmp/x").as_posix() == "/tmp/x"
    assert resolve_path(None) is None


def test_structured_logging_writes_json_lines(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("DEBUG", log_file)
    get_logger("tests").info("hello", extra={"frames": 3})
    for handler in logging.getLogger("latent_loop").handlers:
        handler.flush()
    record = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert record["message"] == "hello" and record["frames"] == 3
    assert record["logger"] == "latent_loop.tests" and record["level"] == "INFO"
    setup_logging("INFO")  # detach the file handler again
