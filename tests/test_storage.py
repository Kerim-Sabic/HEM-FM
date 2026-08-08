from copy import deepcopy

import pytest

from hemfm.config import assert_local_mutable_paths, load_config


def test_protocol_mutable_paths_are_local():
    config = load_config()
    assert_local_mutable_paths(config)


def test_network_run_root_is_rejected():
    config = load_config()
    bad = deepcopy(config)
    bad["paths"]["run_root"] = r"Z:\hemfm-v4"
    with pytest.raises(ValueError):
        assert_local_mutable_paths(bad)

