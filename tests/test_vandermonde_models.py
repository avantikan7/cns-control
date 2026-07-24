import json

import numpy as np
import pytest

from cns_control.vandermonde import (
    load_vandermonde_model,
    save_vandermonde_model,
)


def test_multi_objective_save_preserves_other_calibrations(tmp_path):
    path = tmp_path / "vandermonde.json"
    c_10x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c_40x = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    save_vandermonde_model(
        path, c_10x, 1, img_center=[512, 672], objective="10X"
    )
    save_vandermonde_model(
        path, c_40x, 1, img_center=[512, 672], objective="40X"
    )

    data = json.loads(path.read_text())
    assert data["schema_version"] == 2
    assert set(data["objectives"]) == {"10X", "40X"}
    loaded_10x, degree_10x = load_vandermonde_model(path, objective="10X")
    loaded_40x, degree_40x = load_vandermonde_model(path, objective="40X")
    np.testing.assert_array_equal(loaded_10x, c_10x)
    np.testing.assert_array_equal(loaded_40x, c_40x)
    assert degree_10x == degree_40x == 1


def test_multi_objective_load_requires_matching_objective(tmp_path):
    path = tmp_path / "vandermonde.json"
    save_vandermonde_model(path, np.ones((3, 2)), 1, objective="20X")

    with pytest.raises(
        ValueError, match="objective calibration key is required"
    ):
        load_vandermonde_model(path)
    with pytest.raises(
        ValueError, match="No Vandermonde calibration.*available.*20X"
    ):
        load_vandermonde_model(path, objective="40X")


def test_legacy_single_objective_model_still_loads(tmp_path):
    path = tmp_path / "legacy.json"
    coefficients = [[1, 2], [3, 4], [5, 6]]
    path.write_text(json.dumps({"degree": 1, "C": coefficients}))

    loaded, degree = load_vandermonde_model(path, objective="10X")

    np.testing.assert_array_equal(loaded, coefficients)
    assert degree == 1


def test_saving_named_objective_migrates_legacy_file(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"degree": 1, "C": [[0, 0]] * 3}))
    coefficients = np.arange(6).reshape(3, 2)

    save_vandermonde_model(path, coefficients, 1, objective="10X")

    data = json.loads(path.read_text())
    assert set(data) == {
        "schema_version", "objectives", "unassigned_legacy_model"
    }
    assert set(data["objectives"]) == {"10X"}
    assert data["unassigned_legacy_model"]["degree"] == 1
    np.testing.assert_array_equal(
        data["objectives"]["10X"]["C"], coefficients
    )
