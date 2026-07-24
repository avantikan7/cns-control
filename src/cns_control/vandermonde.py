"""Persistence helpers for pixel-to-stage Vandermonde calibrations."""

import json

import numpy as np


def load_vandermonde_model(json_path, objective=None):
    """Load coefficients for one objective from a Vandermonde JSON file.

    Multi-objective files require an exact objective calibration key.
    Legacy files containing top-level ``degree`` and ``C`` keys remain
    readable for backward compatibility.
    """
    with open(json_path) as f:
        model = json.load(f)

    if "objectives" in model:
        if not objective:
            raise ValueError(
                "An objective calibration key is required for this "
                "multi-objective "
                "Vandermonde model"
            )
        objective = str(objective)
        models = model["objectives"]
        if not isinstance(models, dict):
            raise ValueError(
                "Invalid Vandermonde model: 'objectives' must be an object"
            )
        if objective not in models:
            available = ", ".join(sorted(models)) or "none"
            raise ValueError(
                f"No Vandermonde calibration for objective {objective!r}; "
                f"available objectives: {available}"
            )
        model = models[objective]

    coefficients = np.array(model["C"])
    degree = int(model["degree"])
    return coefficients, degree


def save_vandermonde_model(
    json_path, coefficients, degree, img_center=None, xy_center=None,
    objective=None,
):
    """Save or update one pixel-to-stage calibration.

    Providing ``objective`` writes the version-2 multi-objective format and
    preserves every other existing objective entry. Omitting it writes the
    legacy single-objective format. When converting a legacy file, its
    unidentified calibration is retained under ``unassigned_legacy_model``.
    """
    entry = {
        "degree": int(degree),
        "C": np.asarray(coefficients).tolist(),
    }
    if img_center is not None:
        entry["img_center"] = np.asarray(img_center).tolist()
    if xy_center is not None:
        entry["xy_center"] = np.asarray(xy_center).tolist()

    if objective is None:
        model = entry
    else:
        objective = str(objective).strip()
        if not objective:
            raise ValueError("objective must be a non-empty preset name")

        try:
            with open(json_path) as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = {}

        if "objectives" in existing:
            objectives = existing["objectives"]
            if not isinstance(objectives, dict):
                raise ValueError(
                    "Invalid Vandermonde model: 'objectives' must be an object"
                )
            objectives = dict(objectives)
            model = dict(existing)
        else:
            # A legacy file has no objective identity. The newly fitted,
            # explicitly named calibration becomes the first v2 entry. Keep
            # the old data for recovery rather than guessing its objective.
            objectives = {}
            model = {}
            if "degree" in existing and "C" in existing:
                model["unassigned_legacy_model"] = existing

        objectives[objective] = entry
        model["schema_version"] = 2
        model["objectives"] = objectives

    with open(json_path, "w") as f:
        json.dump(model, f, indent=2)

    objective_msg = f", objective={objective}" if objective is not None else ""
    print(
        f"Saved Vandermonde model (degree={degree}{objective_msg}) "
        f"to {json_path}"
    )
