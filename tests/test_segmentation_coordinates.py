from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from cns_control import utils


@dataclass(frozen=True)
class _Position:
    x: float
    y: float


@dataclass(frozen=True)
class _Sequence:
    stage_positions: tuple[_Position, ...]

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class _Points:
    def __init__(self):
        self.data = []

    def add(self, point):
        self.data.append(point)


class _Core:
    def __init__(self):
        self.run_calls = []
        self.calls = []

    def setConfig(self, group, config):
        self.calls.append(("setConfig", group, config))

    def waitForConfig(self, group, config):
        self.calls.append(("waitForConfig", group, config))

    def setXYPosition(self, _x, _y):
        self.calls.append(("setXYPosition",))

    def waitForSystem(self):
        pass

    def snapImage(self):
        self.calls.append(("snapImage",))

    def getImage(self):
        return np.zeros((8, 10), dtype=np.uint16)

    def run_mda(self, sequence):
        self.run_calls.append(sequence)


def test_automated_selection_converts_napari_yx_to_segmentation_xy(
    monkeypatch,
):
    captured = {}
    sequence = _Sequence((_Position(1.0, 2.0),))

    monkeypatch.setattr(utils, "_get_seq_from_napari", lambda _window: sequence)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)

    def fake_segment(_image, **kwargs):
        captured["circle_center"] = kwargs["circle_center"]
        return np.zeros((8, 10), dtype=np.uint16)

    monkeypatch.setattr(utils, "_segment_single_img", fake_segment)
    monkeypatch.setattr(
        utils,
        "get_n_most_centered_coms",
        lambda *_args, **_kwargs: np.array([[5.0, 2.0]]),
    )
    source = SimpleNamespace(_points=_Points())
    monkeypatch.setattr(
        utils,
        "create_point_sources",
        lambda *_args, **_kwargs: [source],
    )

    core = _Core()
    utils.automated_point_selections(
        core,
        viewer=object(),
        main_window=object(),
        point_transformer=object(),
        N=1,
        center=(5, 2),
        radius=3,
        autofocus_object=None,
        batch=True,
        segmentation_channel="Phase",
    )

    assert captured["circle_center"] == (2, 5)
    assert core.calls[:2] == [
        ("setConfig", "Channel", "Phase"),
        ("waitForConfig", "Channel", "Phase"),
    ]
    assert core.calls.index(
        ("setConfig", "Channel", "Phase")
    ) < core.calls.index(("snapImage",))


def test_centered_coms_default_to_label_image_center():
    labels = np.zeros((8, 12), dtype=np.uint16)
    labels[4, 6] = 1
    labels[7, 11] = 2

    points = utils.get_n_most_centered_coms(
        labels,
        N=1,
        center=None,
        radius=np.inf,
        autofocus_object=None,
    )

    np.testing.assert_array_equal(points, [[4, 6]])
