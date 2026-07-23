from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from cns_control import utils


@dataclass(frozen=True)
class FakePosition:
    x: float
    y: float
    z: float = 3.0

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@dataclass(frozen=True)
class FakeChannel:
    config: str
    exposure: float

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@dataclass(frozen=True)
class FakeSequence:
    stage_positions: tuple[FakePosition, ...]
    channels: tuple[FakeChannel, ...]

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class FakeCore:
    def __init__(self):
        self.run_calls = []

    def getXYPosition(self):
        return 10.0, 20.0

    def run_mda(self, sequence):
        self.run_calls.append(sequence)


class FakePoints:
    def __init__(self):
        self.data = []

    def add(self, point):
        self.data.append(point)


class FakeSource:
    def __init__(self):
        self._points = FakePoints()


class FakeLayer:
    def __init__(self, data, name, visible):
        self.data = data
        self.name = name
        self.visible = visible


class FakeLayers:
    def __init__(self):
        self._layers = {}

    def __getitem__(self, name):
        return self._layers[name]

    def remove(self, layer):
        del self._layers[layer.name]


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayers()
        self.add_image_calls = 0

    def add_image(self, data, *, name, visible):
        self.add_image_calls += 1
        layer = FakeLayer(data, name, visible)
        self.layers._layers[name] = layer
        return layer


class FakeMDASettings:
    def __init__(self, sequence_holder):
        self._sequence_holder = sequence_holder

    def setValue(self, sequence):
        self._sequence_holder[0] = sequence


class FakeDock:
    def __init__(self, settings):
        self._children = [None, None, None, None, settings]

    def children(self):
        return self._children


class FakeMainWindow:
    def __init__(self, sequence_holder):
        settings = FakeMDASettings(sequence_holder)
        self._dock_widgets = {"MDA": FakeDock(settings)}


@pytest.fixture
def grid_setup(monkeypatch):
    original_channels = (
        FakeChannel("BF", 10.0),
        FakeChannel("GFP", 25.0),
    )
    sequence_holder = [
        FakeSequence((FakePosition(10.0, 20.0),), original_channels)
    ]
    core = FakeCore()
    viewer = FakeViewer()
    main_window = FakeMainWindow(sequence_holder)

    monkeypatch.setattr(
        utils, "_get_seq_from_napari", lambda _window: sequence_holder[0]
    )
    monkeypatch.setattr(
        utils,
        "create_point_sources",
        lambda *_args, **_kwargs: [FakeSource()],
    )

    return core, viewer, main_window, original_channels


def run_grid(
    grid_setup, *, preview_channel, use_placeholder=False, x_range=1.0,
    corner_positions=None, x_count=None, y_count=None,
    tilt_reference_points=None,
    tilt_degree=1,
    snake_axis=None,
):
    core, viewer, main_window, _channels = grid_setup
    return utils.grid_point_selections(
        core,
        viewer,
        main_window,
        point_transformer=object(),
        fov_x=111,
        fov_y=222,
        x_range=x_range,
        y_range=0.0,
        x_step=1.0,
        y_step=1.0,
        repeats=2,
        preview_channel=preview_channel,
        use_placeholder=use_placeholder,
        corner_positions=corner_positions,
        x_count=x_count,
        y_count=y_count,
        tilt_reference_points=tilt_reference_points,
        tilt_degree=tilt_degree,
        snake_axis=snake_axis,
    )


def test_centered_grid_still_uses_current_stage_xy(grid_setup):
    _sources, _autofocus_p, sequence = run_grid(
        grid_setup, preview_channel=None
    )

    assert [(p.x, p.y) for p in sequence.stage_positions] == [
        (9.0, 20.0), (10.0, 20.0), (11.0, 20.0)
    ]
    assert all(position.z == 3.0 for position in sequence.stage_positions)


def test_corner_grid_normalizes_opposite_corner_directions(grid_setup):
    _sources, autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((12.0, 21.0), (10.0, 19.0)),
    )

    assert [(p.x, p.y) for p in sequence.stage_positions] == [
        (10.0, 19.0), (10.0, 20.0), (10.0, 21.0),
        (11.0, 19.0), (11.0, 20.0), (11.0, 21.0),
        (12.0, 19.0), (12.0, 20.0), (12.0, 21.0),
    ]
    np.testing.assert_array_equal(autofocus_p, np.arange(9))


def test_corner_grid_rejects_invalid_positions(grid_setup):
    with pytest.raises(ValueError, match="two finite"):
        run_grid(
            grid_setup,
            preview_channel=None,
            corner_positions=((1.0, 2.0), (np.nan, 4.0)),
        )


def test_grid_axis_includes_both_bounds_without_exceeding_max_step():
    axis = utils._grid_axis(0.0, 10.0, 6.0)

    np.testing.assert_allclose(axis, [0.0, 5.0, 10.0])


def test_grid_axis_uses_exact_point_count_including_endpoints():
    axis = utils._grid_axis(0.0, 10.0, count=6)

    np.testing.assert_allclose(axis, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_corner_grid_uses_requested_xy_point_counts(grid_setup):
    _sources, autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((10.0, 19.0), (12.0, 21.0)),
        x_count=2,
        y_count=5,
    )

    assert len(sequence.stage_positions) == 10
    assert sorted({position.x for position in sequence.stage_positions}) == [
        10.0, 12.0
    ]
    assert sorted({position.y for position in sequence.stage_positions}) == [
        19.0, 19.5, 20.0, 20.5, 21.0
    ]
    np.testing.assert_array_equal(autofocus_p, np.arange(10))


def test_grid_rejects_only_one_axis_count(grid_setup):
    with pytest.raises(ValueError, match="provided together"):
        run_grid(
            grid_setup,
            preview_channel=None,
            x_count=3,
        )


def test_x_fast_snake_reverses_alternating_y_rows(grid_setup):
    _sources, _autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((0.0, 0.0), (2.0, 2.0)),
        x_count=3,
        y_count=3,
        snake_axis="x",
    )

    assert [(position.x, position.y) for position in sequence.stage_positions] == [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (2.0, 1.0), (1.0, 1.0), (0.0, 1.0),
        (0.0, 2.0), (1.0, 2.0), (2.0, 2.0),
    ]


def test_y_fast_snake_reverses_alternating_x_columns_and_keeps_channels(
    grid_setup,
):
    _core, _viewer, _main_window, original_channels = grid_setup
    _sources, _autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((0.0, 0.0), (2.0, 2.0)),
        x_count=3,
        y_count=3,
        snake_axis="y",
    )

    assert [(position.x, position.y) for position in sequence.stage_positions] == [
        (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
        (1.0, 2.0), (1.0, 1.0), (1.0, 0.0),
        (2.0, 0.0), (2.0, 1.0), (2.0, 2.0),
    ]
    assert sequence.channels == original_channels


def test_tilt_plane_fit_recovers_slopes_and_offset():
    references = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 3.0),
        (0.0, 1.0, 4.0),
        (1.0, 1.0, 6.0),
    ]

    origin, coefficients, rmse = utils._fit_tilt_plane(references)

    np.testing.assert_allclose(origin, [0.5, 0.5])
    np.testing.assert_allclose(coefficients, [2.0, 3.0, 3.5])
    assert rmse < 1e-12


def test_grid_positions_receive_fitted_tilt_z(grid_setup):
    references = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 3.0),
        (0.0, 1.0, 4.0),
        (1.0, 1.0, 6.0),
    ]
    _sources, _autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((0.0, 0.0), (1.0, 1.0)),
        x_count=2,
        y_count=2,
        tilt_reference_points=references,
    )

    np.testing.assert_allclose(
        [(position.x, position.y, position.z)
         for position in sequence.stage_positions],
        [(0.0, 0.0, 1.0), (0.0, 1.0, 4.0),
         (1.0, 0.0, 3.0), (1.0, 1.0, 6.0)],
    )


def test_tilt_plane_rejects_collinear_xy_references():
    with pytest.raises(ValueError, match="do not constrain"):
        utils._fit_tilt_plane([
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 2.0),
            (2.0, 2.0, 3.0),
        ])


def test_tilt_plane_requires_three_references():
    with pytest.raises(ValueError, match="at least 3"):
        utils._fit_tilt_plane([
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
        ])


def test_degree_two_vandermonde_corrects_curved_grid_z(grid_setup):
    def curved_z(x, y):
        return 1.0 + 2.0*x + 3.0*y + 4.0*x*x + 5.0*x*y + 6.0*y*y

    references = [
        (x, y, curved_z(x, y))
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
    ]
    _sources, _autofocus_p, sequence = run_grid(
        grid_setup,
        preview_channel=None,
        corner_positions=((-1.0, -1.0), (1.0, 1.0)),
        x_count=3,
        y_count=3,
        tilt_reference_points=references,
        tilt_degree=2,
    )

    for position in sequence.stage_positions:
        assert np.isclose(position.z, curved_z(position.x, position.y))


def test_degree_two_tilt_fit_requires_six_references():
    with pytest.raises(ValueError, match="requires at least 6"):
        utils._fit_tilt_surface([
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 3.0),
            (1.0, 1.0, 4.0),
        ], degree=2)


def test_raman_grid_without_placeholder_runs_prescan(grid_setup):
    core, viewer, _main_window, original_channels = grid_setup
    run_grid(grid_setup, preview_channel=None, use_placeholder=True)

    sources, autofocus_p, sequence = run_grid(
        grid_setup, preview_channel=None
    )

    assert core.run_calls == [sequence]
    with pytest.raises(KeyError):
        viewer.layers[utils._GRID_PLACEHOLDER_LAYER]
    assert sequence.channels == original_channels
    np.testing.assert_array_equal(autofocus_p, np.arange(3))
    assert len(sources[0]._points.data) == 6
    assert [point[1] for point in sources[0]._points.data] == [0, 0, 1, 1, 2, 2]
    assert all(point[-2:] == [222, 111] for point in sources[0]._points.data)


def test_raman_grid_uses_placeholders_without_running_mda(grid_setup):
    core, viewer, _main_window, original_channels = grid_setup

    _sources, _autofocus_p, sequence = run_grid(
        grid_setup, preview_channel=None, use_placeholder=True
    )

    assert core.run_calls == []
    placeholder = viewer.layers[utils._GRID_PLACEHOLDER_LAYER]
    assert placeholder.data.shape == (1, 3, 1, 1, 1, 1)
    assert placeholder.data.dtype == np.uint8
    assert placeholder.visible is False
    assert viewer.add_image_calls == 1
    assert sequence.channels == original_channels


def test_raman_grid_updates_existing_placeholder(grid_setup):
    _core, viewer, _main_window, _channels = grid_setup
    run_grid(grid_setup, preview_channel=None, use_placeholder=True)

    run_grid(
        grid_setup,
        preview_channel=None,
        use_placeholder=True,
        x_range=2.0,
    )

    placeholder = viewer.layers[utils._GRID_PLACEHOLDER_LAYER]
    assert placeholder.data.shape == (1, 5, 1, 1, 1, 1)
    assert viewer.add_image_calls == 1


def test_real_channel_previews_once_and_preserves_raman_channels(grid_setup):
    core, viewer, _main_window, original_channels = grid_setup
    run_grid(grid_setup, preview_channel=None, use_placeholder=True)

    _sources, _autofocus_p, sequence = run_grid(
        grid_setup, preview_channel="GFP"
    )

    with pytest.raises(KeyError):
        viewer.layers[utils._GRID_PLACEHOLDER_LAYER]
    assert len(core.run_calls) == 1
    preview_sequence = core.run_calls[0]
    assert len(preview_sequence.channels) == 1
    assert preview_sequence.channels[0].config == "GFP"
    assert preview_sequence.channels[0].exposure == 10.0
    assert sequence.channels == original_channels
