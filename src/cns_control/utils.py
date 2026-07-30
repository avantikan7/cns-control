import time
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from skimage.draw import disk
from scipy.ndimage import center_of_mass, distance_transform_edt
from tqdm.auto import tqdm

from .vandermonde import load_vandermonde_model


# Values that mean "do not use an autofocus object / layer".
_NO_AUTOFOCUS = (None, "None", "none", "")
_GRID_PLACEHOLDER_LAYER = "Grid position placeholders"


def _save_segmentation_debug(directory, position, image, labels):
    """Save one lossless segmentation input/label pair for debugging."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"position_{int(position):04d}"
    image_path = directory / f"{stem}_input.npy"
    labels_path = directory / f"{stem}_labels.npy"
    np.save(image_path, np.asarray(image), allow_pickle=False)
    np.save(labels_path, np.asarray(labels), allow_pickle=False)
    return image_path, labels_path


def _grid_axis(start, stop, max_step=None, count=None):
    """Return an axis defined by maximum spacing or an exact point count."""
    if count is not None:
        count = int(count)
        if count < 1:
            raise ValueError("grid point count must be at least 1")
        if count == 1:
            return np.array([(start + stop) / 2.0], dtype=float)
        return np.linspace(start, stop, count)

    if max_step is None:
        raise ValueError("max_step is required when count is not provided")
    if max_step <= 0:
        raise ValueError("grid step must be greater than zero")
    span = stop - start
    if span == 0:
        return np.array([start], dtype=float)
    count = int(np.ceil(span / max_step)) + 1
    return np.linspace(start, stop, count)


def _tilt_vandermonde_design(xy, degree):
    """Build [1, x, y, x^2, xy, y^2, ...] through ``degree``."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    x, y = xy[:, 0], xy[:, 1]
    terms = []
    for total_degree in range(degree + 1):
        for x_power in range(total_degree, -1, -1):
            y_power = total_degree - x_power
            terms.append((x ** x_power) * (y ** y_power))
    return np.stack(terms, axis=1)


def _fit_tilt_surface(reference_points, degree=1):
    """Fit a centered/scaled 2D Vandermonde Z surface."""
    degree = int(degree)
    if degree < 1 or degree > 3:
        raise ValueError("tilt fit degree must be between 1 and 3")
    points = np.asarray(reference_points, dtype=float)
    term_count = (degree + 1) * (degree + 2) // 2
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < term_count
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            f"degree {degree} tilt correction requires at least "
            f"{term_count} finite XYZ points"
        )

    origin = points[:, :2].mean(axis=0)
    centered_xy = points[:, :2] - origin
    scale = np.max(np.abs(centered_xy), axis=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("tilt reference XY points must span both X and Y")
    normalized_xy = centered_xy / scale
    design = _tilt_vandermonde_design(normalized_xy, degree)
    coefficients, _residuals, rank, _singular_values = lstsq(
        design, points[:, 2]
    )
    if rank < term_count:
        raise ValueError(
            f"tilt reference XY points do not constrain a degree {degree} fit"
        )
    fitted = design @ coefficients
    errors = fitted - points[:, 2]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return origin, scale, coefficients, rmse


def _evaluate_tilt_surface(xy, model, degree):
    """Evaluate a fitted tilt surface at one or more stage XY positions."""
    origin, scale, coefficients, _rmse = model
    normalized_xy = (np.atleast_2d(xy) - origin) / scale
    return _tilt_vandermonde_design(normalized_xy, degree) @ coefficients


def _fit_tilt_plane(reference_points):
    """Backward-compatible degree-1 tilt fit with physical X/Y slopes."""
    origin, scale, coefficients, rmse = _fit_tilt_surface(
        reference_points, degree=1
    )
    physical_coefficients = np.array([
        coefficients[1] / scale[0],
        coefficients[2] / scale[1],
        coefficients[0],
    ])
    return origin, physical_coefficients, rmse


def _ordered_grid_coordinates(xs, ys, snake_axis=None):
    """Yield grid XY coordinates in raster or axis-selected snake order."""
    if snake_axis not in (None, "x", "y"):
        raise ValueError("snake_axis must be None, 'x', or 'y'")

    if snake_axis == "x":
        for row, gy in enumerate(ys):
            row_xs = xs if row % 2 == 0 else xs[::-1]
            for gx in row_xs:
                yield gx, gy
        return

    for column, gx in enumerate(xs):
        column_ys = (
            ys[::-1] if snake_axis == "y" and column % 2 else ys
        )
        for gy in column_ys:
            yield gx, gy


def _remove_grid_placeholder(viewer):
    """Remove the grid placeholder layer if a previous Raman grid created it."""
    try:
        layer = viewer.layers[_GRID_PLACEHOLDER_LAYER]
    except (KeyError, TypeError):
        return
    viewer.layers.remove(layer)


def _get_seq_from_napari(main_window):
    """Import the napari-MDA adapter only when a sequence is requested."""
    from raman_mda_engine.utils import get_seq_from_napari

    return get_seq_from_napari(main_window)


def _get_mda_widget_from_napari(main_window):
    """Import the napari-MDA widget adapter only when it is needed."""
    from raman_mda_engine.utils import get_mda_widget_from_napari

    return get_mda_widget_from_napari(main_window)


def _segment_single_img(image, **kwargs):
    """Load the optional Cellpose stack only for automated segmentation."""
    from raman_mda_engine.aiming.autotracking import segment_single_img

    return segment_single_img(image, **kwargs)


def _is_no_autofocus(autofocus_object) -> bool:
    """True when the caller has asked for no autofocus object."""
    return autofocus_object in _NO_AUTOFOCUS


def _vandermonde_design(src, degree):
    """
    Build a 2D polynomial design matrix from src=(x, y) up to `degree`,
    including cross terms (e.g. degree=2 -> [1, x, y, x^2, xy, y^2]).
    `src` can be a single (2,) point or an (n, 2) array.
    """
    src = np.atleast_2d(src)
    x, y = src[:, 0], src[:, 1]
    terms = []
    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            terms.append((x**i) * (y**j))
    return np.stack(terms, axis=1)


def apply_vandermonde_model(src, C, degree):
    """
    Apply a fitted Vandermonde model to a single (x, y) pixel offset,
    returning the corresponding (dx, dy) stage displacement.
    """
    D = _vandermonde_design(src, degree)
    return (D @ C).ravel()


# ---------------------------------------------------------------------------
# napari layer / mask helpers
# ---------------------------------------------------------------------------

def add_mask_with_hole(
    viewer,
    image_size,
    circle_radius=200,
    color=(255, 0, 0),
    alpha=50,
    circle_center=None,
    small_circle_radius=10,
    small_circle_color=(0, 255, 0),
    small_circle_alpha=255
):
    """
    Adds a semi-transparent colored mask with a transparent circular hole and a solid
    colored small circle (dot) in the center to a napari viewer.

    Parameters:
    -----------
    viewer : napari.Viewer
        The napari viewer instance.
    image_size : tuple
        The (height, width) of the image.
    circle_radius : int
        Radius of the main transparent hole.
    color : tuple
        RGB color of the main mask.
    alpha : int
        Transparency of the main mask (0-255).
    circle_center : tuple or None
        (y, x) center of both holes. Defaults to image center.
    small_circle_radius : int
        Radius of the small central dot.
    small_circle_color : tuple
        RGB color of the small dot.
    small_circle_alpha : int
        Alpha for the small dot (default fully opaque).
    """

    image_size = tuple(image_size)
    rgba_image = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)

    # Fill with the main mask color and alpha
    rgba_image[:, :, :3] = color
    rgba_image[:, :, 3] = alpha

    # Set center
    if circle_center is None:
        circle_center = (image_size[0] / 2, image_size[1] / 2)

    # Transparent main circular hole
    rr_main, cc_main = disk(circle_center, circle_radius, shape=rgba_image.shape[:2])
    rgba_image[rr_main, cc_main, :] = 0

    # Solid small colored circle in the center
    rr_small, cc_small = disk(circle_center, small_circle_radius, shape=rgba_image.shape[:2])
    rgba_image[rr_small, cc_small, :3] = small_circle_color
    rgba_image[rr_small, cc_small, 3] = small_circle_alpha

    # Add to napari
    viewer.add_image(rgba_image, rgb=True)


def create_point_sources(viewer,
                         point_transformer,
                         broadcast_dims=(0, 2, 3),
                         ndim=6,
                         size=35,
                         names=['cells', 'autofocus'],
                         colors=['#aa0000ff', 'springgreen'],
                         roles=None,
                         display_names=None,
                         source_ids=None):
    """
    Creates and adds multiple point source layers to a Napari viewer.

    Parameters:
    -----------
    viewer : napari.Viewer
        The Napari viewer where the point sources will be added.
    point_transformer : callable
        A transformation function applied to the point coordinates.
    broadcast_dims : tuple, optional
        Dimensions along which the points should be broadcasted (default: (0, 2, 3)).
    ndim : int, optional
        Number of dimensions for the point sources (default: 6).
    size : float, optional
        Size of the points in pixels (default: 35).
    names : list of str, optional
        Names of the point source layers (default: ['cells', 'autofocus']).
    colors : list of str, optional
        List of colors for the point sources, specified in hex format.
    roles : list of str or None, optional
        Explicit semantic roles such as ``"cell"`` and ``"autofocus"``.
        When omitted, sources infer their role from their legacy names.
    display_names : list of str or None, optional
        User-facing names. Source ``name`` remains the stable saved
        designation used by the Raman writer.
    source_ids : list of str or None, optional
        Stable IDs used in acquisition metadata.

    Returns:
    --------
    list of PointsLayerSource
        A list of `PointsLayerSource` objects, each corresponding to a created point layer.
    """
    # These imports initialize napari's layer stack. Keep them out of module
    # scope so lightweight helpers (for example ``filter_mean``) remain usable
    # without importing napari, the MDA engine, Cellpose, or PyTorch.
    from napari_broadcastable_points import BroadcastablePoints
    from raman_mda_engine.aiming import PointsLayerSource

    sources = []
    roles = [None] * len(names) if roles is None else list(roles)
    display_names = (
        [None] * len(names) if display_names is None else list(display_names)
    )
    source_ids = (
        [None] * len(names) if source_ids is None else list(source_ids)
    )
    if not (
        len(colors) == len(names)
        and len(roles) == len(names)
        and len(display_names) == len(names)
        and len(source_ids) == len(names)
    ):
        raise ValueError(
            "names, colors, roles, display_names, and source_ids must "
            "have the same length"
        )
    for name, color, role, display_name, source_id in zip(
        names, colors, roles, display_names, source_ids
    ):
        points = BroadcastablePoints(
                    None,
                    #               t, c, z
                    broadcast_dims=broadcast_dims,
                    ndim=ndim,
                    name=name,
                    size=size,
                    face_color=color,
                    border_color="#5500ffff",
                )
        viewer.add_layer(points)
        sources.append(
            PointsLayerSource(
                points,
                name=name,
                transformer=point_transformer,
                role=role,
                display_name=display_name,
                source_id=source_id,
                color=color,
            )
        )
    return sources


def _create_fixed_grid_sources(
    core,
    point_transformer,
    fov_x,
    fov_y,
    repeats,
    include_autofocus,
):
    """Create fixed grid sources without allocating napari points per position."""
    from raman_mda_engine.aiming import FixedPointsSource

    image_shape = core.getImageWidth(), core.getImageHeight()
    cell_points = np.repeat(
        np.array([[fov_y, fov_x]], dtype=float), repeats, axis=0
    )
    sources = [
        FixedPointsSource(
            cell_points,
            name="cells",
            img_shape=image_shape,
            transformer=point_transformer,
        )
    ]
    if include_autofocus:
        sources.append(
            FixedPointsSource(
                np.array([[fov_y, fov_x]], dtype=float),
                name="autofocus",
                img_shape=image_shape,
                transformer=point_transformer,
            )
        )
    return sources


def filter_mean(spec, f=2):
    mean_spec = np.mean(spec, axis=0)
    std_spec = np.std(spec, axis=0)

    # Create a mask for values within 3 standard deviations
    mask = (spec >= (mean_spec - f * std_spec)) & (spec <= (mean_spec + f * std_spec))

    # Compute the mean while ignoring values outside 3 std
    filtered_mean_spec = np.sum(spec * mask, axis=0) / np.sum(mask, axis=0)
    return filtered_mean_spec


def set_up_new_seq(main_window, point_transformer, engine, seq=None, total_exposure=1000, batch=False, z_plan='all'):
    """
    Configures a new Raman sequence with updated metadata and exposure settings.

    Parameters:
    -----------
    main_window : object
        The main Napari window instance used to retrieve the sequence.
    point_transformer : object
        An object that provides the `multiplier` attribute for adjusting exposure time.
    engine : object
        The engine controlling Raman measurements, which stores default exposure settings.
    total_exposure : int, optional (default=1000)
        The total exposure time in milliseconds (ms) for Raman acquisition.
    batch : bool, optional (default=False)
        If False, the exposure time is divided by `point_transformer.multiplier`.
        If True, `total_exposure` is used directly without modification.
    z_plan : str, optional (default='all')
        Determines which Z positions to acquire Raman spectra from:
        - `'all'`: Acquire Raman data at all Z positions.
        - `'middle'`: Acquire Raman data only at the middle Z position.
        - Any other value raises a `ValueError`.

    Returns:
    --------
    new_seq : object
        A modified sequence object with updated metadata specifying the Raman acquisition plan.
    """

    if not batch:
        engine.default_rm_exposure = total_exposure / point_transformer.multiplier
    else:
        engine.default_rm_exposure = total_exposure

    if engine.default_rm_exposure < 73.8:
        raise ValueError('Minimal exposure time per Raman collection is 0.0738s')

    if seq is None:
        seq = _get_seq_from_napari(main_window)
    new_meta = dict(seq.metadata)
    num_z = seq.z_plan.num_positions()
    # add in raman metadata to do raman
    if z_plan == 'all':
        new_meta["raman"] = {
            "channel": "RM",
            "z": np.arange(num_z).tolist(),
        }  # do all Zs
    elif z_plan == 'middle':
        new_meta["raman"] = {
            "channel": "RM",
            "z": [num_z//2],
        }  # only do the middle Z
    else:
        raise ValueError('Please choose a valid z_plan: middle or all')
    new_seq = seq.replace(metadata=new_meta)
    return new_seq


def find_clear_center_point(mask, threshold=20):
    background = mask == 0
    dist_map = distance_transform_edt(background)
    center = np.array(mask.shape) / 2
    valid_points = np.argwhere(dist_map >= threshold)

    if len(valid_points) == 0:
        raise ValueError("No point found that meets the threshold distance.")

    dists_to_center = np.linalg.norm(valid_points - center, axis=1)
    best_idx = np.argmin(dists_to_center)

    return np.array(valid_points[best_idx])


def get_n_most_centered_coms(
    label_mask,
    N=10,
    center=None,
    radius=250,
    autofocus_object='glass',
    bkd_threshold=50
):
    """
    Returns up to N center-of-mass points closest to a given center,
    within a distance threshold.

    When `autofocus_object` is one of the focus targets, a clear-background
    point is inserted at index 0 (used for autofocus). When it is None / "None"
    (or 'cell'), NO autofocus point is prepended -- every returned point is a
    cell center-of-mass.

    Parameters:
    -----------
    label_mask : 2D array
        Labeled mask image.
    N : int
        Number of points to return.
    center : tuple or None
        (y, x) center to compare distances. Defaults to the image center.
    radius : float
        Max distance from center to include a point. Pass np.inf for no limit.
    autofocus_object : str or None
        If a focus target ('glass', 'quartz', 'laser', 'software'), adds a clear
        point at index 0. If None / "None", no autofocus point is added.
    bkd_threshold : int
        Passed to find_clear_center_point.

    Returns:
    --------
    np.ndarray of shape (<=N, 2): list of center-of-mass coordinates
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]  # ignore background

    if center is None:
        center = np.array(label_mask.shape) / 2
    else:
        center = np.array(center)

    coms = []
    dists = []

    for label in labels:
        com = center_of_mass(np.ones_like(label_mask), labels=label_mask, index=label)
        com = np.array(com)
        dist = np.linalg.norm(com - center)
        if dist <= radius:
            coms.append(com)
            dists.append(dist)

    # Sort by distance to center
    sorted_coms = [com for _, com in sorted(zip(dists, coms), key=lambda x: x[0])]

    # Optionally insert autofocus point (skipped entirely when no autofocus).
    if autofocus_object in ['glass', 'quartz', 'laser', 'software']:
        clear_com = find_clear_center_point(label_mask, threshold=bkd_threshold)
        sorted_coms.insert(0, clear_com)

    return np.array(sorted_coms[:N])


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def automated_point_selections(
    core, viewer, main_window, point_transformer, N,
    center=None, radius=250, autofocus_object='glass', bkd_thres=50, batch=True,
    center_cell=False, vandermonde_model_path=None, cellpose_model='cyto2',
    objective=None, segmentation_channel='BF', segmentation_debug_dir=None,
    cell_diameter=15,
):
    """
    Parameters
    ----------
    center_cell : bool, optional (default=False)
        If False, behaves exactly as before: for each original stage
        position, up to N cells are detected at their raw pixel location
        (within `radius` of `center`), and the position is repeated once
        per detected cell.

        If True, up to N cells are still detected per original stage
        position -- but the search is UNBOUNDED (the `radius` argument is
        ignored in this mode, since we want every cell in the FOV, not just
        ones near center). Instead of repeating the SAME stage position for
        each detected cell, a NEW, independently-corrected stage position is
        generated per cell using a pre-fit Vandermonde pixel-offset ->
        stage-offset model, such that each cell ends up exactly under the
        image center. So N=100 means: try to find up to 100 cells in that
        one original FOV, then split it into up to 100 new stage positions,
        each one centered on a different cell.

        The 'cells' point recorded for every new position is therefore
        always the image center (that's where the cell will sit once the
        stage moves there). If `autofocus_object` is set, the SAME raw
        autofocus point (detected once for that original FOV, at its
        original, un-shifted pixel location) is recorded against every new
        position split from it -- autofocus is a property of the FOV /
        reference surface, not of an individual cell.

    vandermonde_model_path : str or None
        Path to a JSON file with the fitted model (see
        `load_vandermonde_model`). Required when `center_cell=True`. Must be
        fit on (pixel offset from image center) -> (stage offset) pairs.
    objective : str or None
        Objective device state index, represented as a string, used to choose
        one calibration from a multi-objective model. Ignored by legacy files.
    segmentation_channel : str, optional
        Micro-Manager imaging preset used for the Cellpose input frame.
        Defaults to ``"BF"`` so selection cannot inherit an active RM laser
        preset from autofocus.
    segmentation_debug_dir : path-like or None, optional
        When provided, save the exact camera input and Cellpose label mask for
        every original stage position as lossless ``.npy`` arrays.
    cell_diameter : float or None, optional
        Expected cell diameter in pixels in the original camera image.
        ``None`` uses the diameter stored with the Cellpose model.

    Returns
    -------
    sources, autofocus_p, new_seq
        Same 3-tuple contract as before. `autofocus_p` is simply
        `np.arange(len(new_positions))` when `center_cell=True`, since every
        new position is already its own independent FOV (no repeats/groups
        to index into).
    """
    no_autofocus = _is_no_autofocus(autofocus_object)

    if center_cell:
        if vandermonde_model_path is None:
            raise ValueError("vandermonde_model_path must be provided when center_cell=True")
        C, degree = load_vandermonde_model(
            vandermonde_model_path, objective=objective
        )

    seq = _get_seq_from_napari(main_window)
    images = []
    masks = []
    points = []          # per ORIGINAL position: raw detected point(s), pixel coords (y, x)
    new_positions = []   # only populated when center_cell=True -- one entry per detected cell

    # Selection must not inherit the RM preset left by laser autofocus.  In
    # particular, that preset holds the Raman laser shutter open and its spot
    # can hide a cell from Cellpose.  Switch explicitly to the requested
    # imaging preset before any frame used for segmentation is acquired.
    core.setConfig("Channel", segmentation_channel)
    core.waitForConfig("Channel", segmentation_channel)

    for i in tqdm(range(len(seq.stage_positions))):
        orig_pos = seq.stage_positions[i]
        core.setXYPosition(orig_pos.x, orig_pos.y)
        core.waitForSystem()
        time.sleep(5)
        core.snapImage()
        core.waitForSystem()
        image = core.getImage()
        core.waitForSystem()
        time.sleep(1)
        images.append(image)
        circle_center_xy = (
            None if center is None else (center[1], center[0])
        )
        mask = _segment_single_img(
            image,
            scale=1,
            cellpose_model=cellpose_model,
            circle_center=circle_center_xy,
            circle_radius=radius,
            diameter=cell_diameter,
        )
        masks.append(mask)
        if segmentation_debug_dir is not None:
            image_path, labels_path = _save_segmentation_debug(
                segmentation_debug_dir,
                i,
                image,
                mask,
            )
            print(
                f"[segmentation debug] position={i}, "
                f"labels={int(np.max(mask)) if np.size(mask) else 0}, "
                f"diameter={cell_diameter if cell_diameter is not None else 'model training'}, "
                f"input={image_path}, mask={labels_path}"
            )

        if center_cell:
            # try to find up to N cells (+1 autofocus point if requested);
            # radius is unbounded so we don't miss cells far from center.
            n_needed = N if no_autofocus else N + 1
            found = get_n_most_centered_coms(
                mask, N=n_needed, center=center, radius=np.inf,
                autofocus_object=autofocus_object, bkd_threshold=bkd_thres,
            )
            points.append(found)

            if found.size == 0:
                continue

            cell_slice = slice(0, None) if no_autofocus else slice(1, None)
            cells_yx = found[cell_slice, :]

            img_center_yx = (
                np.array(image.shape[:2]) / 2.0 if center is None else np.array(center)
            )

            # one new, independently-corrected stage position PER detected cell
            for cell_yx in cells_yx:
                offset_yx = cell_yx - img_center_yx
                offset_xy = np.array([offset_yx[1], offset_yx[0]])
                stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)

                print(f"[debug] cell_yx={cell_yx}, img_center_yx={img_center_yx}, "
                    f"offset_yx={offset_yx}, offset_xy={offset_xy}, "
                    f"stage_dx={stage_dx:.3f}, stage_dy={stage_dy:.3f}")
                new_positions.append(
                    orig_pos.replace(
                        x=float(orig_pos.x - stage_dx),
                        y=float(orig_pos.y - stage_dy),
                    )
                )
        else:
            points.append(
                get_n_most_centered_coms(
                    mask, N=N, center=center, radius=radius,
                    autofocus_object=autofocus_object, bkd_threshold=bkd_thres,
                )
            )

    images = np.array(images)
    masks = np.array(masks)

    # ------------------------- center_cell = True -------------------------
    if center_cell:
        new_seq = seq.replace(stage_positions=new_positions)

        if no_autofocus:
            sources = create_point_sources(
                viewer, point_transformer, size=15,
                names=['cells'], colors=['#aa0000ff'],
            )
        else:
            sources = create_point_sources(viewer, point_transformer, size=15)

        core.run_mda(new_seq)

        # image center is the same for every FOV (same camera/detector)
        img_center_yx = np.array(images.shape[1:3]) / 2.0 if center is None else np.array(center)

        p = 0
        for found in points:
            if found.size == 0:
                continue
            cell_slice = slice(0, None) if no_autofocus else slice(1, None)
            n_cells_here = found[cell_slice, :].shape[0]
            autofocus_yx = None if no_autofocus else found[0]

            for _ in range(n_cells_here):
                if not no_autofocus:
                    sources[1]._points.add([0, p, 0, 0, autofocus_yx[0], autofocus_yx[1]])
                # 2 repeated points at center (DAQ needs >= 2 samples per channel)
                sources[0]._points.add([0, p, 0, 0, img_center_yx[0], img_center_yx[1]])
                if point_transformer.multiplier <= 1:
                    sources[0]._points.add([0, p, 0, 0, img_center_yx[0], img_center_yx[1]])
                p += 1

        return sources, np.arange(len(new_positions)), new_seq

    # ------------------------- center_cell = False (original) --------------
    if no_autofocus:
        cell_slice = slice(0, None)
        repeats = [len(point) for point in points]
    else:
        cell_slice = slice(1, None)
        repeats = [len(point) - 1 for point in points]

    repeated_positions = [pos for pos, n in zip(seq.stage_positions, repeats) for _ in range(n)]
    new_seq = seq.replace(stage_positions=repeated_positions)

    if no_autofocus:
        sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        sources = create_point_sources(viewer, point_transformer, size=15)

    if batch:
        core.run_mda(new_seq)
        expanded_indices = [i for i, n in enumerate(repeats) for _ in range(n)]
        # all cell points across every FOV, in expanded order
        all_cells = np.vstack([arr[cell_slice, :] for arr in points])
        for p in range(len(repeated_positions)):
            if not no_autofocus:
                sources[1]._points.add(
                    [0, p, 0, 0,
                     points[expanded_indices[p]][0, 0],
                     points[expanded_indices[p]][0, 1]]
                )
            pt = all_cells[p]
            sources[0]._points.add([0, p, 0, 0, pt[0], pt[1]])
        return sources, np.cumsum([0] + repeats[:-1]), new_seq
    else:
        core.run_mda(seq)
        for p in range(len(seq.stage_positions)):
            if not no_autofocus:
                sources[1]._points.add([0, p, 0, 0, points[p][0, 0], points[p][0, 1]])
            for pt in points[p][cell_slice, :]:
                sources[0]._points.add([0, p, 0, 0, pt[0], pt[1]])
        return sources, np.arange(len(seq.stage_positions)), seq


def manual_point_selections(core, viewer, main_window, point_transformer, N,
                            autofocus_object='glass', batch=True):
    """
    Set up point-source layers for MANUAL selection (click points by hand).

    Mirrors `automated_point_selections` but does NO imaging, segmentation or
    point placement -- it only creates the (empty) source layers and returns
    the same (sources, autofocus_p, new_seq) contract the MDA expects. The user
    clicks points into the layers afterwards.

    N cells-per-FOV is fixed up front (exactly like automated selection):
    - batch=True : each stage position is repeated N times to build new_seq,
      and the user must click N cell points per FOV to match.
    - batch=False: new_seq is the current sequence unchanged; the user clicks
      however many points they like per FOV.

    Layers:
    - autofocus_object None/"None" -> one layer  ('cells').
    - real focus target           -> two layers ('cells','autofocus').

    Returns
    -------
    sources : list[PointsLayerSource]
    autofocus_p : np.ndarray
        batch=True  -> first flat index of each FOV's repeated block.
        batch=False -> every stage-position index.
    new_seq : MDASequence
        batch=True  -> positions repeated N times.
        batch=False -> the current napari sequence, unchanged.
    """
    no_autofocus = _is_no_autofocus(autofocus_object)

    seq = _get_seq_from_napari(main_window)

    # Create the empty source layers (one or two).
    if no_autofocus:
        sources = create_point_sources(
            viewer, point_transformer, size=15,
            names=['cells'], colors=['#aa0000ff'],
        )
    else:
        sources = create_point_sources(viewer, point_transformer, size=15)

    if batch:
        # Repeat every position N times, exactly like the automated batch path.
        repeats = [N for _ in range(len(seq.stage_positions))]
        repeated_positions = [
            pos for pos in seq.stage_positions for _ in range(N)
        ]
        new_seq = seq.replace(stage_positions=repeated_positions)

        # Establish the broadcast/position dimensions on the empty layers so
        # per-position clicks map correctly (automated does this too).
        core.run_mda(new_seq)

        return sources, np.cumsum([0] + repeats[:-1]), new_seq
    else:
        core.run_mda(seq)
        return sources, np.arange(len(seq.stage_positions)), seq
    
def center_manual_selections(core, viewer, main_window, point_transformer,
                             sources, vandermonde_model_path,
                             autofocus_object='glass', center=None,
                             objective=None):
    """
    Convert hand-clicked cell points into centered stage positions.

    Workflow: run manual_point_selections (batch=False), click your cell
    points (and one autofocus point per FOV if autofocus is enabled), then
    call this. Each clicked cell becomes its OWN new stage position, shifted
    with the Vandermonde model so the cell lands at the image center --
    exactly like automated center_cell mode, but with hand-picked cells.

    NOTE: only valid after a NON-batch manual selection (batch repeats
    positions, which breaks the p -> stage-position mapping used here).

    ``objective`` is the Micro-Manager preset used to select a calibration
    from a multi-objective Vandermonde JSON file. Legacy files ignore it.

    Returns
    -------
    sources, autofocus_p, new_seq : same 3-tuple contract as the other
        selection functions. autofocus_p = np.arange(n_new).
    """
    no_autofocus = _is_no_autofocus(autofocus_object)
    C, degree = load_vandermonde_model(
        vandermonde_model_path, objective=objective
    )
    seq = _get_seq_from_napari(main_window)
    img_y = int(core.getImageHeight())
    img_x = int(core.getImageWidth())
    img_center_yx = (
        np.array([img_y / 2.0, img_x / 2.0]) if center is None
        else np.array(center)
    )

    cell_sources = [
        source
        for source in sources
        if getattr(source, "role", None) == "cell"
        or (
            getattr(source, "role", None) is None
            and "cell" in source.name.lower()
        )
    ]
    autofocus_sources = [
        source
        for source in sources
        if getattr(source, "role", None) == "autofocus"
        or (
            getattr(source, "role", None) is None
            and "autofocus" in source.name.lower()
        )
    ]
    if not cell_sources:
        raise ValueError("No cell selection layers found.")

    clicked_cells = []
    for source_index, source in enumerate(cell_sources):
        cell_data = np.asarray(source._points.data)
        if cell_data.size == 0:
            continue
        cell_data = np.atleast_2d(cell_data)
        for row in cell_data:
            clicked_cells.append(
                (int(row[1]), source_index, np.asarray(row[-2:], dtype=float))
            )
    if not clicked_cells:
        raise ValueError("No clicked cell points found -- click points first.")

    af_data = None
    if not no_autofocus:
        if not autofocus_sources:
            raise ValueError(
                "Autofocus is enabled but no autofocus points were clicked."
            )
        af_data = np.asarray(autofocus_sources[0]._points.data)
        if af_data.size == 0:
            raise ValueError(
                "Autofocus is enabled but no autofocus points were clicked."
            )
        af_data = np.atleast_2d(af_data)

    new_positions = []
    af_points = []   # raw autofocus pixel point carried to each new position
    new_position_source_indices = []
    for p in sorted({record[0] for record in clicked_cells}):
        orig_pos = seq.stage_positions[p]
        if not no_autofocus:
            af_here = af_data[af_data[:, 1].astype(int) == p]
            if len(af_here) == 0:
                raise ValueError(f"No autofocus point clicked at position {p}.")
            af_yx = af_here[0, -2:]
        for _, source_index, cell_yx in (
            record for record in clicked_cells if record[0] == p
        ):
            offset_yx = cell_yx - img_center_yx
            offset_xy = np.array([offset_yx[1], offset_yx[0]])
            stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)
            print(f"[manual center] p={p}, cell_yx={cell_yx}, "
                  f"stage_dx={stage_dx:.3f}, stage_dy={stage_dy:.3f}")
            new_positions.append(orig_pos.replace(
                x=float(orig_pos.x - stage_dx),
                y=float(orig_pos.y - stage_dy),
            ))
            new_position_source_indices.append(source_index)
            if not no_autofocus:
                af_points.append(af_yx)

    new_seq = seq.replace(stage_positions=new_positions)
    palette = (
        "#aa0000ff",
        "#0066ccff",
        "#cc6600ff",
        "#8a2be2ff",
        "#008866ff",
        "#cc2288ff",
    )
    names = [source.name for source in cell_sources]
    colors = [
        getattr(source, "color", None) or palette[index % len(palette)]
        for index, source in enumerate(cell_sources)
    ]
    roles = ["cell"] * len(cell_sources)
    display_names = [
        getattr(source, "display_name", source.name)
        for source in cell_sources
    ]
    source_ids = [
        getattr(source, "source_id", None)
        for source in cell_sources
    ]
    if not no_autofocus:
        autofocus_source = autofocus_sources[0]
        names.append(autofocus_source.name)
        colors.append(getattr(autofocus_source, "color", None) or "springgreen")
        roles.append("autofocus")
        display_names.append(
            getattr(autofocus_source, "display_name", autofocus_source.name)
        )
        source_ids.append(getattr(autofocus_source, "source_id", None))
    new_sources = create_point_sources(
        viewer,
        point_transformer,
        size=15,
        names=names,
        colors=colors,
        roles=roles,
        display_names=display_names,
        source_ids=source_ids,
    )

    core.run_mda(new_seq)
    for p in range(len(new_positions)):
        if not no_autofocus:
            new_sources[len(cell_sources)]._points.add(
                [0, p, 0, 0, af_points[p][0], af_points[p][1]]
            )
        cell_source = new_sources[new_position_source_indices[p]]
        cell_source._points.add(
            [0, p, 0, 0, img_center_yx[0], img_center_yx[1]]
        )
        # 2 repeated points at center when the transformer doesn't multiply
        # (DAQ needs >= 2 samples per channel)
        if point_transformer.multiplier <= 1:
            cell_source._points.add(
                [0, p, 0, 0, img_center_yx[0], img_center_yx[1]]
            )
    return new_sources, np.arange(len(new_positions)), new_seq

def grid_point_selections(core, viewer, main_window, point_transformer,
                          fov_x, fov_y,
                          x_range, y_range, x_step, y_step,
                          repeats=2,
                          autofocus_object='None',
                          corner_positions=None,
                          x_count=None, y_count=None,
                          tilt_reference_points=None,
                          tilt_degree=1,
                          snake_axis=None):
    """
    Build a grid of stage positions, each carrying `repeats` copies of the
    SAME single fixed point (fov_x, fov_y). By default the grid is centered on
    the current stage XY. Pass two corner positions to define explicit bounds.

    repeats : int >= 2
        How many identical points to place at (fov_x, fov_y) per position. Must
        be >= 2 (the DAQ needs at least 2 samples per channel).

    corner_positions : ((x1, y1), (x2, y2)) or None
        Opposite stage-space corners of the grid. Coordinate order is
        normalized, so either physical corner may contain the larger X or Y.
        When None, the current stage XY and the X/Y half-ranges are used.
    x_count, y_count : int or None
        Exact numbers of grid positions along X and Y, including endpoints.
        When omitted, ``x_step`` and ``y_step`` are treated as maximum spacing.
    tilt_reference_points : sequence of (x, y, z) or None
        Focused stage-space references used to fit a least-squares Z surface.
        When provided, the fitted Z replaces the template Z at every grid
        position. References must be sufficiently distributed to constrain the
        selected degree.
    tilt_degree : int
        Total degree of the centered/scaled 2D Vandermonde Z surface (1--3).
        The required reference count is (degree + 1)(degree + 2)/2.
    snake_axis : {'x', 'y'} or None
        Fast axis for snake traversal. X reverses direction on alternating Y
        rows; Y reverses direction on alternating X columns. None preserves
        the original raster order. The resulting position order is shared by
        every acquisition channel in the MDA sequence.

    Grid generation does not run a preview or write position layers into the
    napari MDA editor. Fixed point sources are generated procedurally for each
    MDA event, and the returned sequence keeps its original acquisition
    channels.
    autofocus_object : str or None
        If a real focus target ('glass', 'quartz', 'laser', 'software',
        'cell'), an 'autofocus' layer is also created with one point per
        position at the same fixed (fov_y, fov_x). If None/"None", only the
        'cells' layer is created (original behavior).
    """
    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("repeats must be an integer >= 2")

    no_autofocus = _is_no_autofocus(autofocus_object)
    seq = _get_seq_from_napari(main_window)

    if corner_positions is None:
        origin_x, origin_y = core.getXYPosition()
        x_min, x_max = origin_x - x_range, origin_x + x_range
        y_min, y_max = origin_y - y_range, origin_y + y_range
    else:
        corners = np.asarray(corner_positions, dtype=float)
        if corners.shape != (2, 2) or not np.all(np.isfinite(corners)):
            raise ValueError(
                "corner_positions must contain two finite (x, y) positions"
            )
        x_min, x_max = np.sort(corners[:, 0])
        y_min, y_max = np.sort(corners[:, 1])

    if (x_count is None) != (y_count is None):
        raise ValueError("x_count and y_count must be provided together")
    xs = _grid_axis(x_min, x_max, x_step, x_count)
    ys = _grid_axis(y_min, y_max, y_step, y_count)
    tilt_model = (
        None if tilt_reference_points is None
        else _fit_tilt_surface(tilt_reference_points, tilt_degree)
    )
    if tilt_model is not None:
        origin, _scale, _coefficients, rmse = tilt_model
        print(
            f"[grid setup] degree {tilt_degree} tilt surface "
            f"origin=({origin[0]:.4f}, {origin[1]:.4f}), "
            f"RMSE={rmse:.6f} um"
        )
    template = seq.stage_positions[0]
    grid_positions = []
    for gx, gy in _ordered_grid_coordinates(xs, ys, snake_axis):
        position_values = {"x": float(gx), "y": float(gy)}
        if tilt_model is not None:
            position_values["z"] = float(
                _evaluate_tilt_surface(
                    np.array([gx, gy]), tilt_model, tilt_degree
                )[0]
            )
        grid_positions.append(template.replace(**position_values))
    new_seq = seq.replace(stage_positions=grid_positions)

    sources = _create_fixed_grid_sources(
        core,
        point_transformer,
        fov_x,
        fov_y,
        repeats,
        include_autofocus=not no_autofocus,
    )
    _remove_grid_placeholder(viewer)
    print(
        "[grid setup] grid ready "
        f"({len(new_seq.stage_positions)} positions, no preview or napari layers)"
    )

    # A range keeps membership checks O(1) without allocating another
    # position-sized numpy array.
    return sources, range(len(new_seq.stage_positions)), new_seq
def unload(core, N=20):
    n = 0.1  # starting sleep time
    for attempt in range(N):
        if attempt == N-1:
            print('reach reloading maxiter')
        try:
            time.sleep(n)
            try:
                core.events.channelGroupChanged.disconnect()
            except Exception:
                None
            try:
                core.events.configGroupChanged.disconnect()
            except Exception:
                None
            try:
                core.events.propertyChanged.disconnect()
            except Exception:
                None
            try:
                core.events.systemConfigurationLoaded.disconnect()
            except Exception:
                None
            try:
                core.events.configSet.disconnect()
            except Exception:
                None

            core.unloadAllDevices()
            core.waitForSystem()
            return  # success!
        except Exception as e:
            n += 1  # increase wait time and retry
