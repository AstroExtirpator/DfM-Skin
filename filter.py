"""
Morphology-based minimum-feature-size filter for AFP manufacturing.

Enforces a minimum cut length (MCL) on the discrete zone field so all retained
features are manufacturable. Ordinal zone fields (0 = thinnest) are handled by
threshold decomposition: the k-zone problem splits into k-1 binary thresholds,
each filtered morphologically, then reassembled.

Methods: "close_only" (fill thin islands, material only grows), "open_only"
(remove thin protrusions, material only shrinks), "close_open" and "open_close"
(both, roughly area-conservative, differing in which acts first).

Refs: Sigmund, Struct Multidisc Optim 33 (2007) 401-424; Pratt, Digital Image
Processing, Wiley 1991 (threshold decomposition).
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    zoom,
    gaussian_filter,
    label,
)
from skimage.morphology import disk

_VERBOSE = True  # module-level flag for plot "Saved:" messages; set via set_verbose()


def set_verbose(v: bool):
    global _VERBOSE
    _VERBOSE = v


def build_grid_mapping(
    X: np.ndarray,
    Y: np.ndarray,
    eids: list[int],
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Build a 2D grid mapping from scattered element data.

    Assumes a regular rectangular grid (uniform element spacing).

    Returns:
        eid_grid:      2D array (ny, nx) of element IDs at each grid position
        elem_to_grid:  dict {eid: (iy, ix)}
        unique_x:      sorted unique X coordinates
        unique_y:      sorted unique Y coordinates
    """
    unique_x = np.sort(np.unique(np.round(X, 2)))
    unique_y = np.sort(np.unique(np.round(Y, 2)))
    nx, ny = len(unique_x), len(unique_y)

    x_to_ix = {x: i for i, x in enumerate(unique_x)}
    y_to_iy = {y: i for i, y in enumerate(unique_y)}

    eid_grid = np.full((ny, nx), -1, dtype=int)
    elem_to_grid = {}

    for idx, eid in enumerate(eids):
        ix = x_to_ix[round(X[idx], 2)]
        iy = y_to_iy[round(Y[idx], 2)]
        eid_grid[iy, ix] = eid
        elem_to_grid[eid] = (iy, ix)

    return eid_grid, elem_to_grid, unique_x, unique_y


def zones_to_grid(
    cluster_data: dict,
    cluster_centers: list[float],
    elem_to_grid: dict,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert cluster assignments to a 2D ordinal zone grid.

    Zones are numbered 0 (thinnest) to k-1 (thickest).
    """
    sorted_ci = sorted(range(len(cluster_centers)), key=lambda i: cluster_centers[i])
    ci_to_rank = {ci: rank for rank, ci in enumerate(sorted_ci)}

    zone_grid = np.full(grid_shape, -1, dtype=int)
    for ci, elements in cluster_data.items():
        rank = ci_to_rank[ci]
        for eid, _ in elements:
            if eid in elem_to_grid:
                iy, ix = elem_to_grid[eid]
                zone_grid[iy, ix] = rank

    return zone_grid


def make_structuring_element(radius_mm: float, element_size_mm: float,
                             angles_deg: list[float] | None = None,
                             tape_width_mm: float | None = None) -> np.ndarray:
    """Structuring element for morphological ops. angles_deg=None gives a disk
    (enforces MCL in all directions). With ply angles it's a directional SE:
    the union of tape-width arms along each angle, each clipped to the disk
    radius, so it's always a subset of the disk (never adds more than isotropic).
    radius_mm = MCL/2; tape_width_mm defaults to one pixel."""
    r_elements = int(round(radius_mm / element_size_mm))
    r_elements = max(r_elements, 1)

    if angles_deg is None:
        return disk(r_elements)

    unique_angles = sorted(set(round(a) % 180 for a in angles_deg))

    if tape_width_mm is None:
        tape_width_mm = element_size_mm
    half_w = max(int(round(tape_width_mm / element_size_mm / 2)), 0)

    size = 2 * r_elements + 1
    se = np.zeros((size, size), dtype=bool)
    centre = size // 2

    for angle in unique_angles:
        theta = np.radians(angle)
        dx = np.cos(theta)
        dy = np.sin(theta)
        for step in range(-r_elements, r_elements + 1):
            cx = centre + step * dx
            cy = centre - step * dy
            for off in range(-half_w, half_w + 1):
                ix = int(round(cx + off * (-dy)))
                iy = int(round(cy + off * dx))
                if 0 <= ix < size and 0 <= iy < size:
                    d = np.sqrt((ix - centre) ** 2 + (iy - centre) ** 2)
                    if d <= r_elements:
                        se[iy, ix] = True

    if not se.any():
        se[centre, centre] = True

    return se


def clamp_per_angle_zone_grids(
    per_angle_zone_grids: dict[float, np.ndarray],
    zone_grid_combined: np.ndarray,
) -> dict[float, np.ndarray]:
    """Clamp per-angle zone grids so they never exceed the final combined grid."""
    clamped = {}
    valid = zone_grid_combined >= 0
    for angle, ag in per_angle_zone_grids.items():
        out = np.minimum(ag, zone_grid_combined)
        out[~valid] = -1
        clamped[angle] = out
    return clamped


def combine_per_angle_zone_grids(
    per_angle_zone_grids: dict[float, np.ndarray],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine per-angle zone grids into one zone grid by element-wise max."""
    if not per_angle_zone_grids:
        raise ValueError("per_angle_zone_grids must not be empty")

    combined = None
    for ag in per_angle_zone_grids.values():
        if combined is None:
            combined = ag.copy()
        else:
            combined = np.maximum(combined, ag)

    if valid_mask is None:
        valid_mask = combined >= 0
    combined = combined.copy()
    combined[~valid_mask] = -1
    return combined


def _masked_close(mask: np.ndarray, se: np.ndarray,
                  valid: np.ndarray) -> np.ndarray:
    """Morphological closing (dilate then erode) using *se* for both steps."""
    r = se.shape[0] // 2
    padded = np.pad(mask, r, mode='edge')
    dilated = binary_dilation(padded, structure=se)
    eroded = binary_erosion(dilated, structure=se)
    result = eroded[r:-r, r:-r]
    result[~valid] = False
    return result


def _masked_open(mask: np.ndarray, se: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """Morphological opening (erode then dilate) using *se* for both steps."""
    r = se.shape[0] // 2
    padded = np.pad(mask, r, mode='edge')
    eroded = binary_erosion(padded, structure=se)
    dilated = binary_dilation(eroded, structure=se)
    result = dilated[r:-r, r:-r]
    result[~valid] = False
    return result


def _masked_close_open(mask: np.ndarray, se: np.ndarray,
                       valid: np.ndarray) -> np.ndarray:
    """Close-then-open."""
    return _masked_open(_masked_close(mask, se, valid), se, valid)


def _masked_open_close(mask: np.ndarray, se: np.ndarray,
                       valid: np.ndarray) -> np.ndarray:
    """Open-then-close."""
    return _masked_close(_masked_open(mask, se, valid), se, valid)


def _masked_erode(mask: np.ndarray, se: np.ndarray,
                  valid: np.ndarray) -> np.ndarray:
    """Erode with edge padding."""
    r = se.shape[0] // 2
    padded = np.pad(mask, r, mode='edge')
    result = binary_erosion(padded, structure=se)
    result = result[r:-r, r:-r]
    result[~valid] = False
    return result


def _masked_dilate(mask: np.ndarray, se: np.ndarray,
                   valid: np.ndarray) -> np.ndarray:
    """Dilate while preserving the panel boundary via edge-padding."""
    r = se.shape[0] // 2
    padded = np.pad(mask, r, mode='edge')
    result = binary_dilation(padded, structure=se)
    result = result[r:-r, r:-r]
    result[~valid] = False
    return result


def keep_largest_connected_component_binary(zone_grid: np.ndarray) -> np.ndarray:
    """Keep only the largest connected solid component in a binary zone grid.

    Intended for topology-verification style 0/1 fields where small detached
    islands are numerical artefacts of a branch-preserving regrow step.
    """
    solid = zone_grid >= 1
    labels, n_lab = label(solid)
    if n_lab <= 1:
        return zone_grid.copy()

    sizes = np.bincount(labels.ravel())[1:]
    keep = 1 + int(np.argmax(sizes))

    out = np.zeros_like(zone_grid)
    out[labels == keep] = 1
    out[zone_grid < 0] = -1
    return out


def _get_equidistant_offsets(
    ux: np.ndarray,
    uy: np.ndarray,
    angle_deg: float,
    tape_width: float,
    element_size: float,
):
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = -st, ct
    nx, ny = len(ux), len(uy)
    corners_x = np.array([ux[0], ux[-1], ux[0], ux[-1]])
    corners_y = np.array([uy[0], uy[0], uy[-1], uy[-1]])
    proj = corners_x * cp + corners_y * sp
    pmin, pmax = proj.min(), proj.max()
    off_mm = np.arange(pmin + tape_width / 2.0, pmax, tape_width)
    if len(off_mm) == 0:
        return np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0
    cx = ux[nx // 2] if nx > 1 else ux[0]
    cy = uy[ny // 2] if ny > 1 else uy[0]
    cp0 = cx * cp + cy * sp
    off_px = (off_mm - cp0) / element_size
    return off_px, off_mm, cx, cy, cp, sp


def _find_short_fiber_segments(mask, ux, uy, angle_deg, mcl, es, tw, skip_edge=False):
    import numpy as np
    ny, nx = mask.shape
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = -st, ct
    off_px, *_ = _get_equidistant_offsets(ux, uy, angle_deg, tw, es)
    if len(off_px) == 0:
        return np.zeros_like(mask, dtype=bool)
    cx_phys = ux[len(ux) // 2]
    cy_phys = uy[len(uy) // 2]
    tp = (cx_phys * ct + cy_phys * st) / es
    cx = nx / 2.0
    cy = ny / 2.0
    mr = np.sqrt(nx ** 2 + ny ** 2)
    t_px = np.linspace(-mr, mr, int(4 * mr) + 1)
    rem = np.zeros_like(mask, dtype=bool)
    for op in off_px:
        ox = cx + op * cp
        oy = cy + op * sp
        px = ox + t_px * ct
        py = oy + t_px * st
        iy = np.round(py).astype(int)
        ix = np.round(px).astype(int)
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        iyc = iy.clip(0, ny - 1)
        ixc = ix.clip(0, nx - 1)
        vp = inb & mask[iyc, ixc]
        if not vp.any():
            continue
        v = vp.astype(int)
        d = np.diff(v)
        sts = np.where(d == 1)[0] + 1
        eds = np.where(d == -1)[0] + 1
        if v[0]:
            sts = np.concatenate([[0], sts])
        if v[-1]:
            eds = np.concatenate([eds, [len(v)]])
        for s, e in zip(sts, eds):
            if skip_edge:
                touches_edge = (s == 0 or not inb[s - 1] or
                                e == len(v) or not inb[e] if e < len(v) else True)
                if touches_edge:
                    continue
            seg_len = es * abs(t_px[e - 1] - t_px[s]) + es
            if seg_len < mcl:
                for idx in range(s, e):
                    if inb[idx]:
                        rem[iy[idx], ix[idx]] = True
    return rem


def _count_short_segments(mask, ux, uy, angle_deg, mcl, es, tw):
    """Count the number of distinct short (< mcl) fibre segments."""
    ny, nx = mask.shape
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = -st, ct

    offsets_px, *_ = _get_equidistant_offsets(ux, uy, angle_deg, tw, es)
    if len(offsets_px) == 0:
        return 0

    cx_phys = ux[len(ux) // 2]
    cy_phys = uy[len(uy) // 2]
    tp = (cx_phys * ct + cy_phys * st) / es
    cx = nx / 2.0; cy = ny / 2.0
    mr = np.sqrt(nx ** 2 + ny ** 2)
    t_px = np.linspace(-mr, mr, int(4 * mr) + 1)

    count = 0
    for op in offsets_px:
        ox = cx + op * cp; oy = cy + op * sp
        px = ox + t_px * ct; py = oy + t_px * st
        iy = np.round(py).astype(int); ix = np.round(px).astype(int)
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        iyc = iy.clip(0, ny - 1); ixc = ix.clip(0, nx - 1)
        vp = inb & mask[iyc, ixc]
        if not vp.any():
            continue
        v = vp.astype(int)
        d = np.diff(v)
        sts = np.where(d == 1)[0] + 1; eds = np.where(d == -1)[0] + 1
        if v[0]: sts = np.concatenate([[0], sts])
        if v[-1]: eds = np.concatenate([eds, [len(v)]])
        for s, e in zip(sts, eds):
            if e - s < 3:
                continue
            seg_len = es * abs(t_px[e - 1] - t_px[s]) + es
            if seg_len < mcl:
                count += 1
    return count


def enforce_fiber_length(
    zone_grid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    per_zone_angles: dict[int, list[float]],
    mcl: float,
    element_size: float,
    tape_width: float,
    verbose: bool = False,
) -> np.ndarray:
    """Enforce minimum AFP fibre segment length on the zone grid.

    Each zone threshold is processed top-down.  Within a threshold every
    surviving angle checks the same original mask, so angle order does
    not matter — all short‑segment removals are accumulated independently.
    """
    from scipy.ndimage import binary_dilation

    valid = zone_grid >= 0
    nz = int(zone_grid.max()) + 1

    thresholds = {t: (zone_grid >= t) & valid for t in range(1, nz)}
    total = 0

    for t in range(nz - 1, 0, -1):
        mask = thresholds[t]
        if not mask.any():
            continue
        if t not in per_zone_angles or not per_zone_angles[t]:
            continue

        remove = np.zeros_like(mask, dtype=bool)
        for angle in per_zone_angles[t]:
            seg_remove = _find_short_fiber_segments(
                mask, ux, uy, angle, mcl, element_size, tape_width,
            )
            if seg_remove.any():
                perp_deg = angle + 90.0
                tp = np.deg2rad(perp_deg)
                pc, ps = np.cos(tp), np.sin(tp)
                hw = max(1, int(round(tape_width / element_size / 2)))
                k = 2 * hw + 1
                perp_se = np.zeros((k, k), dtype=bool)
                ch = k // 2
                for s in range(-hw, hw + 1):
                    ix = int(round(ch + s * pc))
                    iy = int(round(ch - s * ps))
                    if 0 <= ix < k and 0 <= iy < k:
                        perp_se[iy, ix] = True
                perp_se[ch, ch] = True
                seg_remove = binary_dilation(seg_remove, structure=perp_se)
            remove |= seg_remove

        n = int(remove.sum())
        if n:
            total += n
            thresholds[t] = mask & ~remove

    result = np.zeros_like(zone_grid, dtype=int)
    for t in range(1, nz):
        result += thresholds[t].astype(int)
    result[~valid] = -1

    # count remaining short fibre segments per threshold
    per_thr = {}
    for t in range(1, nz):
        if t not in per_zone_angles or not per_zone_angles[t]:
            continue
        mask = (result >= t) & valid
        n = 0
        for angle in per_zone_angles[t]:
            n += _count_short_segments(
                mask, ux, uy, angle, mcl, element_size, tape_width,
            )
        per_thr[t] = n

    if total and verbose:
        print(f"  Fibre length enforcement: {total} zone-pixels demoted")
    if any(per_thr.values()) and verbose:
        summary = ", ".join(f"t{t}={per_thr[t]}" for t in sorted(per_thr))
        print(f"  Remaining short (< {mcl:.0f} mm) segments: {sum(per_thr.values())}  ({summary})")

    return result


def trim_remaining_short_fibers_per_angle(
    per_angle_zone_grids: dict[float, np.ndarray],
    ux: np.ndarray,
    uy: np.ndarray,
    mcl: float,
    element_size: float,
    tape_width: float,
    max_passes: int = 4,
    verbose: bool = False,
) -> dict[float, np.ndarray]:
    """Trim any remaining short fibres independently on each per-angle grid.

    This runs after directional filtering and after the stiffener-direction ply
    reassignment has already been applied. Only the affected angle/threshold
    masks are trimmed; no coupled zone demotion is performed.
    """
    from scipy.ndimage import binary_dilation

    if not per_angle_zone_grids or mcl <= 0:
        return per_angle_zone_grids

    def _make_perp_band(angle: float) -> np.ndarray:
        perp_deg = angle + 90.0
        tp = np.deg2rad(perp_deg)
        pc, ps = np.cos(tp), np.sin(tp)
        hw = max(1, int(round(tape_width / element_size / 2)))
        k = 2 * hw + 1
        perp_se = np.zeros((k, k), dtype=bool)
        ch = k // 2
        for s in range(-hw, hw + 1):
            ix = int(round(ch + s * pc))
            iy = int(round(ch - s * ps))
            if 0 <= ix < k and 0 <= iy < k:
                perp_se[iy, ix] = True
        perp_se[ch, ch] = True
        return perp_se

    current = {a: ag.copy() for a, ag in per_angle_zone_grids.items()}
    total_removed = 0

    for pass_idx in range(max_passes):
        pass_removed = 0
        next_grids = {}
        for angle, ag in current.items():
            valid = ag >= 0
            if not valid.any():
                next_grids[angle] = ag.copy()
                continue

            nz = int(np.max(ag[valid])) + 1
            thresholds = {t: ((ag >= t) & valid) for t in range(1, nz)}
            perp_se = _make_perp_band(float(angle))
            removed_here = 0

            for t in range(nz - 1, 0, -1):
                mask = thresholds[t]
                if not mask.any():
                    continue
                seg_remove = _find_short_fiber_segments(
                    mask, ux, uy, float(angle), mcl, element_size, tape_width,
                    skip_edge=True,
                )
                if not seg_remove.any():
                    continue
                seg_remove = binary_dilation(seg_remove, structure=perp_se)
                n_removed = int((mask & seg_remove).sum())
                thresholds[t] = mask & ~seg_remove
                removed_here += n_removed

            out = np.zeros_like(ag, dtype=int)
            for t in range(1, nz):
                out += thresholds[t].astype(int)
            out[~valid] = -1
            next_grids[angle] = out
            pass_removed += removed_here

        current = next_grids
        total_removed += pass_removed
        if verbose and pass_removed > 0:
            print(f"  Final per-angle trim: pass {pass_idx + 1} removed {pass_removed} px")
        if pass_removed == 0:
            break

    if verbose and total_removed > 0:
        remaining = 0
        for angle, ag in current.items():
            valid = ag >= 0
            nz = int(np.max(ag[valid])) + 1 if valid.any() else 0
            for t in range(1, nz):
                remaining += _count_short_segments(
                    (ag >= t) & valid, ux, uy, float(angle), mcl,
                    element_size, tape_width,
                )
        print(f"  Final per-angle trim total: {total_removed} px removed")
        if remaining > 0:
            print(f"  Remaining short (< {mcl:.0f} mm) angle-segments after final trim: {remaining}")

    return current


def final_manufacturability_trim(
    per_angle_zone_grids: dict[float, np.ndarray],
    ux: np.ndarray,
    uy: np.ndarray,
    mcl: float,
    element_size: float,
    tape_width: float,
    verbose: bool = False,
) -> dict[float, np.ndarray]:
    """One-shot removal of EXACTLY the fibres the coverage plots draw red.

    Measures fibre segments with EXACTLY the geometry used by
    ``plot_zone_fiber_coverage`` / ``_plot_angle_fibers`` — same equidistant
    AFP scan lines, same segment-length formula (no +1-element padding),
    the same edge exemption (segments touching the panel boundary are
    manufacturable because tows overshoot the edge), and the same >= 3 px
    draw threshold.  Each red segment is deleted from the ply shape of THAT
    angle and threshold only — the per-angle shapes are decoupled, so other
    angles' shapes at the same location are left untouched.

    Single pass, green-protected: the red cells are dilated by a
    perpendicular tape band so the interstitial cells between scan lines
    (which the plot never traces) are cleared together with the red fibre,
    but every cell carrying a GREEN (kept) segment of the same angle — at
    this or any higher threshold — is explicitly protected from removal.
    Earlier versions dilated without this protection and iterated to
    convergence, which clipped long manufacturable fibres running along
    stiffeners (collateral damage the plots never marked red); removing a
    whole red segment while protecting green cells cannot shorten any green
    fibre, so no iteration is needed.
    """
    from scipy.ndimage import binary_dilation

    if not per_angle_zone_grids or mcl <= 0:
        return per_angle_zone_grids

    def _perp_band(angle: float) -> np.ndarray:
        perp_deg = angle + 90.0
        tp = np.deg2rad(perp_deg)
        pc, ps = np.cos(tp), np.sin(tp)
        hw = max(1, int(round(tape_width / element_size / 2)))
        k = 2 * hw + 1
        perp_se = np.zeros((k, k), dtype=bool)
        ch = k // 2
        for s in range(-hw, hw + 1):
            jx = int(round(ch + s * pc))
            jy = int(round(ch - s * ps))
            if 0 <= jx < k and 0 <= jy < k:
                perp_se[jy, jx] = True
        perp_se[ch, ch] = True
        return perp_se

    def _red_cells(mask: np.ndarray, angle: float):
        """(removal_mask, keep_mask, n_red) with the plot's exact metric.

        *removal_mask* holds the cells of the plot-red segments; *keep_mask*
        holds the cells of every OTHER traced segment (green: long or
        edge-touching, plus sub-3-sample fragments the plot never draws).
        A cell can be visited by two scan lines (rounding zigzag at the
        edges of diagonal shapes, e.g. ±45° stiffener strips), so a red
        fragment may share cells with a long green fibre — those shared
        cells must NOT be removed or the green fibre gets holes punched
        into it.
        """
        ny, nx = mask.shape
        theta = np.deg2rad(angle)
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = -st, ct
        off_px, *_ = _get_equidistant_offsets(
            ux, uy, angle, tape_width, element_size)
        rem = np.zeros_like(mask, dtype=bool)
        keep = np.zeros_like(mask, dtype=bool)
        if len(off_px) == 0:
            return rem, keep, 0
        cx, cy = nx / 2.0, ny / 2.0
        mr = np.sqrt(nx ** 2 + ny ** 2)
        t_px = np.linspace(-mr, mr, int(4 * mr) + 1)
        n_red = 0
        for op in off_px:
            px = cx + op * cp + t_px * ct
            py = cy + op * sp + t_px * st
            iy = np.round(py).astype(int)
            ix = np.round(px).astype(int)
            inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            iyc = iy.clip(0, ny - 1)
            ixc = ix.clip(0, nx - 1)
            vp = inb & mask[iyc, ixc]
            if not vp.any():
                continue
            v = vp.astype(int)
            d = np.diff(v)
            sts = np.where(d == 1)[0] + 1
            eds = np.where(d == -1)[0] + 1
            if v[0]:
                sts = np.concatenate([[0], sts])
            if v[-1]:
                eds = np.concatenate([eds, [len(v)]])
            for s, e in zip(sts, eds):
                touches_edge = (s == 0 or not inb[s - 1]
                                or e == len(v) or not inb[e])
                seg_len = element_size * abs(t_px[e - 1] - t_px[s])
                is_red = (e - s >= 3) and not touches_edge and seg_len < mcl
                sl = slice(s, e)
                if is_red:
                    n_red += 1
                    rem[iy[sl][inb[sl]], ix[sl][inb[sl]]] = True
                else:
                    keep[iy[sl][inb[sl]], ix[sl][inb[sl]]] = True
        return rem, keep, n_red

    current = {}
    total_removed = 0
    total_red = 0
    for angle, ag in per_angle_zone_grids.items():
        valid = ag >= 0
        if not valid.any():
            current[angle] = ag.copy()
            continue
        nz = int(np.max(ag[valid])) + 1
        thresholds = {t: ((ag >= t) & valid) for t in range(1, nz)}
        # Measure ALL thresholds on the pre-trim masks first.
        rem_t, keep_t = {}, {}
        for t in range(1, nz):
            rem_t[t], keep_t[t], n_red = _red_cells(
                thresholds[t], float(angle))
            total_red += n_red
        # Remove red cells, dilated by a perpendicular tape band so the
        # interstitial cells between scan lines (never traced by the plot)
        # go with them — but protect any cell that carries a green fibre
        # at this OR any higher threshold (mask_t' subset of mask_t for
        # t' > t, so nesting below would otherwise punch holes into it).
        band = _perp_band(float(angle))
        cum_keep = np.zeros_like(valid, dtype=bool)
        for t in range(nz - 1, 0, -1):
            cum_keep |= keep_t[t]
            if not rem_t[t].any():
                continue
            removal = binary_dilation(rem_t[t], structure=band) & ~cum_keep
            if removal.any():
                total_removed += int((thresholds[t] & removal).sum())
                thresholds[t] = thresholds[t] & ~removal
        # Keep thresholds nested (higher level subset of lower level)
        for t in range(2, nz):
            thresholds[t] &= thresholds[t - 1]
        out = np.zeros_like(ag, dtype=int)
        for t in range(1, nz):
            out += thresholds[t].astype(int)
        out[~valid] = -1
        current[angle] = out

    # Verification with the plot metric: report any residual red segments
    # (red fragments that shared every cell with a protected green fibre
    # cannot be fully removed — report them honestly).
    residual = 0
    for angle, ag in current.items():
        valid = ag >= 0
        if not valid.any():
            continue
        nz = int(np.max(ag[valid])) + 1
        for t in range(1, nz):
            _, _, n_red = _red_cells((ag >= t) & valid, float(angle))
            residual += n_red
    if total_removed > 0 or verbose:
        print(f"  Final manufacturability trim: {total_red} plot-red segments "
              f"({total_removed} px) removed, "
              f"{residual} unmanufacturable segments remain"
              + ("" if residual == 0 else
                 "  (residual segments overlap protected green fibres)"))

    return current


def decoupled_directional_filter(
    zone_grid_iso: np.ndarray,
    per_zone_angles: dict[int, list[float]],
    ux: np.ndarray,
    uy: np.ndarray,
    mcl: float,
    element_size: float,
    tape_width: float,
    n_zones: int,
    edge_overshoot: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Decouple ply shapes per angle and remove short fibres. From the
    isotropically-filtered grid, each fibre direction gets its own threshold
    masks and its short segments (< mcl) are removed by line scan; edge-touching
    fibres are kept (tows can overshoot the edge). Returns (combined_grid,
    per_angle_zone_grids), combined = element-wise max (union) over angles."""
    from scipy.ndimage import binary_dilation

    valid = (zone_grid_iso >= 0)

    # Handle edge overshoot padding
    if edge_overshoot > 0:
        opx = max(1, int(round(edge_overshoot / element_size)))
        zone_grid_iso = np.pad(zone_grid_iso, opx, mode='edge')
        valid = (zone_grid_iso >= 0)
        _dx = float(ux[1] - ux[0]) if len(ux) > 1 else element_size
        _dy = float(uy[1] - uy[0]) if len(uy) > 1 else element_size
        _ux = np.concatenate([ux[0] - np.arange(opx, 0, -1) * _dx, ux])
        _uy = np.concatenate([uy[0] - np.arange(opx, 0, -1) * _dy, uy])
    else:
        opx = 0
        _ux, _uy = ux, uy

    # Collect all unique angles across thresholds
    unique_angles = sorted(set(a for angles in per_zone_angles.values() for a in angles))

    # Base threshold masks from isotropic result
    base_masks = {t: (zone_grid_iso >= t) & valid for t in range(1, n_zones)}

    per_angle_accum = {}

    for angle in unique_angles:
        accum = np.zeros_like(zone_grid_iso, dtype=int)

        for t in range(1, n_zones):
            mask = base_masks[t].copy()
            if not mask.any():
                continue
            if t not in per_zone_angles or angle not in per_zone_angles[t]:
                accum += mask.astype(int)
                continue

            seg_remove = _find_short_fiber_segments(
                mask, _ux, _uy, float(angle), mcl, element_size, tape_width,
                skip_edge=True,
            )
            if seg_remove.any():
                # Dilate perpendicular to fiber direction to remove full tape band
                perp_deg = angle + 90.0
                tp = np.deg2rad(perp_deg)
                pc, ps = np.cos(tp), np.sin(tp)
                hw = max(1, int(round(tape_width / element_size / 2)))
                k = 2 * hw + 1
                perp_se = np.zeros((k, k), dtype=bool)
                ch = k // 2
                for s in range(-hw, hw + 1):
                    ix = int(round(ch + s * pc))
                    iy = int(round(ch - s * ps))
                    if 0 <= ix < k and 0 <= iy < k:
                        perp_se[iy, ix] = True
                perp_se[ch, ch] = True
                seg_remove = binary_dilation(seg_remove, structure=perp_se)

                n_removed = int((mask & seg_remove).sum())
                mask = mask & ~seg_remove
                if verbose:
                    print(f"    [{angle:>3.0f}\u00b0] Threshold >= {t}: removed {n_removed} px "
                          f"({int(seg_remove.sum())} short-fibre px)")

            accum += mask.astype(int)

        per_angle_accum[angle] = accum

    # Combine: element-wise maximum across all angles (union / max expansion)
    combined = np.zeros_like(zone_grid_iso, dtype=int)
    for angle in unique_angles:
        combined = np.maximum(combined, per_angle_accum[angle])
    combined[~valid] = -1

    # Crop back if padded
    if opx > 0:
        combined = combined[opx:-opx, opx:-opx]
        per_angle_accum = {
            a: acc[opx:-opx, opx:-opx] for a, acc in per_angle_accum.items()
        }

    if verbose:
        print(f"  Decoupled directional filter: {len(unique_angles)} angles, "
              f"combined via max-expansion (union)")

    return combined, per_angle_accum


def balance_angle_grids(
    per_angle_zone_grids: dict[float, np.ndarray],
    verbose: bool = False,
) -> dict[float, np.ndarray]:
    """Enforce ±angle balance by taking element-wise min of symmetric pairs.

    For each symmetric angle pair (+a, -a), both grids are replaced by their
    element-wise minimum.  This ensures both angles always have the same zone
    level at every element, preventing unbalanced laminates.

    Returns a new dict; the input is not modified.
    """
    effective = dict(per_angle_zone_grids)
    paired = set()
    for ua in per_angle_zone_grids:
        ru = round(ua)
        if ru <= 0 or ua in paired:
            continue
        neg_key = next((k for k in per_angle_zone_grids if round(k) == -ru), None)
        if neg_key is not None:
            combined = np.minimum(per_angle_zone_grids[ua], per_angle_zone_grids[neg_key])
            effective[ua] = combined
            effective[neg_key] = combined
            paired.add(ua)
            paired.add(neg_key)
            if verbose:
                diff_px = int((per_angle_zone_grids[ua] != combined).sum() +
                              (per_angle_zone_grids[neg_key] != combined).sum())
                print(f"    \u00b1{ru:.0f}\u00b0 balanced: {diff_px} px adjusted to min zone")
    return effective


def compute_per_element_plies(
    per_angle_zone_grids: dict[float, np.ndarray],
    eid_grid: np.ndarray,
    sst,
    n_zones: int,
    balance_pairs: bool = True,
    verbose: bool = False,
) -> dict[int, tuple[int, ...]]:
    """Half-plies present per element from the per-angle zones. Each half-ply
    has an introduction zone (where it first appears thin->thick); a ply is
    present in an element if its angle's per-angle zone >= that intro zone. With
    balance_pairs, +/-theta grids are balanced by element-wise minimum so both
    stay equal (avoids coupling). Returns {eid: tuple of present half-ply idx}."""
    sorted_counts = sorted(sst.zone_ply_counts)
    n_max = sst.n_max

    # Determine introduction zone for each half-ply
    def _surviving_indices(zone_idx):
        n = sorted_counts[zone_idx]
        n_drops = (n_max - n) // 2
        return set(
            p.half_idx for p in sst.half_plies
            if p.is_continuous or (p.drop_rank is not None and p.drop_rank > n_drops)
        )

    intro_zone = {}
    prev = set()
    for z in range(len(sorted_counts)):
        curr = _surviving_indices(z)
        for idx in curr - prev:
            intro_zone[idx] = z
        prev = curr

    # Map each angle to rounded value for matching per_angle_zone_grids keys
    angle_map = {}
    for ua in per_angle_zone_grids:
        for p in sst.half_plies:
            if round(p.angle) == round(ua):
                angle_map[p.half_idx] = ua

    effective_grids = (balance_angle_grids(per_angle_zone_grids, verbose=verbose)
                       if balance_pairs else per_angle_zone_grids)

    # For each element, determine present half-plies
    elem_half_plies = {}
    ny, nx = eid_grid.shape
    for iy in range(ny):
        for ix in range(nx):
            eid = eid_grid[iy, ix]
            if eid < 0:
                continue
            present = []
            for p in sst.half_plies:
                ua = angle_map.get(p.half_idx)
                if ua is None:
                    continue
                z_intro = intro_zone.get(p.half_idx, 0)
                ag = effective_grids[ua]
                angle_zone = int(ag[iy, ix])
                if angle_zone >= z_intro:
                    present.append(p.half_idx)
            elem_half_plies[eid] = tuple(present)

    return elem_half_plies



def enforce_local_radius(
    zone_grid: np.ndarray,
    n_zones: int,
    mcl: float,
    element_size: float,
) -> np.ndarray:
    """Ensure every boundary pixel has a disk of radius MCL fully inside
    the zone.  This doubles the corner‑rounding radius so that any straight
    line through any pixel crosses ≥ MCL of zone — eliminating thin
    protrusions narrower than MCL that create short fibre segments.

    Each threshold mask is opened (erode → dilate) with a disk SE of
    radius = MCL.  Edge‑padding preserves the panel boundary.
    """
    from skimage.morphology import disk
    from scipy.ndimage import binary_erosion, binary_dilation

    valid = zone_grid >= 0
    r_px = max(1, int(round(mcl / element_size / 2)))
    se = disk(r_px)

    result = np.zeros_like(zone_grid, dtype=int)
    for t in range(1, n_zones):
        mask = (zone_grid >= t) & valid
        if not mask.any():
            continue
        padded = np.pad(mask, r_px, mode='edge')
        eroded = binary_erosion(padded, structure=se)
        dilated = binary_dilation(eroded, structure=se)
        mask = dilated[r_px:-r_px, r_px:-r_px]
        mask[~valid] = False
        result += mask.astype(np.int32)

    result[~valid] = -1
    return result


def apply_morphology_filter(
    zone_grid: np.ndarray,
    mcl: float,
    element_size: float,
    n_zones: int,
    method: str = "close_only",
    ply_angles: list[float] | None = None,
    tape_width: float | None = None,
    directional_mode: str = "star",
    per_zone_angles: dict[int, list[float]] | None = None,
    per_angle_zone_grids: dict[float, np.ndarray] | None = None,
    run_fiber_enforcement: bool = True,
    mcl_check: float = 0.0,
    post_process_mcl: float = 0.0,
    local_radius_check: bool = False,
    ux: np.ndarray | None = None,
    uy: np.ndarray | None = None,
    edge_overshoot: float = 0.0,
    _out_per_angle: dict | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Filter the ordinal zone field via threshold decomposition: for each
    level t, build the binary mask (zone >= t), apply the morphological
    op(s), then reassemble zone = sum of the true thresholds.

    method: close_only (fill holes, material only grows), open_only (remove
    protrusions, material only shrinks), close_open / open_close (both).

    Directional SE (ply_angles + tape_width) uses per-angle star arms;
    per_zone_angles restricts each threshold to the angles that survive in that
    zone. directional_mode: "star" (star SE only) or "clamped" (star then
    clipped against the isotropic result). edge_overshoot virtually extends the
    panel boundary by that many mm (edge replication) before filtering so edge
    features aren't penalised. Returns the filtered grid, same shape/range."""
    valid = (zone_grid >= 0)
    filtered = np.zeros_like(zone_grid)

    overshoot_px = 0
    if edge_overshoot > 0:
        overshoot_px = max(1, int(round(edge_overshoot / element_size)))
        zone_grid = np.pad(zone_grid, overshoot_px, mode='edge')
        if per_angle_zone_grids is not None:
            per_angle_zone_grids = {
                ua: np.pad(ag, overshoot_px, mode='edge')
                for ua, ag in per_angle_zone_grids.items()
            }
        valid = (zone_grid >= 0)
        filtered = np.zeros_like(zone_grid)
        if verbose:
            print(f"  Edge overshoot: {edge_overshoot:.1f} mm ({overshoot_px} px)")

    _DISPATCH = {
        "close_open": _masked_close_open,
        "open_close": _masked_open_close,
        "close_only": _masked_close,
        "open_only": _masked_open,
        "erode_only": _masked_erode,
        "dilate_only": _masked_dilate,
    }
    if method not in _DISPATCH:
        raise ValueError(f"Unknown filtering method '{method}'. "
                         f"Use 'close_only', 'open_only', 'close_open', "
                         f"'open_close', 'dilate_only', or 'erode_only'.")
    morph_fn = _DISPATCH[method]

    if ply_angles is not None:
        # ================================================================
        # NEW DIRECTIONAL FILTERING: isotropic first, then decouple per
        # angle and enforce MCL fiber length directionally.
        # ================================================================
        se_disk = make_structuring_element(mcl / 2.0, element_size)
        if verbose:
            print(f"\n  Directional: isotropic disk (r={mcl/2:.0f}mm) then decoupled fiber enforcement")
            print(f"  Isotropic filtering method: {method}")

        for t in range(1, n_zones):
            mask = (zone_grid >= t) & valid
            n_before = int(mask.sum())
            result = morph_fn(mask, se_disk, valid)
            filtered += result.astype(int)
            n_after = int(result.sum())
            if verbose:
                print(f"    Threshold >= {t}: {n_before} -> {n_after} ({n_after - n_before:+d} elements)")

        # Crop overshoot padding before decoupled filtering
        if overshoot_px > 0:
            s = overshoot_px
            filtered = filtered[s:-s, s:-s]
            zone_grid = zone_grid[s:-s, s:-s]
            valid = (zone_grid >= 0)
            overshoot_px = 0  # prevent double-crop below

        filtered[~valid] = -1

        # Decoupled directional fiber enforcement
        if per_zone_angles is not None and mcl > 0:
            _ux = ux if ux is not None else np.arange(filtered.shape[1]) * element_size
            _uy = uy if uy is not None else np.arange(filtered.shape[0]) * element_size
            t_width = tape_width or element_size

            filtered, _per_angle = decoupled_directional_filter(
                filtered, per_zone_angles, _ux, _uy,
                mcl, element_size, t_width, n_zones,
                edge_overshoot=edge_overshoot,
                verbose=verbose,
            )
            if _out_per_angle is not None:
                _out_per_angle.clear()
                _out_per_angle.update(_per_angle)

    else:
        se_disk = make_structuring_element(mcl / 2.0, element_size)
        if verbose:
            print(f"\n  Structuring element: isotropic disk, radius={mcl/2:.0f}mm")
            print(f"  Filtering method: {method}")

        for t in range(1, n_zones):
            mask = (zone_grid >= t) & valid
            n_before = int(mask.sum())
            result = morph_fn(mask, se_disk, valid)
            filtered += result.astype(int)
            n_after = int(result.sum())
            if verbose:
                print(f"    Threshold >= {t}: {n_before} -> {n_after} ({n_after - n_before:+d} elements)")

    if overshoot_px > 0:
        s = overshoot_px
        filtered = filtered[s:-s, s:-s]

    if post_process_mcl > 0:
        if verbose:
            print(f"\n  Post-processing: close_open (disk) at MCL={post_process_mcl:.1f} mm")
        filtered = apply_morphology_filter(
            filtered, post_process_mcl, element_size, n_zones,
            method="close_open", ply_angles=None, tape_width=None,
            directional_mode=None,
            per_zone_angles=None,
            run_fiber_enforcement=False,
            post_process_mcl=0.0,
            verbose=verbose,
        )

    if local_radius_check:
        n_before = int(filtered.sum())
        filtered = enforce_local_radius(filtered, n_zones, mcl, element_size)
        n_after = int(filtered.sum())
        if n_before != n_after and verbose:
            print(f"  Local radius check (r={mcl/2:.0f}mm): {n_after - n_before:+d} zone-pixels")

    if run_fiber_enforcement and per_zone_angles is not None and mcl > 0 and ply_angles is None:
        _ux = ux if ux is not None else np.arange(zone_grid.shape[1]) * element_size
        _uy = uy if uy is not None else np.arange(zone_grid.shape[0]) * element_size
        t_width = tape_width or element_size
        check_mcl = mcl_check if mcl_check > 0 else mcl
        filtered = enforce_fiber_length(
            filtered, _ux, _uy, per_zone_angles, check_mcl, element_size, t_width,
            verbose=verbose,
        )

    return filtered


def print_steiner_verification(
    zone_grid: np.ndarray,
    mcl: float,
    element_size: float,
    n_zones: int,
):
    """Print a table comparing Steiner-formula area-change predictions
    against actual dilate-only results for each zone threshold.

    The Steiner formula for a 2D set K dilated by a disk of radius r is:

        A(K ⊕ B_r) = A(K) + P(K) · r + π · r² · χ(K)

    where P(K) is the Crofton perimeter and χ the Euler characteristic.
    This function verifies that the morphological operations follow the
    theoretical scaling by printing predicted vs. actual area deltas at
    each threshold level of the zone grid.
    """
    from skimage.measure import perimeter_crofton, euler_number
    from skimage.morphology import convex_hull_image
    from scipy.ndimage import distance_transform_edt, label

    def _P_int(comp: np.ndarray, v: np.ndarray) -> float:
        """Crofton perimeter counting only interior edges (both
        neighbours inside *v*).  Excludes edges at the panel boundary."""
        m = comp.astype(np.int32)
        k = v.astype(np.int32)
        h = (np.abs(m[:-1, :] - m[1:, :]) * (k[:-1, :] & k[1:, :])).sum()
        w = (np.abs(m[:, :-1] - m[:, 1:]) * (k[:, :-1] & k[:, 1:])).sum()
        return float(h + w)

    r_mm = mcl / 2.0
    r_el = int(round(r_mm / element_size))
    valid = (zone_grid >= 0)

    # Run both in parallel — same perimeter, opposite sign.
    dil_filtered = apply_morphology_filter(
        zone_grid, mcl, element_size, n_zones, method="dilate_only",
    )
    ero_filtered = apply_morphology_filter(
        zone_grid, mcl, element_size, n_zones, method="erode_only",
    )

    # ── 1  DILATION ──────────────────────────────────────────────────────
    print(f"\n  Steiner Formula — Dilation")
    print(f"    MCL = {mcl:.1f} mm,  r = {r_mm:.1f} mm ({r_el} elements)")
    print(f"    dA_pred = +P_int * r + pi * r^2")
    print(f"  {'Thresh':>7} {'Area':>6} {'P_int':>7} {'P_hull':>7} {'Cvx':>6}"
          f"  {'dA_act':>8}   {'dA_pred':>8} {'act/pred':>9} {'dA_%':>7}")
    print(f"  {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6}  {'-'*8}   {'-'*8} {'-'*9} {'-'*7}")

    t_dil_act = 0; t_dil_pred = 0.0; t_dil_A = 0
    for t in range(1, n_zones):
        mask = (zone_grid >= t) & valid
        m_after = (dil_filtered >= t) & valid
        A = int(mask.sum()); Aa = int(m_after.sum())
        if A == 0: continue
        t_dil_A += A

        labels, n_lab = label(mask.astype(bool))
        Pi = Ph = 0.0
        for li in range(1, n_lab + 1):
            c = (labels == li)
            Pi += _P_int(c, valid)
            hull = convex_hull_image(c)
            Ph += _P_int(hull, valid)
        cvx = Pi / Ph if Ph > 1e-9 else 1.0

        dA_pred = Pi * r_mm + np.pi * r_mm ** 2
        dA_act = Aa - A
        t_dil_act += dA_act; t_dil_pred += dA_pred
        r = dA_act / dA_pred if dA_pred > 0 else 0.0
        pct = dA_act / A * 100.0 if A > 0 else 0.0
        print(f"  {'>= ' + str(t):>7} {A:>6} {Pi:>7.1f} {Ph:>7.1f} {cvx:>6.2f}"
              f"  {dA_act:>+8d}   {dA_pred:>+8.1f}  {r:>9.2f} {pct:>+7.2f}")

    tr = t_dil_act / t_dil_pred if t_dil_pred > 0 else 0.0
    tpa = t_dil_act / t_dil_A * 100.0 if t_dil_A > 0 else 0.0
    print(f"  {'Tot':>7} {'':>6} {'':>7} {'':>7} {'':>6}"
          f"  {t_dil_act:>+8d}   {t_dil_pred:>+8.1f}  {tr:>9.2f} {tpa:>+7.2f}")
    print(f"\n  dA_act/dA_pred should approach 1 as Cvx approaches 1.")

    # ── 2  EROSION ───────────────────────────────────────────────────────
    print(f"\n  Steiner Formula — Erosion")
    print(f"    r = {r_mm:.1f} mm,  dA_pred = -P * r + pi * r^2")
    print(f"  {'Thresh':>7} {'Area':>6} {'P':>7} {'P/A':>7}  {'dA':>8}   {'dA_%':>7}")
    print(f"  {'-'*7} {'-'*6} {'-'*7} {'-'*7}  {'-'*8} {'-'*7}")

    t_ero_act = 0; t_ero_A = 0
    for t in range(1, n_zones):
        mask = (zone_grid >= t) & valid
        m_after = (ero_filtered >= t) & valid
        A = int(mask.sum()); Aa = int(m_after.sum())
        if A == 0: continue
        t_ero_A += A

        labels, n_lab = label(mask.astype(bool))
        Pi = 0.0
        for li in range(1, n_lab + 1):
            c = (labels == li)
            Pi += _P_int(c, valid)
        pa = Pi / A if A > 0 else 0.0

        dA_act = Aa - A
        t_ero_act += dA_act
        pct = dA_act / A * 100.0 if A > 0 else 0.0
        print(f"  {'>= ' + str(t):>7} {A:>6} {Pi:>7.1f} {pa:>7.4f}  {dA_act:>+8d} {pct:>+7.2f}")

    print(f"  {'Tot':>7} {'':>6} {'':>7} {'':>7}  {t_ero_act:>+8d}")


def grid_to_elem_nplies(
    filtered_grid: np.ndarray,
    eid_grid: np.ndarray,
    zone_ply_counts: list[int],
) -> dict[int, int]:
    """
    Convert the filtered 2D zone grid back to {eid: n_plies} mapping.

    Args:
        filtered_grid:   2D array of zone indices after filtering
        eid_grid:        2D array of element IDs
        zone_ply_counts: sorted list of ply counts per zone (index = zone rank)

    Returns:
        {element_id: n_plies}
    """
    sorted_counts = sorted(zone_ply_counts)
    elem_nplies = {}
    ny, nx = filtered_grid.shape

    for iy in range(ny):
        for ix in range(nx):
            eid = eid_grid[iy, ix]
            if eid < 0:
                continue
            zone = filtered_grid[iy, ix]
            zone = max(0, min(zone, len(sorted_counts) - 1))
            elem_nplies[eid] = sorted_counts[zone]

    return elem_nplies


def elem_nplies_to_grid(
    elem_nplies: dict[int, int],
    eid_grid: np.ndarray,
    zone_ply_counts: list[int],
) -> np.ndarray:
    """Convert a per-element ply-count map back to a compact ordinal grid.

    Unlike the raw threshold grid, this representation has exactly one ordinal
    level per distinct exported laminate thickness. This is the same logic used
    by the scatter-based comparison plots and avoids phantom contours where two
    raw zone indices collapse to the same final laminate.
    """
    sorted_counts = sorted(zone_ply_counts)
    n_to_zone = {int(n): i for i, n in enumerate(sorted_counts)}
    out = np.full(eid_grid.shape, -1, dtype=int)
    ny, nx = eid_grid.shape
    for iy in range(ny):
        for ix in range(nx):
            eid = int(eid_grid[iy, ix])
            if eid < 0 or eid not in elem_nplies:
                continue
            out[iy, ix] = n_to_zone[int(elem_nplies[eid])]
    return out


# ==============================================================================
#  Visualization
# ==============================================================================

def _draw_stiffener_outline_scatter(ax, stiffener_xy, linewidth=0.5):
    """Draw stiffener boundary as a black contour from scatter coordinates."""
    sx = np.asarray(stiffener_xy[0])
    sy = np.asarray(stiffener_xy[1])
    ux_s = np.sort(np.unique(np.round(sx, 2)))
    uy_s = np.sort(np.unique(np.round(sy, 2)))
    if len(ux_s) < 2 or len(uy_s) < 2:
        return
    stiff_grid = np.zeros((len(uy_s), len(ux_s)))
    x_to_ix = {round(float(x), 2): i for i, x in enumerate(ux_s)}
    y_to_iy = {round(float(y), 2): i for i, y in enumerate(uy_s)}
    for x, y in zip(sx, sy):
        ix = x_to_ix.get(round(float(x), 2))
        iy = y_to_iy.get(round(float(y), 2))
        if ix is not None and iy is not None:
            stiff_grid[iy, ix] = 1.0
    ax.contour(ux_s, uy_s, stiff_grid,
               levels=[0.5], colors="black", linewidths=linewidth, zorder=5)

def plot_filtered_comparison(
    X: np.ndarray,
    Y: np.ndarray,
    eids: list[int],
    elem_nplies_before: dict[int, int],
    elem_nplies_after: dict[int, int],
    sst,
    save_path: str = "filtered_zones_comparison.png",
    stiffener_xy=None,
    stiffener_base_nplies=None,
):
    """Plot side-by-side comparison of zones before and after morphological filtering."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    all_counts = sorted(set(elem_nplies_before.values()) | set(elem_nplies_after.values()))
    n_zones = len(all_counts)
    c2i = {n: i for i, n in enumerate(all_counts)}

    colors_before = [c2i[elem_nplies_before[eid]] for eid in eids]
    colors_after = [c2i[elem_nplies_after[eid]] for eid in eids]

    cmap = plt.get_cmap("RdYlBu_r", n_zones)
    norm = mcolors.BoundaryNorm(np.arange(n_zones + 1) - 0.5, n_zones)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for ax, colors, title in [
        (ax1, colors_before, "SST discretized"),
        (ax2, colors_after, "SST + MCL filtering"),
    ]:
        sc = ax.scatter(X, Y, c=colors, s=8, cmap=cmap, norm=norm,
                        marker="s", edgecolors="none")
        if stiffener_xy is not None:
            _draw_stiffener_outline_scatter(ax, stiffener_xy)
        ax.set_aspect("equal")
        # Explicit limits from the data: freezing autoscale before matplotlib's
        # lazy autoscaling has run would leave the axes at the default 0..1.
        _Xa, _Ya = np.asarray(X), np.asarray(Y)
        if _Xa.size and _Ya.size:
            ax.set_xlim(_Xa.min(), _Xa.max())
            ax.set_ylim(_Ya.min(), _Ya.max())
        ax.autoscale(False)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle=":", alpha=0.3)

    # Single colorbar
    cbar = fig.colorbar(sc, ax=[ax1, ax2], ticks=np.arange(n_zones), shrink=0.8)
    labels = [f"{n} plies ({n * sst.manufacturing_ply_thickness:.2f} mm)" for n in all_counts]
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("SST Zone", rotation=270, labelpad=20, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def _mcl_to_pct(mcl_values: list[float], panel_size: float) -> list[float]:
    return [m / panel_size * 100.0 for m in mcl_values]


def plot_mcl_sweep(
    mcl_values: list[float],
    pct_vs_opt: list[float],
    pct_vs_baseline: list[float],
    pct_sst_vs_opt: float,
    pct_sst_vs_baseline: float,
    save_path: str = "mcl_sweep.png",
    panel_size: float = None,
):
    """Plot volume-scaled filtered performance vs MCL.

    Left y-axis:  performance gain vs theoretical optimum (%)
    Right y-axis: performance gain vs baseline uniform panel (%)
    Also plots the SST-only (unfiltered) reference as a horizontal line.
    Positive values mean the filtered/scaled panel has lower compliance
    than the reference (i.e. performs better).
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.set_xticks(np.linspace(x[0], x[-1], len(x)))
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax1.plot(x, pct_vs_opt, "o-", linewidth=2,
                      markersize=6, label="vs Guide Ply Optimum", color="black")

    ax2 = ax1.twinx()
    ax2.plot(x, pct_vs_baseline, "o-", linewidth=2, markersize=6, label="vs Baseline (uniform)")
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)

    ax1.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_comparison(
    mcl_values: list[float],
    method_results: dict[str, dict[str, list[float]]],
    pct_sst_vs_opt: float,
    pct_sst_vs_baseline: float,
    save_path: str = "mcl_sweep_combined.png",
    panel_size: float = None,
):
    """Plot multiple filter methods on the same MCL sweep figure."""
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    colors = {"close_only": "#1a936f", "open_only": "#457b9d", "close_open": "#e76f51", "open_close": "#7b2d8e"}
    markers = {"close_only": "o", "open_only": "v", "close_open": "s", "open_close": "D"}
    default_color = "#2c3e50"

    for idx, (method, results) in enumerate(method_results.items()):
        color = colors.get(method, default_color)
        m = markers.get(method, "o")
        label_method = method.replace("_", "-")

        ax1.plot(
            x,
            results["pct_vs_opt"],
            marker=m,
            color=color,
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=label_method,
        )
        ax2.plot(
            x,
            results["pct_vs_baseline"],
            marker=m,
            color=color,
            linestyle="--",
            linewidth=2,
            markersize=6,
        )

    ax1.axhline(
        y=pct_sst_vs_opt,
        color="#444444",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"SST-only vs Guide Opt ({pct_sst_vs_opt:+.1f}%)",
    )
    ax2.axhline(
        y=pct_sst_vs_baseline,
        color="#666666",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"SST-only vs Baseline ({pct_sst_vs_baseline:+.1f}%)",
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_directional_comparison(
    mcl_values: list[float],
    directional_results: dict[str, dict[str, list[float]]],
    isotropic_results: dict[str, dict[str, list[float]]],
    pct_sst_vs_opt: float,
    pct_sst_vs_baseline: float,
    save_path: str = "mcl_sweep_directional_comparison.png",
    panel_size: float = None,
):
    """Plot directional vs isotropic filtering on the same MCL sweep figure.

    Same colour per method; solid = directional (improved), dashed = isotropic.
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()

    method_colors = {"close_only": "#1a936f", "open_only": "#457b9d", "close_open": "#e76f51", "open_close": "#7b2d8e"}
    markers = {"close_only": "o", "open_only": "v", "close_open": "s", "open_close": "D"}

    for method in directional_results:
        color = method_colors.get(method, "#2c3e50")
        m = markers.get(method, "o")
        label = method.replace("_", "-")

        ax1.plot(
            x, isotropic_results[method]["pct_vs_opt"],
            marker=m, color=color, linestyle="--", linewidth=2,
            markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            label=f"{label} (isotropic)",
        )
        ax1.plot(
            x, directional_results[method]["pct_vs_opt"],
            marker=m, color=color, linestyle="-", linewidth=2,
            markersize=6,
            label=f"{label} (directional)",
        )

        ax2.plot(
            x, isotropic_results[method]["pct_vs_baseline"],
            marker=m, color=color, linestyle=":", linewidth=1.5,
            markersize=6, markerfacecolor="none", markeredgewidth=1.5,
        )
        ax2.plot(
            x, directional_results[method]["pct_vs_baseline"],
            marker=m, color=color, linestyle="-.", linewidth=1.5,
            markersize=6,
        )

    ax1.axhline(
        y=pct_sst_vs_opt, color="#444444", linestyle=":", linewidth=1.2, alpha=0.6,
        label=f"SST-only vs Guide Opt ({pct_sst_vs_opt:+.1f}%)",
    )
    ax2.axhline(
        y=pct_sst_vs_baseline, color="#666666", linestyle=":", linewidth=1.2, alpha=0.4,
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.3)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_guide_comparison(
    mcl_values: list[float],
    default_results: dict[str, dict[str, list[float]]],
    optimized_results: dict[str, dict[str, list[float]]],
    pct_sst_default_vs_opt: float,
    pct_sst_default_vs_baseline: float,
    pct_sst_opt_vs_opt: float,
    pct_sst_opt_vs_baseline: float,
    save_path: str = "mcl_sweep_guide_comparison.png",
    panel_size: float = None,
):
    """Plot default vs optimized guide stack on the same MCL sweep figure.

    Same colour per method; solid = optimized (improved), dashed = default.
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()

    method_colors = {"close_only": "#1a936f", "open_only": "#457b9d", "close_open": "#e76f51", "open_close": "#7b2d8e"}
    markers = {"close_only": "o", "open_only": "v", "close_open": "s", "open_close": "D"}

    for method in default_results:
        color = method_colors.get(method, "#2c3e50")
        m = markers.get(method, "o")
        label = method.replace("_", "-")

        ax1.plot(
            x, default_results[method]["pct_vs_opt"],
            marker=m, color=color, linestyle="--", linewidth=2,
            markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            label=f"{label} (default guide)",
        )
        ax1.plot(
            x, optimized_results[method]["pct_vs_opt"],
            marker=m, color=color, linestyle="-", linewidth=2,
            markersize=6,
            label=f"{label} (optimized guide)",
        )

        ax2.plot(
            x, default_results[method]["pct_vs_baseline"],
            marker=m, color=color, linestyle=":", linewidth=1.5,
            markersize=6, markerfacecolor="none", markeredgewidth=1.5,
        )
        ax2.plot(
            x, optimized_results[method]["pct_vs_baseline"],
            marker=m, color=color, linestyle="-.", linewidth=1.5,
            markersize=6,
        )

    ax1.axhline(
        y=pct_sst_default_vs_opt, color=method_colors["close_only"],
        linestyle=":", linewidth=1.2, alpha=0.5,
        label=f"SST default vs Opt ({pct_sst_default_vs_opt:+.1f}%)",
    )
    ax1.axhline(
        y=pct_sst_opt_vs_opt, color=method_colors["close_only"],
        linestyle="-", linewidth=1.0, alpha=0.5,
        label=f"SST optimized vs Opt ({pct_sst_opt_vs_opt:+.1f}%)",
    )
    ax2.axhline(
        y=pct_sst_default_vs_baseline, color=method_colors["open_close"],
        linestyle=":", linewidth=1.0, alpha=0.3,
    )
    ax2.axhline(
        y=pct_sst_opt_vs_baseline, color=method_colors["open_close"],
        linestyle="-.", linewidth=1.0, alpha=0.3,
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.3)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_volume(
    mcl_values: list[float],
    method_vol_deltas: dict[str, list[float]],
    save_path: str = "mcl_sweep_volume.png",
    panel_size: float = None,
):
    """Plot volume delta (%) vs MCL for multiple filter methods."""
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {"close_only": "#1a936f", "open_only": "#457b9d", "close_open": "#e76f51", "open_close": "#7b2d8e"}
    markers = {"close_only": "o", "open_only": "v", "close_open": "s", "open_close": "D"}
    default_color = "#2c3e50"

    for idx, (method, deltas) in enumerate(method_vol_deltas.items()):
        color = colors.get(method, default_color)
        m = markers.get(method, "o")
        label_method = method.replace("_", "-")
        ax.plot(
            x,
            deltas,
            marker=m,
            color=color,
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=label_method,
        )

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Volume Delta vs Unfiltered SST (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_volume_dilate_erode(
    mcl_values: list[float],
    dilate_deltas: list[float],
    erode_deltas: list[float],
    dilate_iso: list[float] = None,
    erode_iso: list[float] = None,
    save_path: str = "mcl_sweep_volume_dilate_erode.png",
    panel_size: float = None,
):
    """Volume delta vs MCL for pure dilate and erode.

    Solid = directional (star), dashed = isotropic (disk).
    Dilate is always positive (adds material), erode always negative.
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, dilate_deltas,
            marker="^", color="#2ca02c", linestyle="-",
            linewidth=2, markersize=7, markerfacecolor="#2ca02c",
            label="dilate-only (directional)")
    ax.plot(x, erode_deltas,
            marker="v", color="#d62728", linestyle="-",
            linewidth=2, markersize=7, markerfacecolor="#d62728",
            label="erode-only (directional)")
    if dilate_iso is not None:
        ax.plot(x, dilate_iso,
                marker="^", color="#2ca02c", linestyle="--",
                linewidth=1.5, markersize=7, markerfacecolor="none",
                markeredgewidth=1.5, markeredgecolor="#2ca02c",
                label="dilate-only (isotropic)")
    if erode_iso is not None:
        ax.plot(x, erode_iso,
                marker="v", color="#d62728", linestyle="--",
                linewidth=1.5, markersize=7, markerfacecolor="none",
                markeredgewidth=1.5, markeredgecolor="#d62728",
                label="erode-only (isotropic)")

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Volume Delta vs Unfiltered SST (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_zone_grid_smoothed(
    zone_grid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    zone_ply_counts: list[int],
    ply_thickness: float,
    save_path: str = "sst_zones_filtered_smoothed.png",
    upscale: int = 8,
    sigma: float = 1.0,
    title: str = None,
):
    """Plot zone grid with smoothed boundaries via threshold-decomposed upsampling.

    Instead of showing raw pixels, each threshold level (zone >= t) is upsampled
    and lightly Gaussian-blurred, then summed into a continuous scalar field.
    Contourf renders smooth boundaries between zones while preserving the nested
    zone structure and approximate zone areas.

    This is a display-only transformation — the underlying zone_grid is not modified.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    valid = zone_grid >= 0
    n_zones = len(zone_ply_counts)

    field_hi = None
    for t in range(1, n_zones):
        mask = ((zone_grid >= t) & valid).astype(np.float64)
        mask_hi = zoom(mask, upscale, order=1)
        mask_hi = gaussian_filter(mask_hi, sigma=sigma)
        if field_hi is None:
            field_hi = np.zeros_like(mask_hi)
        field_hi += mask_hi

    valid_hi = zoom(valid.astype(np.float64), upscale, order=0) > 0.5
    field_hi = np.where(valid_hi, field_hi, np.nan)

    x_hi = np.linspace(ux.min(), ux.max(), field_hi.shape[1])
    y_hi = np.linspace(uy.min(), uy.max(), field_hi.shape[0])

    cmap = plt.get_cmap("RdYlBu_r", n_zones)
    norm = mcolors.BoundaryNorm(np.arange(n_zones + 1) - 0.5, n_zones)

    fig, ax = plt.subplots(figsize=(11, 8))
    cf = ax.contourf(
        x_hi, y_hi, field_hi,
        levels=np.arange(n_zones + 1) - 0.5,
        cmap=cmap, norm=norm,
    )
    ax.contour(
        x_hi, y_hi, field_hi,
        levels=np.arange(n_zones - 1) + 0.5,
        colors="k", linewidths=0.5, alpha=0.25,
    )

    sorted_counts = sorted(zone_ply_counts)
    cbar = plt.colorbar(cf, ticks=np.arange(n_zones))
    cbar.ax.set_yticklabels([
        f"{n} plies ({n * ply_thickness:.2f} mm)" for n in sorted_counts
    ])
    cbar.set_label("SST Zone", rotation=270, labelpad=20, fontweight="bold")

    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_zone_grid_manufacturing(
    zone_grid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    zone_ply_counts: list[int],
    ply_thickness: float,
    save_path: str = "sst_zones_filtered_manufacturing.png",
    upscale: int = 6,
    sigma: float = 3.0,
    title: str = None,
    stiffener_grid_mask=None,
    stiffener_base_zone_idx=None,
):
    """Plot zone grid with smooth boundaries via upsampling + Gaussian blur.

    Identical to ``plot_zone_grid_smoothed`` but with higher sigma (3.0)
    to eliminate visible pixel-scale artifacts.  Uses threshold-decomposed
    upsampling + contourf for correct zone nesting.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    display_grid = zone_grid.copy()

    n_zones = len(zone_ply_counts)
    sorted_counts = sorted(zone_ply_counts)
    cmap = plt.get_cmap("RdYlBu_r", n_zones)
    norm = mcolors.BoundaryNorm(np.arange(n_zones + 1) - 0.5, n_zones)
    valid = (display_grid >= 0)

    field_hi = None
    for t in range(1, n_zones):
        mask = ((display_grid >= t) & valid).astype(np.float64)
        mask_hi = zoom(mask, upscale, order=1)
        mask_hi = gaussian_filter(mask_hi, sigma=sigma)
        if field_hi is None:
            field_hi = np.zeros_like(mask_hi)
        field_hi += mask_hi

    valid_hi = zoom(valid.astype(np.float64), upscale, order=0) > 0.5
    if field_hi is None:
        field_hi = np.zeros((int(valid.shape[0] * upscale), int(valid.shape[1] * upscale)), dtype=np.float64)
    field_hi = np.where(valid_hi, field_hi, np.nan)
    x_hi = np.linspace(ux.min(), ux.max(), field_hi.shape[1])
    y_hi = np.linspace(uy.min(), uy.max(), field_hi.shape[0])

    fig, ax = plt.subplots(figsize=(11, 8))
    cf = ax.contourf(
        x_hi, y_hi, field_hi,
        levels=np.arange(n_zones + 1) - 0.5,
        cmap=cmap, norm=norm,
    )

    ax.set_aspect("equal")
    if stiffener_grid_mask is not None:
        ax.contour(ux, uy, stiffener_grid_mask.astype(float),
                   levels=[0.5], colors="black", linewidths=0.5, zorder=5)
    if ux.size > 1 and uy.size > 1:
        ax.set_xlim(ux.min(), ux.max())
        ax.set_ylim(uy.min(), uy.max())
    ax.autoscale(False)
    if title:
        ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, ticks=np.arange(n_zones))
    cbar.ax.set_yticklabels([
        f"{n} plies ({n * ply_thickness:.2f} mm)" for n in sorted_counts
    ])
    cbar.set_label("SST Zone", rotation=270, labelpad=20, fontweight="bold")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_fso_vs_filtered(
    zone_grid: np.ndarray,
    filtered_grid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    zone_ply_counts: list[int],
    ply_thickness: float,
    save_path: str = "fso_vs_filtered.png",
    upscale: int = 6,
    sigma: float = 3.0,
    stiffener_grid_mask=None,
    stiffener_base_zone_idx=None,
):
    """Side-by-side: SST discretized vs SST + filtered output.

    Left:  SST-only zone grid (before MCL filtering).
    Right: SST + MCL filtered + post-processed zone map.
    Both rendered with the same upsampled + blurred contourf.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_zones = len(zone_ply_counts)
    sorted_counts = sorted(zone_ply_counts)
    cmap = plt.get_cmap("RdYlBu_r", n_zones)
    norm = mcolors.BoundaryNorm(np.arange(n_zones + 1) - 0.5, n_zones)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

    def _draw_contourf(ax, grid, title):
        display_grid = grid.copy()
        valid = (display_grid >= 0)
        valid_hi = zoom(valid.astype(np.float64), upscale, order=0) > 0.5
        field = np.zeros(valid_hi.shape, dtype=np.float64)
        for t in range(1, n_zones):
            mask = ((display_grid >= t) & valid).astype(np.float64)
            mask_hi = zoom(mask, upscale, order=1)
            mask_hi = gaussian_filter(mask_hi, sigma=sigma)
            field += mask_hi
        field = np.where(valid_hi, field, np.nan)
        x_hi = np.linspace(ux.min(), ux.max(), field.shape[1])
        y_hi = np.linspace(uy.min(), uy.max(), field.shape[0])
        cf = ax.contourf(x_hi, y_hi, field,
                         levels=np.arange(n_zones + 1) - 0.5,
                         cmap=cmap, norm=norm)
        ax.set_aspect("equal")
        if stiffener_grid_mask is not None:
            ax.contour(ux, uy, stiffener_grid_mask.astype(float),
                       levels=[0.5], colors="black", linewidths=0.5, zorder=5)
        ax.set_xlim(ux.min(), ux.max()); ax.set_ylim(uy.min(), uy.max())
        ax.autoscale(False)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle=":", alpha=0.3)
        return cf

    _draw_contourf(ax1, zone_grid,    "SST discretized — before MCL filtering")
    _draw_contourf(ax2, filtered_grid, "SST + filtering pipeline")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=[ax1, ax2], shrink=0.8, ticks=np.arange(n_zones))
    cbar.ax.set_yticklabels([
        f"{n} plies ({n * ply_thickness:.2f} mm)" for n in sorted_counts
    ])
    cbar.set_label("SST Zone", rotation=270, labelpad=15, fontweight="bold")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_zone_fiber_coverage(
    zone_grid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    sst,
    tape_width: float = 1.0,
    save_prefix: str = "fiber_coverage",
    mcl: float = 0.0,
    per_angle_zone_grids: dict[float, np.ndarray] | None = None,
    stiffener_grid_mask=None,
    stiffener_base_zone_idx=None,
    stiffener_angle_grid=None,
):
    """Plot AFP fiber centerlines per threshold per surviving angle.

    Each threshold is saved as a separate PNG — one figure per zone
    level with one subplot per angle, at 600 DPI.
    Fibre segments shorter than *mcl* are drawn in red, UNLESS they
    touch the panel boundary — those are always green (manufacturable
    because AFP tows can overshoot beyond the panel edge).

    When *per_angle_zone_grids* is provided (from decoupled directional
    filtering), each angle subplot uses its own zone grid to show the
    actual decoupled ply shape for that angle.
    """
    import matplotlib.pyplot as plt

    display_grid = zone_grid.copy()
    valid = (display_grid >= 0)
    n_zones = len(sst.zone_ply_counts)
    unique_angles = (sorted(float(a) for a in per_angle_zone_grids.keys())
                     if per_angle_zone_grids else sorted(set(sst.ply_angles)))
    n_angles = len(unique_angles)

    for z in range(1, n_zones):
        combined_mask = (display_grid >= z) & valid
        n_plies = sst.zone_ply_counts[z]

        # Angles that survive at this zone level
        lam = sst.get_laminate(n_plies)
        lam_angles = set(lam)

        fig, axes = plt.subplots(1, n_angles,
                                 figsize=(4 * n_angles, 4))
        if n_angles == 1:
            axes = [axes]

        # MCL scale bar position (shared across all subplots)
        es = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
        bx = ux.min() + es * 5
        by = uy.min() + es * 5

        for j, angle in enumerate(unique_angles):
            ax = axes[j]
            if angle in lam_angles:
                # Use per-angle mask when available, otherwise combined mask
                if per_angle_zone_grids is not None and angle in per_angle_zone_grids:
                    ag = per_angle_zone_grids[angle].copy()
                    ag_valid = (ag >= 0)
                    angle_mask = (ag >= z) & ag_valid
                else:
                    angle_mask = combined_mask
                tot, sne = _plot_angle_fibers(ax, angle_mask, ux, uy, angle, tape_width, mcl=mcl)
            else:
                tot, sne = 0, 0
                ax.text(0.5, 0.5, "angle not\npresent",
                        transform=ax.transAxes, ha='center',
                        va='center', fontsize=10, color="gray")

            ax.set_title(rf"${angle:.0f}^\circ$", fontsize=12)
            _format_fib_axes(ax, ux, uy, panel_outline=False)
            if stiffener_grid_mask is not None:
                ax.contour(ux, uy, stiffener_grid_mask.astype(float),
                           levels=[0.5], colors="gray", linewidths=0.5, zorder=5)
            if mcl > 0:
                # MCL scale bar at bottom‑left
                ax.plot([bx, bx + mcl], [by, by], 'r-', linewidth=3, solid_capstyle='butt', zorder=20)
                ax.plot([bx, bx], [by - es, by + es], 'r-', linewidth=2, zorder=20)
                ax.plot([bx + mcl, bx + mcl], [by - es, by + es], 'r-', linewidth=2, zorder=20)
                ax.text(bx + mcl / 2, by + es * 1.0, 'MCL',
                        ha='center', va='bottom', fontsize=7, fontweight='bold', color='red', zorder=20)
                # manufacturable percentage (edge-touching fibers count as manufacturable)
                if angle in lam_angles and tot > 0:
                    pct = (1 - sne / tot) * 100
                    mid_x = (ux.min() + ux.max()) / 2
                    ax.text(mid_x, by, f'{pct:.0f}% \u2265 MCL',
                            ha='center', va='center', fontsize=7, color='green', zorder=20)

        save_path = f"{save_prefix}_zone{z}.png"
        from helper import _savefig_locked_retry
        _savefig_locked_retry(plt, save_path, dpi=600, bbox_inches="tight")
        plt.close()
        if _VERBOSE:
            print(f"  Saved: {save_path}")


def _format_fib_axes(ax, ux, uy, panel_outline=False):
    """Common axis formatting for fibre-coverage subplots."""
    ax.set_aspect("equal")
    ax.set_xlim(ux.min(), ux.max())
    ax.set_ylim(uy.min(), uy.max())
    ax.autoscale(False)
    ax.tick_params(left=False, right=False, labelleft=False,
                   bottom=False, top=False, labelbottom=False)


def _plot_angle_fibers(ax, mask, ux, uy, angle_deg, tape_width, mcl=0.0):
    """Draw fibre centre-lines at *angle_deg* within *mask*.

    Uses AFP-robot-style equidistant offsets: first course at half
    *tape_width* from the panel edge, subsequent courses every *tape_width*.
    Segment endpoints from exact physical fibre-line equation.
    Segments shorter than *mcl* are drawn in red unless they touch the
    panel boundary, in which case they are always green (manufacturable
    because AFP tows can overshoot beyond the panel edge).

    Returns:
        (total_segments, short_non_edge_segments)
    """
    ny, nx = mask.shape
    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = -st, ct

    element_size = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0

    offsets_px, offsets_mm, cx_phys, cy_phys, _cp, _sp = \
        _get_equidistant_offsets(ux, uy, angle_deg, tape_width, element_size)
    if len(offsets_px) == 0:
        return 0, 0

    cx, cy = nx / 2.0, ny / 2.0
    max_r = np.sqrt(nx ** 2 + ny ** 2)

    tau_off_px = (cx_phys * ct + cy_phys * st) / element_size

    t_px = np.linspace(-max_r, max_r, int(4 * max_r) + 1)

    total_segs = 0
    short_non_edge = 0

    for off_px, off_mm in zip(offsets_px, offsets_mm):
        ox = cx + off_px * cp
        oy = cy + off_px * sp
        px = ox + t_px * ct
        py = oy + t_px * st

        iy = np.round(py).astype(int)
        ix = np.round(px).astype(int)

        in_b = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        iyc = iy.clip(0, ny - 1)
        ixc = ix.clip(0, nx - 1)
        valid = in_b & mask[iyc, ixc]

        if not valid.any():
            continue

        v = valid.astype(int)
        d = np.diff(v)
        starts = np.where(d == 1)[0] + 1
        ends   = np.where(d == -1)[0] + 1
        if v[0]:
            starts = np.concatenate([[0], starts])
        if v[-1]:
            ends   = np.concatenate([ends, [len(v)]])

        for s, e in zip(starts, ends):
            if e - s < 3:
                continue
            tau_s = element_size * (t_px[s] - tau_off_px)
            tau_e = element_size * (t_px[e - 1] - tau_off_px)
            x_s = off_mm * cp + tau_s * ct
            y_s = off_mm * sp + tau_s * st
            x_e = off_mm * cp + tau_e * ct
            y_e = off_mm * sp + tau_e * st
            seg_len = abs(tau_e - tau_s)

            touches_edge = (s == 0 or not in_b[s - 1] or
                            e == len(v) or not in_b[e])

            is_short = mcl > 0 and seg_len < mcl
            total_segs += 1

            if touches_edge or not is_short:
                color = 'green'
                lw = 0.25
                alpha = 0.35
            else:
                color = 'red'
                lw = 0.5
                alpha = 0.7
                short_non_edge += 1

            ax.plot([x_s, x_e], [y_s, y_e], color=color, linewidth=lw, alpha=alpha)

    return total_segs, short_non_edge


def plot_mcl_sweep_volume_comparison(
    mcl_values: list[float],
    directional_deltas: dict[str, list[float]],
    isotropic_deltas: dict[str, list[float]],
    save_path: str = "mcl_sweep_volume_comparison.png",
    panel_size: float = None,
):
    """Plot volume delta (% vs unfiltered SST) for directional vs isotropic SE.

    Side-by-side subplots per method.  Solid = directional, dashed = isotropic.
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    method_colors = {"close_only": "#1a936f", "open_only": "#457b9d", "close_open": "#e76f51", "open_close": "#7b2d8e"}
    method_markers = {"close_only": "o", "open_only": "v", "close_open": "s", "open_close": "D"}
    default_color = "#2c3e50"

    all_methods = ("close_only", "close_open", "open_only", "open_close")
    methods = [m for m in all_methods
               if m in directional_deltas or m in isotropic_deltas]
    if not methods:
        return

    layout = {"close_only": (0, 0), "close_open": (0, 1),
              "open_only": (1, 0), "open_close": (1, 1)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for method in methods:
        row, col = layout[method]
        ax = axes[row][col]
        color = method_colors.get(method, default_color)
        m = method_markers.get(method, "o")
        se_types = [
            ("isotropic", isotropic_deltas.get(method, [])),
            ("directional", directional_deltas.get(method, [])),
        ]
        for se_label, deltas in se_types:
            if not deltas:
                continue
            ls = "--" if se_label == "isotropic" else "-"
            ff = "none" if se_label == "isotropic" else color
            ax.plot(
                x, deltas,
                marker=m, color=color,
                linestyle=ls, linewidth=2, markersize=6,
                markerfacecolor=ff, markeredgewidth=1.5,
                label=se_label,
            )

        ax.set_title(method.replace("_", "-"))
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Volume Delta vs Unfiltered SST (%)", fontsize=11)
        ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.grid(True, linestyle=":", alpha=0.3)
    
        ax.legend(loc="best", fontsize=9)

    for method in all_methods:
        if method not in methods:
            row, col = layout[method]
            axes[row][col].set_visible(False)

    fig.suptitle("Volume Delta: Directional vs Isotropic Filtering", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_ordering_comparison(
    mcl_values: list[float],
    close_open_results: dict[str, list[float]],
    open_close_results: dict[str, list[float]],
    pct_sst_vs_opt: float,
    pct_sst_vs_baseline: float,
    save_path: str = "mcl_sweep_ordering_comparison.png",
    panel_size: float = None,
):
    """Compare close-open vs open-close morphological ordering.

    Dual y-axis: left = vs theoretical optimum, right = vs baseline.
    Uses the currently selected settings (directional/isotropic, guide stack).
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    curves = [
        ("close-open", close_open_results, "#e76f51", "s"),
        ("open-close", open_close_results, "#7b2d8e", "D"),
    ]

    for label, results, color, marker in curves:
        ax1.plot(
            x, results["pct_vs_opt"],
            marker=marker, color=color, linestyle="-", linewidth=2,
            markersize=6, label=label,
        )
        ax2.plot(
            x, results["pct_vs_baseline"],
            marker=marker, color=color, linestyle="--", linewidth=2,
            markersize=6,
        )

    ax1.axhline(
        y=pct_sst_vs_opt, color="#444444", linestyle=":", linewidth=1.5,
        alpha=0.8, label=f"SST-only vs Guide Opt ({pct_sst_vs_opt:+.1f}%)",
    )
    ax2.axhline(
        y=pct_sst_vs_baseline, color="#666666", linestyle=":", linewidth=1.5,
        alpha=0.8,
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.3)


    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_volume_ordering(
    mcl_values: list[float],
    close_open_deltas: list[float],
    open_close_deltas: list[float],
    save_path: str = "mcl_sweep_volume_ordering.png",
    panel_size: float = None,
):
    """Side-by-side volume delta for close-open vs open-close ordering."""
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    fig, (ax_co, ax_oc) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, deltas, label, color, marker in [
        (ax_co, close_open_deltas, "close-open", "#e76f51", "s"),
        (ax_oc, open_close_deltas, "open-close", "#7b2d8e", "D"),
    ]:
        ax.plot(
            x, deltas,
            marker=marker, color=color, linestyle="-", linewidth=2,
            markersize=6, label=label,
        )
        ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Volume Delta vs Unfiltered SST (%)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.3)
    
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Volume Delta: close-open vs open-close Ordering", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_mcl_sweep_volume_all(
    mcl_values: list[float],
    method_vol_deltas: dict[str, list[float]],
    save_path: str = "mcl_sweep_volume_all.png",
    panel_size: float = None,
):
    """All four filter methods on the same axes for volume delta comparison."""
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    colors = {"close_only": "#1a936f", "open_only": "#457b9d",
              "close_open": "#e76f51", "open_close": "#7b2d8e"}
    markers = {"close_only": "o", "open_only": "v",
               "close_open": "s", "open_close": "D"}

    fig, ax = plt.subplots(figsize=(11, 6))

    for method, deltas in method_vol_deltas.items():
        ax.plot(
            x, deltas,
            marker=markers.get(method, "o"),
            color=colors.get(method, "#2c3e50"),
            linestyle="-", linewidth=2, markersize=6,
            label=method.replace("_", "-"),
        )

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Volume Delta vs Unfiltered SST (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_ply_shape_differences(
    per_angle_zone_grids: dict[float, np.ndarray],
    zone_grid_combined: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    n_zones: int,
    save_path: str = "ply_shape_differences.png",
    stiffener_grid_mask: np.ndarray | None = None,
    stiffener_angle_grid: np.ndarray | None = None,
    stack_order: list[float] | None = None,
):
    """Show per-threshold ply shape differences between angles.

    For each threshold level, creates a subplot showing all angle masks
    overlaid with distinct colours.  Pixels where angles disagree are
    highlighted.  The combined (max) mask is drawn as a black outline.

    Colour mapping is **global** — the same angle combination gets the same
    colour across all threshold subplots, ensuring visual consistency.

    Args:
        per_angle_zone_grids:  ``{angle: zone_grid}`` from decoupled filter
                               (already ±angle-balanced)
        zone_grid_combined:    max-expansion combined zone grid
        ux, uy:                physical coordinates
        n_zones:               number of zones
        save_path:             output PNG path
        stiffener_grid_mask:   boolean mask of stiffener-base cells
        stiffener_angle_grid:  local stiffener fibre direction per cell (deg)
        stack_order:           ply angles in guide-stack order (outer to
                               mid-plane); legend labels follow it. Defaults
                               to descending angle when not given.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    import colorsys

    def _cell_edges(coords: np.ndarray) -> np.ndarray:
        if len(coords) == 1:
            d = 0.5
            return np.array([coords[0] - d, coords[0] + d])
        mids = 0.5 * (coords[:-1] + coords[1:])
        left = coords[0] - 0.5 * (coords[1] - coords[0])
        right = coords[-1] + 0.5 * (coords[-1] - coords[-2])
        return np.concatenate([[left], mids, [right]])

    def _angle_diff(a: float, b: float) -> float:
        return abs(((a - b + 90.0) % 180.0) - 90.0)

    def _distinct_combo_colors(n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 4))
        colors = []
        for i in range(n):
            # Golden-angle hue spacing with alternating saturation/value gives
            # stronger separation than tab20 for dense categorical legends.
            h = (0.618033988749895 * i) % 1.0
            s = 0.85 if i % 2 == 0 else 0.65
            v = 0.95 if (i % 3) != 1 else 0.78
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((r, g, b, 1.0))
        return np.asarray(colors)

    per_angle_zone_grids = clamp_per_angle_zone_grids(
        per_angle_zone_grids, zone_grid_combined)
    angles = sorted(per_angle_zone_grids.keys())
    n_angles = len(angles)
    # label order: guide-stack sequence when known, else outer-to-inner by
    # descending angle (which is the 90/45/0/-45 chain used in this work)
    _order = [a for a in (stack_order or []) if a in angles]
    _order += [a for a in sorted(angles, reverse=True) if a not in _order]
    valid = (zone_grid_combined >= 0)
    x_edges = _cell_edges(ux)
    y_edges = _cell_edges(uy)

    n_thresholds = n_zones - 1
    if n_thresholds < 1:
        return

    # --- Pass 1: compute global bitmask and find ALL unique combinations ---
    all_bitmasks = []
    global_unique = set()
    for t in range(1, n_zones):
        combined_mask = (zone_grid_combined >= t) & valid
        bitmask = np.zeros_like(combined_mask, dtype=int)
        for bit, angle in enumerate(angles):
            ag = per_angle_zone_grids[angle]
            bitmask |= ((ag >= t) & (ag >= 0)).astype(int) << bit
        bitmask[~combined_mask] = 0
        all_bitmasks.append(bitmask)
        if combined_mask.any():
            global_unique.update(np.unique(bitmask[combined_mask]).tolist())

    # Sort: all-angles-present first (most common), then by number of angles desc
    global_unique = sorted(global_unique,
                           key=lambda bm: (-bin(int(bm)).count('1'), bm))
    n_global = len(global_unique)
    global_colors = _distinct_combo_colors(max(n_global, 1))
    bm_to_color_idx = {bm: i for i, bm in enumerate(global_unique)}
    combo_cmap = mcolors.ListedColormap(global_colors)
    combo_norm = mcolors.BoundaryNorm(np.arange(n_global + 1) - 0.5, max(n_global, 1))

    # --- Pass 2: plot each threshold with consistent colours ---
    # Size both rows to what they actually need. The plot axes draw the panel
    # at equal aspect, so a fixed tall row leaves a wide band of dead space
    # under a wide panel; the legend row only needs the rows its entries fill.
    _pw = 7.0
    _dx = float(x_edges[-1] - x_edges[0]) or 1.0
    _dy = float(y_edges[-1] - y_edges[0]) or 1.0
    _plot_h = _pw * (_dy / _dx) + 1.0          # + title, ticks and x-label
    _max_entries = 1
    for _ti, _t in enumerate(range(1, n_zones)):
        _cm = (zone_grid_combined >= _t) & valid
        if _cm.any():
            _max_entries = max(_max_entries,
                               len(np.unique(all_bitmasks[_ti][_cm])))
    _lc = _max_entries if _max_entries <= 4 else 4 if _max_entries <= 8 else 3
    _leg_h = 0.30 * int(np.ceil(_max_entries / _lc)) + 0.45
    fig, axes = plt.subplots(
        2,
        n_thresholds,
        figsize=(_pw * n_thresholds, _plot_h + _leg_h),
        gridspec_kw={"height_ratios": [_plot_h, _leg_h]},
    )
    if n_thresholds == 1:
        axes = np.asarray(axes).reshape(2, 1)

    plot_axes = axes[0]
    legend_axes = axes[1]

    for t_idx, t in enumerate(range(1, n_zones)):
        ax = plot_axes[t_idx]
        leg_ax = legend_axes[t_idx]
        bitmask = all_bitmasks[t_idx]
        combined_mask = (zone_grid_combined >= t) & valid

        # Find which global combos are present at this threshold
        if combined_mask.any():
            present_bms = [bm for bm in global_unique
                           if ((bitmask == bm) & combined_mask).any()]
        else:
            present_bms = []

        display = np.full_like(bitmask, -1, dtype=float)
        for bm in present_bms:
            ci = bm_to_color_idx[bm]
            display[(bitmask == bm) & combined_mask] = ci
        display[~combined_mask] = np.nan

        if present_bms:
            ax.pcolormesh(
                x_edges,
                y_edges,
                display,
                cmap=combo_cmap,
                norm=combo_norm,
                shading='flat',
            )

        # Draw combined mask outline
        ax.contour(
            ux,
            uy,
            combined_mask.astype(float),
            levels=[0.5], colors='black', linewidths=1.5, alpha=0.7,
        )

        # Legend: only combinations present at this threshold, using global colours
        legend_elements = []
        for bm in present_bms:
            ci = bm_to_color_idx[bm]
            present_angles = [angles[bit] for bit in range(n_angles) if bm & (1 << bit)]
            _seq = [a for a in _order if a in present_angles]
            angle_strs = [f"{a:.0f}" for a in _seq]
            n_pres = int(((bitmask == bm) & combined_mask).sum())
            pct = 100.0 * n_pres / max(int(combined_mask.sum()), 1)
            label = rf"${'/'.join(angle_strs)}^\circ$ ({pct:.1f}%)"

            legend_elements.append(
                Patch(facecolor=global_colors[ci], edgecolor='k', label=label))

        n_disagree = int(sum(
            ((bitmask == bm) & combined_mask).sum()
            for bm in present_bms
            if bin(int(bm)).count('1') < n_angles
        ))
        n_total = int(combined_mask.sum())
        pct_disagree = 100.0 * n_disagree / n_total if n_total > 0 else 0

        ax.set_title(f"Final threshold \u2265 {t}  "
                     f"({n_disagree}/{n_total} px decoupled, {pct_disagree:.1f}%)",
                     fontsize=11)
        ax.set_xlabel("x (mm)", fontsize=10)
        ax.set_ylabel("y (mm)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        leg_ax.axis("off")
        if legend_elements:
            # keep it on as few rows as fit: the legend has its own axes, so
            # it can never overlap the plot, but wide rows shrink the text
            _n = len(legend_elements)
            _ncol = _n if _n <= 4 else 4 if _n <= 8 else 3
            leg_ax.legend(
                handles=legend_elements,
                loc="upper center",
                fontsize=9 if _n <= 8 else 8,
                framealpha=1.0,
                edgecolor="0.6",
                ncol=_ncol,
                handlelength=1.1,
                handleheight=1.1,
                handletextpad=0.5,
                columnspacing=1.3,
                borderpad=0.5,
            )

    fig.suptitle("Per-Angle Ply Shape Differences at Each Threshold", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_single_design_sweep(
    mcl_values: list[float],
    pct_vs_opt: list[float],
    pct_vs_baseline: list[float],
    sst_perf_vs_opt: float,
    sst_perf_vs_baseline: float,
    method: str,
    directional: bool,
    panel_size: float,
    save_path: str = "mcl_sweep_single_design.png",
):
    """Single-design performance vs MCL plot (one curve + SST baseline).

    Same twin-axis format as *plot_mcl_sweep* but shows only the current
    design's curve plus a horizontal SST reference line.
    """
    import matplotlib.pyplot as plt

    x = _mcl_to_pct(mcl_values, panel_size) if panel_size else mcl_values
    xlabel = "Minimum Cut Length (% of panel size)" if panel_size else "Minimum Cut Length (mm)"

    label = method.replace("_", "-")
    if directional:
        label += " (directional)"

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.set_xticks(np.linspace(x[0], x[-1], len(x)))
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Performance vs Theoretical Optimum (%)", fontsize=11)
    ax1.plot(x, pct_vs_opt, "o-", linewidth=2, markersize=6,
             label=label, color="black")
    ax1.axhline(y=sst_perf_vs_opt, color="#e74c3c", linestyle="--",
                linewidth=1.5, alpha=0.8, label=f"SST-only (pre-filter, {sst_perf_vs_opt:+.1f}% vs Opt)")

    ax2 = ax1.twinx()
    ax2.plot(x, pct_vs_baseline, "o-", linewidth=2, markersize=6,
             label=label)
    ax2.axhline(y=sst_perf_vs_baseline, color="#e74c3c", linestyle="--",
                linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Performance vs Baseline (%)", fontsize=11)

    ax1.grid(True, linestyle=":", alpha=0.3)
    ax1.legend(loc="best", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")
