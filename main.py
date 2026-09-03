from kmeans import (
    k_means1D,
    plot_clustered_thickness,
    plot_threshold_masks,
    plot_elbow,
    DesignProblem,
)
from SST import (
    build_sst,
    assign_zones,
    assign_zones_volume_matched,
    assign_zones_perply,
    assign_zones_direct,
    plot_sst_zones,
    plot_sst_zones_scaled,
    plot_sst_zones_vc_evolution,
    plot_sst_table,
    plot_sst_side_view,
)
from helper import (
    _cleanup_junk,
    run_solver,
    export_fem,
    export_fem_fso_exact,
    read_results,
    read_volfrac_from_out,
    get_volume_fraction,
    set_volume_fraction,
    _archive_results,
    read_totalthickness_from_out,
    _mcl_tag,
    _mcl_values,
    setup_run_log,
    detect_stiffener_elements,
    parse_initial_plies,
    detect_smear,
    symmetrize_smear_plies,
    expand_smear_thicknesses,
    parse_ply_thicknesses,
    load_mesh_data,
    clean_optistruct_float,
    parse_total_element_thickness,
    OPTISTRUCT_EXE,
    _apply_lamthk_to_text,
    _render_fem_with_volume_fraction,
    _read_text_if_exists,
    _can_reuse_fso_job,
                    )
from collections import Counter
import numpy as np
import os
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── Log console output to file ─────────────────────────────────────────────────
setup_run_log()
# ── File paths ────────────────────────────────────────────────────────────────
FILE = "bulkhead_simple"    #  "bulkhead_compress"  "bulkhead_asymm2s"  "bulkhead_simple"  "MMConly2mm"   "MMConly20mm"
FEM_FILE = FILE+".fem"
SH_FILE  = FILE+".sh"
OUT_FILE = FILE+".out"
# ── Settings ──────────────────────────────────────────────────────────────────
VERBOSE               = True   # per-step detail, or summary only

# target skin volume fraction of the initial flat-panel skin volume; the DCONSTR
# VF is adjusted stiffener-aware to hit it, since the FSO only sizes the skin
INPUT_VF               = 0.6

# minimum ply thickness (mm); the input ply is halved while it stays above this
MIN_PLY_THICKNESS      = 0.125
DOUBLING_FACTOR        = None    # ply split factor (None = auto)
# zone count comes from the Dn-rule; the elbow analysis may only raise it
AUTO_NZONES            = True
IMPROVEMENT_THRESHOLD = 0.05     # min relative gain for an extra zone
ELBOW_MAX_ZONES        = 12      # largest k tested by the elbow sweep
COVERING_PLIES         = 1       # surface half-plies protected from dropping
MAX_CONTIGUITY         = 2       # max consecutive same-angle plies
MAX_DISORIENTATION     = 45.0    # max angle step between consecutive plies (deg)
POSITION_BIAS          = 0.05    # importance boost for outer plies

# keep the ladder with the fewest infeasible intermediate counts rather than
# requiring every count feasible
SST_TOLERANT_LADDER    = True

# guideline 14: max ply-count jump between adjacent zones. Drives the zone
# count, zones are added until it passes. 0/None = off
DN_RULE                 = 8

# minimum laminate thickness for the FSO to limit nr. of zones due to dN-rule
SET_LAMTHK              = True
MIN_LAMTHK_MARGIN       = 0.75
# compensation keeps the first run's LAMTHK, so this caps how far the corrected
# VF may travel: the floor must stay below LAMTHK_KEEP_RATIO x (VF x t0), or the
# FSO has no room to redistribute and collapses to a flat panel
LAMTHK_KEEP_RATIO       = 0.90

# "global" = mean retention over all elements, "zone_aware" = per-zone weighted
IMPORTANCE_METHOD      = "zone_aware"

MIN_CUT_LENGTH = 90.0    # minimum cut length for AFP (mm); 0/None = no filtering

# after filtering, delete any fibre still shorter than MIN_CUT_LENGTH
FINAL_TRIMMING          = True

# virtually extend the panel edge before filtering so edge features survive
EDGE_OVERSHOOT          = MIN_CUT_LENGTH / 2  # mm (0 = disabled)

# close_only  / open_only / close_open / open_close / dilate_only / erode_only
FILTERING_METHOD        = "close_open"

# SST ply splitting inflates volume: measure the ratio R, adjust the FSO VF and
# re-run. False = only report R and the suggested VF
VOLUME_COMPENSATE       = True
RminVC = 1.02        # min vol mismatch ratio that triggers compensation

FORCE_VF_RERUN         = False   # reuse cached FSO jobs where valid

# volume matched model during compensation: "sst_only", or "sst_filtered"
COMPENSATE_METHOD       = "sst_filtered"

# under FILTER_ANALYSIS these control the single run only; the sweep always
# runs every comparison
DIRECTIONAL_FILTERING   = True  # filter along fibre directions only
DIRECTIONAL_MODE        = "star"  # "star", or "clamped" (star + isotropic clip)
AFP_TAPE_WIDTH          = 1    # mm

OPTIMIZE_GUIDE_STACK    = True   # search sweep variants for bending stiffness

# zone ply count assigmnent: "volume_matched" (k-means + volume match),
# "perply" (FSO per-ply profile), "direct" (rank mapping, no volume matching)
ASSIGNMENT_METHOD      = "direct"

MIN_DISTINCT_ZONES     = None   # None: set to the derived zone count at Step 2

CLEAN_UP    = True  # delete junk files before the run

# convert an isotropic PSHELL skin to a PCOMP LAM=SMEAR on a CFRP MAT8 so the
# composite DfM can run; not used when the skin is already a composite PCOMP
CONVERT_SKIN_TO_CFRP = True

FILTER_ANALYSIS = False # sweep MCL and plot performance
FILTER_ANALYSIS_ALL = False # all filter types, or only FILTERING_METHOD
FILTER_ANALYSIS_RERUN = False  # archive old sweep results and re-run
FILTER_ANALYSIS_MIN   = 0.0    # mm
FILTER_ANALYSIS_MAX   = 40.0   # mm
FILTER_ANALYSIS_STEP  = 2.0    # mm
FILTER_ANALYSIS_PREFIX = "filter_analysis"

OPTISTRUCT_AVAILABLE = os.path.exists(OPTISTRUCT_EXE)

import filter as _filter_module
_filter_module.set_verbose(VERBOSE)
import SST as _sst_module
_sst_module.set_verbose(VERBOSE)
import helper as _helper_module
_helper_module.set_verbose(VERBOSE)

if CLEAN_UP:
    print("\n" + "=" * 64)
    print("  CLEANUP")
    print("=" * 64)
    _cleanup_junk()

# ── Flat-side stiffener conversion ────────────────────────────────────────────
# the FSO runs on the same model used for exports and analyses; conversions
# below are regenerated only when the source is newer, so FSO reuse still works
FSO_JOB = FILE
FSO_FEM_FILE = FEM_FILE

# 0/45/-45/90 of equal thickness summing to the original skin thickness; the
# DSIZE is left alone, it must still reference the skin PID
if CONVERT_SKIN_TO_CFRP:
    from helper import convert_noncomposite_skin_to_composite
    _comp_fem = FILE + "_comp.fem"
    if convert_noncomposite_skin_to_composite(FEM_FILE, _comp_fem, verbose=True):
        FILE = FILE + "_comp"
        FEM_FILE = FILE + ".fem"
        FSO_JOB = FILE
        FSO_FEM_FILE = FEM_FILE
        SH_FILE = FILE + ".sh"
        OUT_FILE = FILE + ".out"

# Stiffeners are modelled as coincident shells on the shared nodes, so the whole
# skin stays designable and the variable thickness grows on the opposite side.
from helper import convert_stiffeners_to_flat_side
_flat_fem = FILE + "_flat.fem"
if (not os.path.exists(_flat_fem)
        or os.path.getmtime(FEM_FILE) > os.path.getmtime(_flat_fem)):
    print(f"\n  Converting {FEM_FILE} to flat-side stiffeners...")
    convert_stiffeners_to_flat_side(FEM_FILE, _flat_fem,
                                    include_stiffeners=True, verbose=True)
else:
    print(f"\n  Reusing flat-side model: {_flat_fem}")
FILE = FILE + "_flat"
FEM_FILE = FILE + ".fem"
FSO_JOB = FILE
FSO_FEM_FILE = FEM_FILE
SH_FILE = FILE + ".sh"
OUT_FILE = FILE + ".out"

FSO_VF = INPUT_VF

# ── Set VF and run FSO ────────────────────────────────────────────────────────
# The FSO runs on FSO_FEM_FILE (== FEM_FILE except in flat-side mode, where
# it is the bare panel without stiffener stiffness).
_existing_vf = get_volume_fraction(FSO_FEM_FILE)
_sh_exists = os.path.exists(SH_FILE)
_out_exists = os.path.exists(OUT_FILE)
_need_fso_run = FORCE_VF_RERUN or not (_sh_exists and _out_exists)

# ── Manufacturing ply thickness by repeated halving ───────────────────────────
# MIN_PLY_THICKNESS is an absolute floor, not a target: halve the input laminate
# ply while the result stays >= the floor.
_in_plies_raw = parse_initial_plies(FSO_FEM_FILE)
if detect_smear(FSO_FEM_FILE):
    # SMEAR lists the half-stack; symmetrizing halves each ply thickness, which
    # is the base the SST actually expands from.
    _in_plies_raw = symmetrize_smear_plies(_in_plies_raw)
_skin_t_in = sum(t for _, t in _in_plies_raw)
_n_unique_in = len(set(a for a, _ in _in_plies_raw))
_base_t_in = _in_plies_raw[0][1]

_n_halvings = 0
_ply_t = _base_t_in
while _ply_t / 2.0 >= MIN_PLY_THICKNESS - 1e-9:
    _ply_t /= 2.0
    _n_halvings += 1
_DF_AUTO = 2 ** _n_halvings          # linear split factor used by build_sst

print(f"\n  Manufacturing ply thickness: {_base_t_in:g} mm halved "
      f"{_n_halvings}x -> {_ply_t:.5f} mm "
      f"(floor {MIN_PLY_THICKNESS:g} mm, split factor {_DF_AUTO}x)")
if _ply_t < MIN_PLY_THICKNESS - 1e-9:
    print(f"    WARNING: the input ply ({_base_t_in:g} mm) is already thinner "
          f"than the {MIN_PLY_THICKNESS:g} mm floor; using it unsplit.")

# This is the ply thickness used from here on.
MIN_PLY_THICKNESS = _ply_t
if DOUBLING_FACTOR is None:
    DOUBLING_FACTOR = _DF_AUTO

# ── Minimum laminate thickness for the FSO ────────────────────────────────────
if SET_LAMTHK:
    from helper import get_min_laminate_thickness, set_min_laminate_thickness
    _t_min_sym = _n_unique_in * MIN_PLY_THICKNESS * 2
    _t_min_margin = MIN_LAMTHK_MARGIN * FSO_VF * _skin_t_in
    _lamthk_target = max(_t_min_sym, _t_min_margin)
    _lamthk_now = get_min_laminate_thickness(FSO_FEM_FILE)

    print(f"\n  LAMTHK: skin {_skin_t_in:g} mm, {_n_unique_in} unique angles, "
          f"ply {MIN_PLY_THICKNESS:.5f} mm, VF {FSO_VF:.4f}")
    print(f"    min symmetric laminate : {_t_min_sym:.4f} mm "
          f"({_n_unique_in} angles x {MIN_PLY_THICKNESS:.5f} x 2)")
    print(f"    VF margin              : {_t_min_margin:.4f} mm "
          f"({MIN_LAMTHK_MARGIN:g} x {FSO_VF:.4f} x {_skin_t_in:g})")
    print(f"    -> LAMTHK              : {_lamthk_target:.4f} mm")

    # LAMTHK forces every element to at least this thickness, so it puts a hard
    # floor on the reachable volume fraction.
    _vf_floor_implied = _lamthk_target / _skin_t_in if _skin_t_in > 0 else 0.0
    if _vf_floor_implied > FSO_VF + 1e-9:
        print(f"    WARNING: LAMTHK implies a minimum volume fraction of "
              f"{_vf_floor_implied:.3f}, above the requested VF="
              f"{FSO_VF:.3f}. The FSO cannot reach the target volume and will "
              f"pin at the floor (near-uniform panel). Lower "
              f"MIN_LAMTHK_MARGIN/MIN_PLY_THICKNESS, or raise INPUT_VF.")

    if _lamthk_now is None:
        print(f"    WARNING: no 'COMP LAMTHK' card found in {FSO_FEM_FILE}; "
              f"cannot set the laminate floor (add one to the DSIZE).")
    elif abs(_lamthk_now - _lamthk_target) >= 1e-6:
        set_min_laminate_thickness(FSO_FEM_FILE, _lamthk_target)
        print(f"    LAMTHK updated: {_lamthk_now:g} -> {_lamthk_target:.4f} mm")
        _need_fso_run = True
    else:
        print(f"    LAMTHK already {_lamthk_now:g} mm; unchanged.")

if _existing_vf is not None and abs(_existing_vf - FSO_VF) >= 1e-4:
    print(f"  FEM file VF = {_existing_vf} -> updating to adjusted FSO VF = "
          f"{FSO_VF:.4f} (INPUT_VF = {INPUT_VF:.4f})")
    set_volume_fraction(FSO_FEM_FILE, FSO_VF)
    _need_fso_run = True

# HyperMesh sometimes exports DESGLB inside the SUBCASE, which OptiStruct
# rejects (fatal ERROR 1039). Move it to the global case-control level.
from helper import ensure_desglb_global, read_optistruct_error, sanitize_pcomp_lam_flags
# Some HyperMesh exports write a garbled 'SYSMEAR' into a PCOMP GE field, which
# OptiStruct rejects (ERROR 1485). Clean it on the actual FSO input regardless
# of whether the skin-conversion path touched the card.
if sanitize_pcomp_lam_flags(FSO_FEM_FILE):
    print(f"  Cleaned malformed PCOMP LAM/GE flag(s) in {FSO_FEM_FILE}.")
    _need_fso_run = True
if ensure_desglb_global(FSO_FEM_FILE):
    print(f"  Relocated DESGLB above SUBCASE in {FSO_FEM_FILE} "
          f"(OptiStruct requires it at the global case-control level).")
    _need_fso_run = True

if _need_fso_run:
    if OPTISTRUCT_AVAILABLE:
        # Do not archive the source FSO outputs. Keep the canonical
        # 3mmstiffeners.sh/.out filenames and overwrite them in place.
        print(f"  Running Free-Size Optimization ({FSO_FEM_FILE}, VF={FSO_VF:.4f})...")
        run_solver(filename=FSO_FEM_FILE, optimize=True)
        # The FSO must produce a .sh (per-ply thickness history). If it didn't,
        # OptiStruct failed on the input — surface that clearly rather than
        # letting a downstream 'missing .sh' FileNotFoundError bury the cause.
        if not os.path.exists(SH_FILE):
            raise RuntimeError(
                f"Free-Size Optimization failed: no '{SH_FILE}' was produced. "
                f"OptiStruct error: {read_optistruct_error(OUT_FILE)}")
    else:
        raise FileNotFoundError(
            f"OptiStruct not available at {OPTISTRUCT_EXE}, and {SH_FILE} / "
            f"{OUT_FILE} are not available for reuse with FORCE_VF_RERUN={FORCE_VF_RERUN}.")
elif VERBOSE:
    print(f"  Reusing existing FSO results: {SH_FILE}, {OUT_FILE}")

BaselineCompliance, OptimalCompliance, _ = read_results(jobname=FSO_JOB, optimize=True)
# vol_frac is the VF the FSO actually ran at (seed for VOLUME_COMPENSATE's
# Newton iteration); INPUT_VF remains the user-facing whole-panel target.
vol_frac = FSO_VF

# ==============================================================================
# Step 1: Parse the initial layup and FSO results
# ==============================================================================
print("\n" + "=" * 64)
print("  STEP 1: Parsing FSO data")
print("=" * 64)

initial_plies_raw = parse_initial_plies(FEM_FILE)
is_smear = detect_smear(FEM_FILE)

if is_smear:
    if VERBOSE:
        print(f"\n  Detected PCOMP SMEAR format")
    smeared_angles = [a for a, _ in initial_plies_raw]
    initial_plies = symmetrize_smear_plies(initial_plies_raw)
    base_ply_thickness = initial_plies[0][1]
    if VERBOSE:
        print(f"  Symmetrized: {len(initial_plies_raw)} plies -> {len(initial_plies)} plies")
else:
    initial_plies = initial_plies_raw
    base_ply_thickness = initial_plies[0][1]

if VERBOSE:
    print(f"\n  Initial layup ({len(initial_plies)} plies, t_ply = {base_ply_thickness} mm):")
    for i, (a, t) in enumerate(initial_plies):
        print(f"    Ply {i+1}: {a:>+6.1f}deg, {t:.3f} mm")

elem_ply_data_raw = parse_ply_thicknesses(SH_FILE, base_ply_thickness)

if is_smear:
    elem_ply_data = expand_smear_thicknesses(
        elem_ply_data_raw, smeared_angles, base_ply_thickness,
    )
    n_raw = len(next(iter(elem_ply_data_raw.values())))
    n_exp = len(next(iter(elem_ply_data.values())))
    if VERBOSE:
        print(f"  Expanded per-ply data: {n_raw} -> {n_exp} values/element")
    thickness_data = {eid: sum(vals) for eid, vals in elem_ply_data.items()}    
else:
    elem_ply_data = elem_ply_data_raw
    thickness_data = parse_total_element_thickness(SH_FILE, base_ply_thickness)

X, Y, T = load_mesh_data(FEM_FILE, thickness_data)

# ── Detect the attached stiffener shells ────────────────────────────────────
# They are non-design elements, but the skin beneath them is ordinary design
# skin already covered by the FSO, so the DfM logic runs on a homogeneous
# domain and only the PIDs are kept, for the exporters to preserve verbatim.
design_pid, _stiffener_eids = detect_stiffener_elements(FEM_FILE)

flat_stiffener_pids = set()
if _stiffener_eids:
    with open(FEM_FILE, "r") as _f:
        for _line in _f:
            if _line.startswith("CQUAD4"):
                try:
                    if int(_line[8:16]) in _stiffener_eids:
                        flat_stiffener_pids.add(int(_line[16:24]))
                except (ValueError, IndexError):
                    pass
    print(f"\n  Flat-side stiffeners: {len(_stiffener_eids)} attached stiffener "
          f"shells ({len(flat_stiffener_pids)} PIDs) will be preserved "
          f"verbatim; whole skin is designable (no base-skin coupling).")

# ==============================================================================
# Step 2: Cluster total thickness into discrete zones
# ==============================================================================
print("\n" + "=" * 64)
print("  STEP 2: Clustering thickness into zones")
print("=" * 64)

# Zone count: driven by the Dn-rule, raised by the elbow if it wants more
_t_vals = np.array(list(thickness_data.values()), dtype=float)
_t_scale = 2.0 if is_smear else 1.0      # thickness_data is the half-stack under SMEAR
_t_max_fso = float(_t_vals.max()) * _t_scale
_t_min_fso = float(_t_vals.min()) * _t_scale

if DN_RULE:
    _nz_ratio = (_t_max_fso - _t_min_fso) / (MIN_PLY_THICKNESS * DN_RULE)
    _nz_guess = int(np.ceil(round(_nz_ratio, 4))) + 1
    print(f"\n  Dn-rule zone guess: FSO thickness {_t_min_fso:.4f}..{_t_max_fso:.4f} mm "
          f"({(_t_max_fso - _t_min_fso) / MIN_PLY_THICKNESS:.1f} plies of "
          f"{MIN_PLY_THICKNESS:.5f} mm) / DN_RULE {DN_RULE} -> {_nz_guess} zones")
else:
    _nz_guess = 2
    print(f"\n  Dn-rule disabled; starting from {_nz_guess} zones")

N_ZONES = max(2, _nz_guess)
MIN_DISTINCT_ZONES = N_ZONES

# ── Dn-rule (guideline 14) machinery ─────────────────────────────────────────
# Grid mapping for element adjacency over the designable skin (thickness_data).
if DN_RULE:
    from filter import build_grid_mapping as _dn_bgm
    _dn_eid_grid, _dn_e2g, _, _ = _dn_bgm(X, Y, list(thickness_data.keys()))


def _dn_rule_check(_nplies_map):
    _g = np.full(_dn_eid_grid.shape, np.nan)
    for _eid, _n in _nplies_map.items():
        if _eid in _dn_e2g:
            _iy, _ix = _dn_e2g[_eid]
            _g[_iy, _ix] = _n
    _d = np.concatenate([np.abs(np.diff(_g, axis=1)).ravel(),
                         np.abs(np.diff(_g, axis=0)).ravel()])
    _d = _d[~np.isnan(_d)]
    if _d.size == 0:
        return 0, 0
    return int(np.sum(_d > DN_RULE)), int(np.max(_d))


# Thinnest laminate the FSO actually produced, in plies. The SST ladder is
# clamped to this so no zone is thinner than anything in the design being
# discretised (the FSO can't go below its own LAMTHK floor).
_MIN_PLIES_FSO = max(2, int(round(_t_min_fso / MIN_PLY_THICKNESS)))
_MIN_PLIES_FSO -= _MIN_PLIES_FSO % 2          # symmetric laminates are even
print(f"  SST ladder floor: {_MIN_PLIES_FSO} plies "
      f"({_MIN_PLIES_FSO * MIN_PLY_THICKNESS:.4f} mm, FSO minimum "
      f"{_t_min_fso:.4f} mm)")


def _cluster_and_build(_nz, _verbose_build=True):
    _cd, _ct, _ = k_means1D(thickness_data, _nz)
    if not _verbose_build:
        _sst_module.set_verbose(False)
    try:
        _s = build_sst(
            initial_plies=initial_plies,
            elem_ply_thicknesses=elem_ply_data,
            n_zones=_nz,
            min_ply_thickness=MIN_PLY_THICKNESS,
            covering_plies=COVERING_PLIES,
            max_contiguity=MAX_CONTIGUITY,
            max_disorientation=MAX_DISORIENTATION,
            position_bias=POSITION_BIAS,
            importance_method=IMPORTANCE_METHOD,
            cluster_data=_cd,
            cluster_centers=_ct,
            doubling_factor_override=DOUBLING_FACTOR,
            optimize_guide_stack=OPTIMIZE_GUIDE_STACK,
            tolerant_drop_order=SST_TOLERANT_LADDER,
            min_plies=_MIN_PLIES_FSO,
            dn_rule=DN_RULE or 0,
        )
    finally:
        if not _verbose_build:
            _sst_module.set_verbose(VERBOSE)
    return _cd, _ct, _s


def _probe_dn(_sst_obj, _cd, _ct, _nz):
    """Worst adjacent-zone ply jump for the Step-4a assignment at *_nz* zones."""
    if ASSIGNMENT_METHOD == "perply":
        _skin = sum(thickness_data.values()) * (2 if is_smear else 1)
        _nplies, _, _ = assign_zones_perply(
            _sst_obj, elem_ply_data, [a for a, _ in initial_plies],
            target_total_thickness=_skin, min_distinct_zones=_nz)
    elif ASSIGNMENT_METHOD == "direct":
        _nplies, _, _ = assign_zones_direct(_sst_obj, _cd, _ct)
    else:
        _skin = sum(thickness_data.values()) * (2 if is_smear else 1)
        _nplies, _, _ = assign_zones_volume_matched(
            _sst_obj, _cd, _ct,
            target_total_thickness=_skin, min_distinct_zones=_nz)
    return _dn_rule_check(_nplies)


def _ladder_dn_check(_sst_obj):
    _levels = sorted(_sst_obj.zone_ply_counts)
    if len(_levels) < 2:
        return 0, 0
    _steps = [_levels[i + 1] - _levels[i] for i in range(len(_levels) - 1)]
    return int(sum(s > DN_RULE for s in _steps)), int(max(_steps))


print("\n" + "=" * 64)
print("  STEP 3: Building SST from FSO ply data")
print("=" * 64)

# Raise the zone count until consecutive ladder levels are within DN_RULE
cluster_data, centers, sst = _cluster_and_build(N_ZONES)

if DN_RULE:
    _dn_nviol, _dn_worst = _ladder_dn_check(sst)
    _prev_levels = len(set(sst.zone_ply_counts))
    _stagnant = 0
    while _dn_nviol > 0:
        if _stagnant >= 2:
            print(f"\n  WARNING: worst ladder step {_dn_worst} > {DN_RULE} and "
                  f"extra zones no longer add distinct levels ({_prev_levels} "
                  f"available from a {sst.n_max}-ply guide). Lower "
                  f"MIN_PLY_THICKNESS for a finer ladder, or raise DN_RULE.")
            break
        print(f"\n  Ladder Dn-rule violated (worst step {_dn_worst} > "
              f"{DN_RULE} at {_dn_nviol} level(s)). Zones {N_ZONES} -> "
              f"{N_ZONES + 1}.")
        N_ZONES += 1
        cluster_data, centers, sst = _cluster_and_build(N_ZONES,
                                                        _verbose_build=False)
        _dn_nviol, _dn_worst = _ladder_dn_check(sst)
        _now_levels = len(set(sst.zone_ply_counts))
        _stagnant = 0 if _now_levels > _prev_levels else _stagnant + 1
        _prev_levels = max(_prev_levels, _now_levels)
    if _dn_nviol == 0:
        print(f"\n  Ladder Dn-rule: PASS with {N_ZONES} zones "
              f"(worst consecutive-level step {_dn_worst} <= {DN_RULE}, "
              f"levels {sorted(sst.zone_ply_counts)})")
    # Spatial jumps between non-consecutive thickness zones are out of the Dn-rule's
    # enforcement scope here, but reported so that the limitation stays visible.
    _sp_nviol, _sp_worst = _probe_dn(sst, cluster_data, centers, N_ZONES)
    if _sp_nviol:
        print(f"  (spatial note: {_sp_nviol} element boundaries jump more than "
              f"{DN_RULE} plies, worst {_sp_worst}, where the FSO gradient "
              f"crosses several zone bands within one element)")

_NZ_DN = N_ZONES   # zones the Dn-rule needs

# ── Elbow analysis: only ever RAISES the zone count ──────────────────────────
if AUTO_NZONES:
    _elbow_max_k = max(ELBOW_MAX_ZONES, _NZ_DN)
    print(f"\n  Elbow analysis (testing K=1..{_elbow_max_k}):")
    _wcss_curve = [k_means1D(thickness_data, _k)[2]
                   for _k in range(1, _elbow_max_k + 1)]

    _improvements = [_wcss_curve[i] - _wcss_curve[i + 1]
                     for i in range(len(_wcss_curve) - 1)]
    _max_imp = max(_improvements) if _improvements else 0
    _threshold = IMPROVEMENT_THRESHOLD * _max_imp

    _elbow_nz = 2
    for i, imp in enumerate(_improvements):
        if imp >= _threshold:
            _elbow_nz = i + 2

    print(f"    {'K':>4} {'WCSS':>14} {'Improvement':>14} {'Norm':>8}")
    print(f"    {'-'*4} {'-'*14} {'-'*14} {'-'*8}")
    for _k in range(_elbow_max_k):
        _imp_str = f"{_improvements[_k - 1]:>14.2f}" if _k > 0 else f"{'—':>14}"
        _norm_str = (f"{_improvements[_k - 1] / _max_imp:>8.2f}"
                     if _k > 0 and _max_imp > 0 else f"{'—':>8}")
        _marker = " <-- elbow" if _k + 1 == _elbow_nz else ""
        print(f"    {_k + 1:>4} {_wcss_curve[_k]:>14.2f} {_imp_str} "
              f"{_norm_str}{_marker}")

    _elbow_png = plot_elbow(len(_wcss_curve), thickness_data,
                            wcss_curve=_wcss_curve, elbow_k=_elbow_nz)
    print(f"    Elbow plot saved: {_elbow_png}")

    if _elbow_nz > _NZ_DN:
        print(f"\n  Elbow wants more zones than the Dn-rule: "
              f"{_NZ_DN} -> {_elbow_nz}; rebuilding.")
        N_ZONES = _elbow_nz
        cluster_data, centers, sst = _cluster_and_build(N_ZONES)
        if DN_RULE:
            _e_nviol, _e_worst = _probe_dn(sst, cluster_data, centers, N_ZONES)
            print(f"  Dn-rule at {N_ZONES} zones: "
                  f"{'PASS' if _e_nviol == 0 else 'FAIL'} "
                  f"(worst jump {_e_worst} vs {DN_RULE})")
    else:
        print(f"\n  Elbow at {_elbow_nz} zones <= Dn-rule's {_NZ_DN}; ignored.")

print(f"\n  Zone count: {N_ZONES}")
if VERBOSE:
    for i, c in enumerate(sorted(centers)):
        print(f"    Zone {i+1}: {c:.4f}")

MIN_DISTINCT_ZONES = N_ZONES
_N_ZONES_MAX = N_ZONES

problem = DesignProblem(X, Y, T)
plot_clustered_thickness(problem, centers, save_path="clustered_thickness.png",
                         thickness_scale=(2 if is_smear else 1))
# One black/white plot per binary threshold the morphological filter decomposes
# the k zones into (k-1 thresholds), same format as clustered_thickness.
_thr_pngs = plot_threshold_masks(problem, centers, save_prefix="filter_threshold",
                                 thickness_scale=(2 if is_smear else 1))
if _thr_pngs:
    print(f"  Saved {len(_thr_pngs)} filter-threshold masks: "
          f"{_thr_pngs[0]} .. {_thr_pngs[-1]}")

if VERBOSE:
    sst.summarize()

# ── Analytical volume estimate ─────────────────────────────────────────────
_mfg_t_ideal = base_ply_thickness / sst.doubling_factor
_mfg_t_actual = max(_mfg_t_ideal, MIN_PLY_THICKNESS)
_R_est_max = _mfg_t_actual / _mfg_t_ideal if _mfg_t_ideal > 0 else 1.0
print(f"\n  SST Volume Estimate:")
print(f"    Doubling factor:  {sst.doubling_factor}x")
print(f"    Mfg ply thickness: {sst.manufacturing_ply_thickness:.4f} mm "
      f"(ideal: {_mfg_t_ideal:.4f} mm)")
print(f"    Max volume ratio:  {_R_est_max:.2f}x "
      f"({'+' if _R_est_max > 1 else ''}{(_R_est_max-1)*100:.0f}% at thickest zone)")
print(f"    Zone ply counts:   {sst.zone_ply_counts}")

# Compute per-zone surviving angles for directional filtering
per_zone_angles = {}
for z in range(1, N_ZONES):
    n_plies = sst.zone_ply_counts[min(z, len(sst.zone_ply_counts) - 1)]
    lam = sst.get_laminate(n_plies)
    per_zone_angles[z] = sorted(set(lam))

# ==============================================================================
# Step 4: Assign elements to SST zones
# ==============================================================================
print("\n" + "=" * 64)
print("  STEP 4: Assigning elements to SST zones")
print("=" * 64)

from SST import assign_zones_volume_matched

eids = list(thickness_data.keys())
fso_skin_total = sum(thickness_data.values())
if is_smear:
    fso_skin_total *= 2
fso_total_thickness = fso_skin_total
print(f"\n  FSO total thickness (target): {fso_total_thickness:.2f} mm")

# ── 4a: SST-only assignment (no morphological filtering) ─────────────────────
print("\n" + "-" * 64)
print(f"  4a: SST-only assignment (method: {ASSIGNMENT_METHOD})")
print("-" * 64)

original_ply_angles = [a for a, _ in initial_plies]

if ASSIGNMENT_METHOD == "perply":
    elem_nplies_sst, best_assignment_sst, rel_error_sst = assign_zones_perply(
        sst,
        elem_ply_data,
        original_ply_angles,
        target_total_thickness=fso_skin_total,
        min_distinct_zones=MIN_DISTINCT_ZONES,
    )
elif ASSIGNMENT_METHOD == "direct":
    elem_nplies_sst, best_assignment_sst, rel_error_sst = assign_zones_direct(
        sst, cluster_data, centers,
    )
else:
    elem_nplies_sst, best_assignment_sst, rel_error_sst = assign_zones_volume_matched(
        sst, cluster_data, centers,
        target_total_thickness=fso_skin_total,
        zone_element_counts=None,
        min_distinct_zones=MIN_DISTINCT_ZONES,
    )

# Assign stiffener base skin to target zone
actual_total_sst = sum(
    elem_nplies_sst[eid] * sst.manufacturing_ply_thickness for eid in eids
)
final_rel_error_sst = (
    (actual_total_sst - fso_total_thickness) / fso_total_thickness * 100
)

print(f"\n  Best assignment: {best_assignment_sst}")
print(f"  Volume error vs FSO: {final_rel_error_sst:+.2f}%")
print(f"  Final total thickness: {actual_total_sst:.2f} mm")

if VERBOSE:
    zone_dist_sst = Counter(elem_nplies_sst.values())
    print(f"\n  Element distribution:")
    for n in sorted(zone_dist_sst.keys()):
        count = zone_dist_sst[n]
        t_mm = n * sst.manufacturing_ply_thickness
        pct = 100.0 * count / len(eids)
        print(f"    {n:>2} plies ({t_mm:.2f} mm): {count:>6} elements ({pct:.1f}%)")

# ── 4b: SST + morphological filtering assignment ─────────────────────────────
print("\n" + "-" * 64)
print("  4b: SST + filtering assignment")
print("-" * 64)

if MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0:
    from filter import (
        build_grid_mapping,
        zones_to_grid,
        apply_morphology_filter,
        plot_zone_grid_smoothed,
        print_steiner_verification,
    )

    # Step 1: Build the filtered zone grid from k-means clusters
    eid_grid, elem_to_grid, ux, uy = build_grid_mapping(X, Y, eids)
    element_size = float(np.median(np.diff(ux)))
    zone_grid = zones_to_grid(
        cluster_data, centers, elem_to_grid, eid_grid.shape
    )

    zone_grid_morph = zone_grid.copy()

    # Build importance grid (FSO thickness per element) for volume matching
    importance_grid = np.full(eid_grid.shape, 0.0)
    for eid, thick in thickness_data.items():
        if eid in elem_to_grid:
            iy, ix = elem_to_grid[eid]
            importance_grid[iy, ix] = thick

    per_angle_zone_grids = None
    if DIRECTIONAL_FILTERING and sst.ply_angles:
        from helper import parse_per_angle_thickness
        per_angle_thick = parse_per_angle_thickness(
            SH_FILE, sst.ply_angles, base_ply_thickness, len(eids),
        )
        per_angle_zone_grids = {}
        for ua, thick_map in per_angle_thick.items():
            ac, _ac_centers, _ = k_means1D(thick_map, N_ZONES)
            ag = zones_to_grid(ac, _ac_centers, elem_to_grid, eid_grid.shape)
            per_angle_zone_grids[ua] = ag
        if VERBOSE:
            print(f"  Built {len(per_angle_zone_grids)} per-angle zone grids")

    if VERBOSE:
        print_steiner_verification(zone_grid, MIN_CUT_LENGTH, element_size, N_ZONES)
    _per_angle_out = {}
    filtered_grid = apply_morphology_filter(
        zone_grid_morph, MIN_CUT_LENGTH, element_size, N_ZONES,
        method=FILTERING_METHOD,
        ply_angles=sst.ply_angles if DIRECTIONAL_FILTERING else None,
        tape_width=AFP_TAPE_WIDTH if DIRECTIONAL_FILTERING else None,
        directional_mode=DIRECTIONAL_MODE if DIRECTIONAL_FILTERING else None,
        per_zone_angles=per_zone_angles,
        per_angle_zone_grids=per_angle_zone_grids,
        run_fiber_enforcement=False,
        ux=ux, uy=uy,
        edge_overshoot=EDGE_OVERSHOOT,
        _out_per_angle=_per_angle_out if DIRECTIONAL_FILTERING else None,
        verbose=VERBOSE,
    )

    # Lock stiffener cells back to base zone — they act as anchors during
    # filtering (influencing neighbours) but must not change zone themselves

    if VERBOSE:
        print("\n  Pipeline: MCL filter")

    # Apply ±angle balance to per-angle grids BEFORE plotting and FEM export.
    # This ensures the plots and the .fem file show the same balanced shapes.
    if DIRECTIONAL_FILTERING and _per_angle_out:
        from filter import (
            balance_angle_grids,
            clamp_per_angle_zone_grids,
            combine_per_angle_zone_grids,
        )
        _per_angle_out = clamp_per_angle_zone_grids(_per_angle_out, filtered_grid)
        _per_angle_out = balance_angle_grids(_per_angle_out, verbose=VERBOSE)
        filtered_grid = combine_per_angle_zone_grids(
            _per_angle_out,
            valid_mask=(filtered_grid >= 0),
        )

    # Compute per-element half-ply presence for directional FEM export
    _elem_half_plies = None
    if DIRECTIONAL_FILTERING and _per_angle_out:
        from filter import compute_per_element_plies
        _elem_half_plies = compute_per_element_plies(
            _per_angle_out, eid_grid, sst, N_ZONES,
            balance_pairs=False,
            verbose=VERBOSE,
        )
        n_unique = len(set(_elem_half_plies.values()))
        if VERBOSE:
            print(f"\n  Per-angle FEM: {n_unique} unique ply-subset zones "
                  f"(from {len(_elem_half_plies)} elements)")

    # Step 2: Get filtered zone element counts directly from the filtered grid
    filtered_zone_counts = [int((filtered_grid == z).sum()) for z in range(N_ZONES)]
    original_zone_counts = [int((zone_grid == z).sum()) for z in range(N_ZONES)]

    if VERBOSE:
        print(f"\n  Zone volume comparison (incremental plies per zone level):")
        print(f"    Zone 0 is the base stack covering the entire panel.")
        print(f"    Higher zones show only their EXTRA plies beyond the previous zone.")
        print(f"    {'Zone':<8} {'Extra':>6} {'Area bef':>10} {'Area aft':>10} {'Delta':>10} {'Vol before':>12} {'Vol after':>12} {'Vol delta':>10}")
        print(f"    {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    sorted_counts_before = sorted(sst.zone_ply_counts)
    mfg_t = sst.manufacturing_ply_thickness
    n_valid = int((zone_grid >= 0).sum())

    cumulative_before = 0
    cumulative_after = 0
    total_vol_before = 0.0
    total_vol_after = 0.0

    for z in range(N_ZONES):
        n_plies_extra = sorted_counts_before[z] - (sorted_counts_before[z - 1] if z > 0 else 0)
        area_before = sum(original_zone_counts[zz] for zz in range(z, N_ZONES))
        area_after = sum(filtered_zone_counts[zz] for zz in range(z, N_ZONES))
        delta_area = area_after - area_before
        vol_before = n_plies_extra * area_before * mfg_t
        vol_after = n_plies_extra * area_after * mfg_t
        vol_delta = vol_after - vol_before
        total_vol_before += vol_before
        total_vol_after += vol_after
        if VERBOSE:
            print(f"    {z+1:<8} {n_plies_extra:>6} {area_before:>10} {area_after:>10} {delta_area:>+10} {vol_before:>12.2f} {vol_after:>12.2f} {vol_delta:>+10.2f}")

    vol_change_pct = (total_vol_after - total_vol_before) / total_vol_before * 100 if total_vol_before > 0 else 0.0
    if VERBOSE:
        print(f"    {'Total':<8} {'':>6} {'':>10} {'':>10} {'':>10} "
              f"{total_vol_before:>12.2f} {total_vol_after:>12.2f} {total_vol_after-total_vol_before:>+10.2f}")
        print(f"    Volume change: {vol_change_pct:+.2f}%")

    # Step 3: Direct rank mapping (same ply counts per zone as SST-only).
    # Volume scaling in Step 6 handles fair comparison.
    sorted_counts = sorted(sst.zone_ply_counts)

    # Step 4: Map each element to its ply count using the filtered grid.
    elem_nplies_filtered = {}
    for iy in range(filtered_grid.shape[0]):
        for ix in range(filtered_grid.shape[1]):
            eid = eid_grid[iy, ix]
            if eid < 0:
                continue
            zone = filtered_grid[iy, ix]
            zone = max(0, min(zone, len(sorted_counts) - 1))
            elem_nplies_filtered[eid] = sorted_counts[zone]

    best_assignment_filtered = tuple(sorted_counts)
else:
    print("  MIN_CUT_LENGTH is 0 or None — filtered model = SST-only model")
    elem_nplies_filtered = dict(elem_nplies_sst)
    best_assignment_filtered = best_assignment_sst
    # Filtering is what builds the per-element directional ply subsets; without
    # it the exporter falls back to the zone ply counts.
    _elem_half_plies = None

actual_total_filtered = sum(
    n * sst.manufacturing_ply_thickness for n in elem_nplies_filtered.values()
)
final_rel_error_filtered = (
    (actual_total_filtered - fso_total_thickness) / fso_total_thickness * 100
)


print(f"\n  Best assignment: {best_assignment_filtered}")
print(f"  Volume error vs FSO: {final_rel_error_filtered:+.2f}%")
print(f"  Final total thickness: {actual_total_filtered:.2f} mm")

if VERBOSE:
    zone_dist_filtered = Counter(elem_nplies_filtered.values())
    print(f"\n  Element distribution:")
    for n in sorted(zone_dist_filtered.keys()):
        count = zone_dist_filtered[n]
        t_mm = n * sst.manufacturing_ply_thickness
        pct = 100.0 * count / len(eids)
        print(f"    {n:>2} plies ({t_mm:.2f} mm): {count:>6} elements ({pct:.1f}%)")

# ==============================================================================
# VF Compensation
# ==============================================================================

# Save original reference values — these are ALWAYS used for downstream
# comparisons (GuidePly baseline, knockdown calculations), regardless of
# whether VF compensation changes the FSO results.
_orig_fso_total = fso_total_thickness
_orig_vol_frac = vol_frac
_orig_optimal = OptimalCompliance
_orig_baseline = BaselineCompliance

# Save pre-VC SST state for comparison plot
_pre_vc_elem_nplies = dict(elem_nplies_sst)
_pre_vc_sst_zpc = list(sst.zone_ply_counts)
_pre_vc_sst_mfg_t = sst.manufacturing_ply_thickness

_R_sst = actual_total_sst / _orig_fso_total if _orig_fso_total > 0 else 1.0
_R_filt = actual_total_filtered / _orig_fso_total if _orig_fso_total > 0 else 1.0

print(f"\n" + "-" * 64)
print(f"  Volume Compensation Summary")
print(f"-" * 64)
print(f"    Original FSO VF:      {_orig_vol_frac:.4f}")
print(f"    FSO total thickness:  {_orig_fso_total:.2f}")
print(f"    SST total thickness:  {actual_total_sst:.2f}")
print(f"    SST+filt thickness:   {actual_total_filtered:.2f}")
print(f"    Volume ratio R (SST): {_R_sst:.2f}x ({(_R_sst-1)*100:+.0f}%)")
print(f"    Volume ratio R (filt):{_R_filt:.2f}x ({(_R_filt-1)*100:+.0f}%)")

if abs(_R_sst - 1.0) > 0.1 and not VOLUME_COMPENSATE:
    _VF_suggested = _orig_vol_frac / _R_sst if _R_sst > 0 else _orig_vol_frac
    print(f"    Suggested VF: {_VF_suggested:.4f} (set VOLUME_COMPENSATE=True to auto-adjust)")

_comp_applied = False
_post_vc_VF = None

# ── Newton-like VF compensation ─────────────────────────────────────────────
# A perturbation run estimates dV/dVF, then interpolates the VF bringing the
# volume closest to target (N_ZONES locked to the elbow; two extra solver runs).
# Bidirectional: overshoot lowers VF, undershoot raises it (the MCL filter can
# add OR remove several percent). The trigger tests the ratio of the compensated
# quantity: for "sst_filtered" that's the filtered volume.
_R_gate = (_R_filt
           if (COMPENSATE_METHOD == "sst_filtered"
               and MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0)
           else _R_sst)

# +-2% tolerance band around the target volume (RminVC = 1.02).
_vc_band = RminVC - 1.0
if VOLUME_COMPENSATE and abs(_R_gate - 1.0) > _vc_band and OPTISTRUCT_AVAILABLE:

    _target = _orig_fso_total  # includes stiffener base
    _VF_0 = _orig_vol_frac

    # ── VF floor from the LAMTHK manufacturing constraint ────────────────
    # A VF request at/below min_lam_t/t0 is infeasible: the optimizer pins every
    # element at the floor and returns a uniform field, so k-means leaves empty
    # clusters and the SST is meaningless. Clamp probes above the floor.
    _min_lam_t = None
    with open(FSO_FEM_FILE, "r") as _f:
        for _line in _f:
            if "LAMTHK" in _line:
                _parts = _line.split()
                try:
                    _min_lam_t = float(_parts[_parts.index("LAMTHK") + 1])
                except (ValueError, IndexError):
                    pass
                break
    _t0_total = sum(t for _, t in initial_plies)
    _VF_floor = (_min_lam_t / _t0_total
                 if (_min_lam_t and _t0_total > 0) else None)
    # Every run in the compensation shares the first run's LAMTHK, so the floor
    # never moves under the secant. LAMTHK_KEEP_RATIO is therefore a bound on
    # how far the corrected VF may travel: below LAMTHK / (RATIO x t0) the mean
    # thickness crowds the fixed floor and the FSO collapses towards a flat
    # panel. Clamping there costs an undercorrection on the most extreme points
    # (which is reported honestly) instead of a mispredicted step.
    _VF_sym_floor = (_t_min_sym / _t0_total) if _t0_total > 0 else 0.0
    _VF_min_allowed = max(_lamthk_target / (LAMTHK_KEEP_RATIO * _skin_t_in),
                          _VF_sym_floor * 1.02, 0.01)
    _VF_max_allowed = 0.9999   # DCONSTR VF must stay below 1 (undershoot cap)
    if _VF_floor:
        print(f"    VF floor: LAMTHK held at {_lamthk_target:.4f} mm for every "
              f"run; VF clamped to >= {_VF_min_allowed:.4f} so the mean "
              f"thickness stays clear of it "
              f"(LAMTHK_KEEP_RATIO {LAMTHK_KEEP_RATIO:g})")

    def _run_and_build(vf_value, suffix):
        """Run FSO at *vf_value*, build SST (N_ZONES locked), return results."""
        _job = FILE + "_" + suffix
        _fem = _job + ".fem"
        _sh = _job + ".sh"
        _out = _job + ".out"

        # Template from the FSO model (bare panel in flat-side mode).
        _expected_fem = _render_fem_with_volume_fraction(FSO_FEM_FILE, vf_value)
        # The minimum laminate thickness is tied to the requested volume
        # fraction (section 3.6), so a probe run at a different VF must carry
        # its own LAMTHK. Holding the original run's value would impose a floor
        # that is too high for the probe, distorting the very thickness field
        # the sensitivity is measured from.
        if SET_LAMTHK:
            # One floor for the entire compensation, never rescaled. Rescaling
            # partway -- measuring the secant on one floor and applying the
            # correction on another -- put the two on different response
            # surfaces: a lower floor lets the FSO spread, which creates more
            # sub-MCL structure for the filter to fill back in, cancelling much
            # of the reduction the step was meant to deliver. Measured at
            # MCL 120: +14.25% residual with the rescale, +1.9% without, from an
            # identical probe and an identical measured sensitivity.
            # LAMTHK_KEEP_RATIO now bounds the corrected VF instead (below).
            _expected_fem = _apply_lamthk_to_text(_expected_fem, _lamthk_target)
        _reuse = (not FORCE_VF_RERUN) and _can_reuse_fso_job(_job, _expected_fem)
        if _reuse:
            print(f"    Reusing {suffix} FSO ({_fem}, VF={vf_value:.4f})...")
        else:
            if not OPTISTRUCT_AVAILABLE:
                raise FileNotFoundError(
                    f"OptiStruct not available at {OPTISTRUCT_EXE}, and {_job}.sh/.out "
                    f"cannot be reused with FORCE_VF_RERUN={FORCE_VF_RERUN}."
                )
            _archive_results(_job)
            with open(_fem, "w") as f:
                f.write(_expected_fem)
            print(f"    Running {suffix} FSO ({_fem}, VF={vf_value:.4f})...")
            run_solver(filename=_fem, optimize=True)

        _raw = parse_ply_thicknesses(_sh, base_ply_thickness)
        if is_smear:
            _data = expand_smear_thicknesses(
                _raw, smeared_angles, base_ply_thickness,
            )
            _thick = {eid: sum(v) for eid, v in _data.items()}
        else:
            _data = _raw
            _thick = parse_total_element_thickness(_sh, base_ply_thickness)

        _cl, _ce, _ = k_means1D(_thick, N_ZONES)
        # Rebuild the SST for this probe's field: the whole point of the
        # compensation is that a leaner FSO yields a leaner SST, so the laminate
        # is re-derived rather than reused. The ladder floor is recomputed from
        # THIS field, so a thinner probe can legitimately reach thinner
        # laminates instead of being clamped to the original run's minimum.
        _mp_p = int(round(min(_thick.values()) * (2 if is_smear else 1)
                          / (base_ply_thickness / DOUBLING_FACTOR)))
        _mp_p = max(2, _mp_p - (_mp_p % 2))
        _st = build_sst(
            initial_plies=initial_plies,
            elem_ply_thicknesses=_data,
            n_zones=N_ZONES,
            min_ply_thickness=MIN_PLY_THICKNESS,
            covering_plies=COVERING_PLIES,
            max_contiguity=MAX_CONTIGUITY,
            max_disorientation=MAX_DISORIENTATION,
            position_bias=POSITION_BIAS,
            importance_method=IMPORTANCE_METHOD,
            cluster_data=_cl,
            cluster_centers=_ce,
            doubling_factor_override=DOUBLING_FACTOR,
            optimize_guide_stack=OPTIMIZE_GUIDE_STACK,
            tolerant_drop_order=SST_TOLERANT_LADDER,
            min_plies=_mp_p,
            dn_rule=DN_RULE or 0,
        )
        # The candidate SST may land on a different DF / zone ladder than the
        # original, so its ACTUAL stiffener-base volume must be computed from
        # the candidate itself, not the outer (original-SST) value.
        _base_total_c = 0
        if ASSIGNMENT_METHOD == "direct":
            _assign, _, _ = assign_zones_direct(_st, _cl, _ce)
        else:
            _assign, _, _ = assign_zones_volume_matched(
                _st, _cl, _ce,
                target_total_thickness=_target - _base_total_c,
                min_distinct_zones=N_ZONES,
            )
        _vol = sum(_assign[eid] * _st.manufacturing_ply_thickness for eid in eids) + _base_total_c
        return {
            "sh": _sh, "data": _data, "thick": _thick,
            "cl": _cl, "ce": _ce, "sst": _st,
            "assign": _assign, "vol": _vol,
        }

    def _compute_filtered_volume(r):
        """Run the morphology filter on a compensation result dict and
        return the SST+filtered total volume."""
        _st = r["sst"]
        _cl = r["cl"]
        _ce = r["ce"]
        _thick = r["thick"]
        _sh = r["sh"]
        _data = r["data"]

        if not MIN_CUT_LENGTH or MIN_CUT_LENGTH <= 0:
            return r["vol"]

        _zg = zones_to_grid(_cl, _ce, elem_to_grid, eid_grid.shape)
        _szbi = len(_st.zone_ply_counts) - 1
        _zg_morph = _zg.copy()

        _ig = np.full(eid_grid.shape, 0.0)
        for _eid, _t in _thick.items():
            if _eid in elem_to_grid:
                _iy, _ix = elem_to_grid[_eid]
                _ig[_iy, _ix] = _t

        _pza = {}
        for _z in range(1, N_ZONES):
            _np = _st.zone_ply_counts[min(_z, len(_st.zone_ply_counts) - 1)]
            _lam = _st.get_laminate(_np)
            _pza[_z] = sorted(set(_lam))

        _pazg = None
        if DIRECTIONAL_FILTERING and _st.ply_angles:
            _pat = parse_per_angle_thickness(
                _sh, _st.ply_angles, base_ply_thickness, len(eids))
            _pazg = {}
            for _ua, _tm in _pat.items():
                _ac, _acc, _ = k_means1D(_tm, N_ZONES)
                _ag = zones_to_grid(_ac, _acc, elem_to_grid, eid_grid.shape)
                _pazg[_ua] = _ag

        _pao = {}
        _fg = apply_morphology_filter(
            _zg_morph, MIN_CUT_LENGTH, element_size, N_ZONES,
            method=FILTERING_METHOD,
            ply_angles=_st.ply_angles if DIRECTIONAL_FILTERING else None,
            tape_width=AFP_TAPE_WIDTH if DIRECTIONAL_FILTERING else None,
            directional_mode=DIRECTIONAL_MODE if DIRECTIONAL_FILTERING else None,
            per_zone_angles=_pza,
            per_angle_zone_grids=_pazg,
            run_fiber_enforcement=False,
            ux=ux, uy=uy,
            edge_overshoot=EDGE_OVERSHOOT,
            _out_per_angle=_pao if DIRECTIONAL_FILTERING else None,
            verbose=False,
        )

        _sc = sorted(_st.zone_ply_counts)
        _vol_f = 0.0
        for _iy in range(_fg.shape[0]):
            for _ix in range(_fg.shape[1]):
                _eid = eid_grid[_iy, _ix]
                if _eid < 0:
                    continue
                _zone = max(0, min(_fg[_iy, _ix], len(_sc) - 1))
                _vol_f += _sc[_zone] * _st.manufacturing_ply_thickness
        return _vol_f

    def _fso_degenerate(r):
        """True if the FSO probe collapsed to a near-uniform field (empty
        k-means clusters) — the signature of a VF at/below the LAMTHK floor."""
        _tv = list(r["thick"].values())
        if not _tv or (max(_tv) - min(_tv)) < 1e-6:
            return True
        _nonempty = sum(1 for _els in r["cl"].values() if len(_els) > 0)
        return _nonempty < N_ZONES

    _use_filtered = (COMPENSATE_METHOD == "sst_filtered"
                     and MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0)
    _vol_label = "filtered" if _use_filtered else "SST"

    if _use_filtered:
        _V_0 = actual_total_filtered
    else:
        _V_0 = actual_total_sst

    # Direction to compensate: overshoot -> lower VF, undershoot -> raise VF.
    _dir = -1.0 if _V_0 > _target else 1.0

    print(f"\n  Newton-like VF compensation (R={_R_gate:.2f}x, "
          f"method={_vol_label}, target={_target:.0f} mm, "
          f"{'overshoot' if _dir < 0 else 'undershoot'})")
    print(f"    Original: VF={_VF_0:.4f}, {_vol_label} vol={_V_0:.0f} mm")

    def _probe_vf(delta):
        """Perturbed VF one step toward the target, clamped to [floor, ceiling]."""
        if _dir < 0:
            return max(_VF_0 - delta, _VF_min_allowed)
        return min(_VF_0 + delta, _VF_max_allowed)

    def _more_extreme(a, b):
        """The probe further from VF_0 in the compensation direction."""
        if a is None:
            return b
        return min(a, b) if _dir < 0 else max(a, b)

    # ── Step 1+2: Perturbation run(s) + sensitivity, with adaptive step ──
    # A large VF step can move FSO features across the filter scale (MCL) so the
    # morphological filter adds/removes a different amount and the measured
    # sensitivity inverts (non-monotonic). Instead of giving up on the first
    # inverted probe, halve the step and re-probe closer to VF_0, where the
    # response is locally monotonic.
    _delta = max(0.05, _VF_0 * 0.10)
    _MAX_PERT_RETRIES = 3
    _dV_dVF = None
    _VF_failed = None   # most extreme probe (toward target) measured non-monotonic
    for _pert_try in range(_MAX_PERT_RETRIES + 1):
        _VF_pert = _probe_vf(_delta)
        print(f"    Perturbation: VF={_VF_pert:.4f} (delta={_delta:.4f})"
              + (f" [retry {_pert_try}]" if _pert_try else ""))
        _r_pert = _run_and_build(_VF_pert, "pert")
        if abs(_VF_pert - _VF_0) < 1e-9 or _fso_degenerate(_r_pert):
            # Probe pinned at the floor/ceiling (no gradient) or a degenerate
            # uniform field at/below the LAMTHK floor: treat as a failed probe.
            print(f"    Perturbed FSO unusable (pinned or degenerate at "
                  f"VF={_VF_pert:.4f}).")
            _VF_failed = _more_extreme(_VF_failed, _VF_pert)
            _dV_dVF = -1.0
            if _pert_try < _MAX_PERT_RETRIES:
                _delta *= 0.5
                print(f"    Retrying with halved perturbation "
                      f"(delta={_delta:.4f}).")
            continue
        if _use_filtered:
            _V_pert = _compute_filtered_volume(_r_pert)
        else:
            _V_pert = _r_pert["vol"]
        print(f"    Perturbed {_vol_label} vol: {_V_pert:.0f} mm")

        _dV_dVF = (_V_pert - _V_0) / (_VF_pert - _VF_0)
        print(f"    Sensitivity dV/dVF: {_dV_dVF:,.1f} mm/VF")

        if _dV_dVF > 0:
            break
        _VF_failed = _more_extreme(_VF_failed, _VF_pert)
        if _pert_try < _MAX_PERT_RETRIES:
            _delta *= 0.5
            print(f"    Non-monotonic (VF step did not move {_vol_label} vol "
                  f"toward target). Retrying with halved perturbation "
                  f"(delta={_delta:.4f}).")

    if _dV_dVF <= 0:
        print(f"    Non-monotonic (lower VF did not reduce {_vol_label} vol) "
              f"after {_MAX_PERT_RETRIES} halved retries. Skipping.")
    elif abs(_dV_dVF) < 1e-3:
        print(f"    Sensitivity too low. Skipping.")
    else:
        # ── Step 3: Interpolate target VF ────────────────────────────────
        _VF_target = _VF_0 + (_target - _V_0) / _dV_dVF
        _VF_raw = _VF_target
        _VF_target = max(_VF_min_allowed, min(_VF_target, _VF_max_allowed))
        if abs(_VF_target - _VF_raw) > 1e-9:
            print(f"    Newton target {_VF_raw:.4f} outside "
                  f"[{_VF_min_allowed:.4f}, {_VF_max_allowed:.4f}]; "
                  f"clamping to {_VF_target:.4f}.")

        # Do not extrapolate past a probe measured non-monotonic (its volume
        # moved the wrong way): cap midway between that probe and the last one.
        if _VF_failed is not None and _dir * (_VF_target - _VF_failed) >= 0:
            _VF_capped = 0.5 * (_VF_failed + _VF_pert)
            print(f"    Newton target {_VF_target:.4f} lies beyond the "
                  f"non-monotonic probe VF={_VF_failed:.4f}; "
                  f"capping to {_VF_capped:.4f}.")
            _VF_target = _VF_capped

        if abs(_VF_target - _VF_0) / _VF_0 < 0.02:
            print(f"    Predicted VF {_VF_target:.4f} ~ original. Skipping.")
        else:
            print(f"    Interpolated VF: {_VF_target:.4f}")

            # ── Step 4: Correction run ───────────────────────────────────
            _r_final = _run_and_build(_VF_target, "comp")
            if _fso_degenerate(_r_final):
                print(f"    Corrected FSO degenerate (uniform "
                      f"minimum-thickness field at VF={_VF_target:.4f}); "
                      f"compensation rejected, original {_vol_label} kept.")
            else:
                if _use_filtered:
                    _V_final = _compute_filtered_volume(_r_final)
                else:
                    _V_final = _r_final["vol"]
                _R_final = abs(_V_final - _target) / _target
                _R_orig = abs(_V_0 - _target) / _target
                print(f"    Corrected {_vol_label} vol: {_V_final:.0f} mm "
                      f"(error: {_R_final:.1%} vs original {_R_orig:.1%})")

                # ── Step 5: Pick best ─────────────────────────────────────
                if _R_final < _R_orig:
                    print(f"    Compensated {_vol_label} adopted.")
                    _comp_applied = True
                    _post_vc_VF = _VF_target
                else:
                    print(f"    Original {_vol_label} kept "
                          f"(compensated was worse).")

    # ── Apply compensated results to pipeline ─────────────────────────────
    if _comp_applied:
        print(f"\n  Applying compensated SST: N_ZONES={N_ZONES}, "
              f"DF={_r_final['sst'].doubling_factor}")

        elem_ply_data = _r_final["data"]
        thickness_data = _r_final["thick"]
        cluster_data = _r_final["cl"]
        centers = _r_final["ce"]
        sst = _r_final["sst"]
        elem_nplies_sst = _r_final["assign"]
        actual_total_sst = _r_final["vol"]
        DOUBLING_FACTOR = _r_final["sst"].doubling_factor

        # Re-compute per_zone_angles
        per_zone_angles = {}
        for z in range(1, N_ZONES):
            n_plies = sst.zone_ply_counts[min(z, len(sst.zone_ply_counts) - 1)]
            lam = sst.get_laminate(n_plies)
            per_zone_angles[z] = sorted(set(lam))

        final_rel_error_sst = (
            (actual_total_sst - _orig_fso_total) / _orig_fso_total * 100
        )

        # Re-run filtering with compensated configuration
        if MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0:
            zone_grid = zones_to_grid(
                cluster_data, centers, elem_to_grid, eid_grid.shape
            )
            zone_grid_morph = zone_grid.copy()

            importance_grid = np.full(eid_grid.shape, 0.0)
            for eid, thick in thickness_data.items():
                if eid in elem_to_grid:
                    iy, ix = elem_to_grid[eid]
                    importance_grid[iy, ix] = thick

            per_angle_zone_grids = None
            if DIRECTIONAL_FILTERING and sst.ply_angles:
                per_angle_thick = parse_per_angle_thickness(
                    _r_final["sh"], sst.ply_angles, base_ply_thickness, len(eids),
                )
                per_angle_zone_grids = {}
                for ua, thick_map in per_angle_thick.items():
                    ac, _ac_centers, _ = k_means1D(thick_map, N_ZONES)
                    ag = zones_to_grid(ac, _ac_centers, elem_to_grid, eid_grid.shape)
                    per_angle_zone_grids[ua] = ag

            _per_angle_out = {}
            filtered_grid = apply_morphology_filter(
                zone_grid_morph, MIN_CUT_LENGTH, element_size, N_ZONES,
                method=FILTERING_METHOD,
                ply_angles=sst.ply_angles if DIRECTIONAL_FILTERING else None,
                tape_width=AFP_TAPE_WIDTH if DIRECTIONAL_FILTERING else None,
                directional_mode=DIRECTIONAL_MODE if DIRECTIONAL_FILTERING else None,
                per_zone_angles=per_zone_angles,
                per_angle_zone_grids=per_angle_zone_grids,
                run_fiber_enforcement=False,
                ux=ux, uy=uy,
                edge_overshoot=EDGE_OVERSHOOT,
                _out_per_angle=_per_angle_out if DIRECTIONAL_FILTERING else None,
                verbose=VERBOSE,
            )

            # Lock stiffener cells back to base zone

            _elem_half_plies = None
            if DIRECTIONAL_FILTERING and _per_angle_out:
                from filter import (
                    balance_angle_grids,
                    clamp_per_angle_zone_grids,
                    combine_per_angle_zone_grids,
                                compute_per_element_plies,
                )
                _per_angle_out = clamp_per_angle_zone_grids(_per_angle_out, filtered_grid)
                _per_angle_out = balance_angle_grids(_per_angle_out, verbose=VERBOSE)
                filtered_grid = combine_per_angle_zone_grids(
                    _per_angle_out,
                    valid_mask=(filtered_grid >= 0),
                )
                _elem_half_plies = compute_per_element_plies(
                    _per_angle_out, eid_grid, sst, N_ZONES,
                    balance_pairs=False,
                    verbose=VERBOSE,
                )
            sorted_counts = sorted(sst.zone_ply_counts)
            elem_nplies_filtered = {}
            for iy in range(filtered_grid.shape[0]):
                for ix in range(filtered_grid.shape[1]):
                    eid = eid_grid[iy, ix]
                    if eid < 0:
                        continue
                    zone = filtered_grid[iy, ix]
                    zone = max(0, min(zone, len(sorted_counts) - 1))
                    elem_nplies_filtered[eid] = sorted_counts[zone]
            best_assignment_filtered = tuple(sorted_counts)

            # The compensated run replaced the SST, the filtered grid and the
            # per-element ply subsets, so the stiffener-base skin overrides
            # built from the pre-compensation state are stale — rebuild them.
        else:
            elem_nplies_filtered = dict(elem_nplies_sst)
            best_assignment_filtered = best_assignment_sst
        actual_total_filtered = sum(
            n * sst.manufacturing_ply_thickness for n in elem_nplies_filtered.values()
        )
        final_rel_error_filtered = (
            (actual_total_filtered - _orig_fso_total) / _orig_fso_total * 100
        )
        _R_final = actual_total_sst / _orig_fso_total
        print(f"\n  Final SST volume ratio R: {_R_final:.2f}x "
              f"(target: 1.0x = {_orig_fso_total:.0f})")

elif VOLUME_COMPENSATE and not OPTISTRUCT_AVAILABLE:
    print(f"\n  [SKIP] VOLUME_COMPENSATE requires OptiStruct for FSO re-run.")

# ── ALWAYS restore original reference values ───────────────────────────────
# The GuidePly baseline, knockdown calculations, and volume-scaled comparisons
# must use the ORIGINAL VF results, not the compensated ones.
fso_total_thickness = _orig_fso_total
vol_frac = _orig_vol_frac
OptimalCompliance = _orig_optimal
BaselineCompliance = _orig_baseline

# ==============================================================================
# FINAL_TRIMMING: guarantee 100% manufacturable fibres in the final design
# ==============================================================================
# Deletes any fibre segment still shorter than MCL (per the fibre-coverage
# plot metric) from the per-angle ply shapes — per angle, decoupled: only the
# offending angle's shape is trimmed locally.  Everything downstream (fibre
# coverage plots, SST_filtered / SST_filtered_scaled exports, volumes,
# compliances, all visual plots) is rebuilt from the trimmed grids.
if (FINAL_TRIMMING and MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0
        and DIRECTIONAL_FILTERING and _per_angle_out):
    print("\n" + "=" * 64)
    print("  FINAL TRIMMING: removing residual unmanufacturable fibres")
    print("=" * 64)
    from filter import (
        final_manufacturability_trim,
        combine_per_angle_zone_grids,
        compute_per_element_plies,
    )

    _ft_mcl = MIN_CUT_LENGTH
    _per_angle_out = final_manufacturability_trim(
        _per_angle_out, ux, uy, _ft_mcl, element_size, AFP_TAPE_WIDTH,
        verbose=VERBOSE,
    )

    # Rebuild the combined grid and re-lock the stiffener base region.
    filtered_grid = combine_per_angle_zone_grids(
        _per_angle_out, valid_mask=(filtered_grid >= 0))

    # Rebuild per-element ply subsets and zone assignment from the trimmed
    # grids so the exported models reflect the removed fibres.
    _elem_half_plies = compute_per_element_plies(
        _per_angle_out, eid_grid, sst, N_ZONES,
        balance_pairs=False,
        verbose=False,
    )
    _sorted_counts_ft = sorted(sst.zone_ply_counts)
    elem_nplies_filtered = {}
    for _iy in range(filtered_grid.shape[0]):
        for _ix in range(filtered_grid.shape[1]):
            _eid = eid_grid[_iy, _ix]
            if _eid < 0:
                continue
            _zone = max(0, min(filtered_grid[_iy, _ix],
                               len(_sorted_counts_ft) - 1))
            elem_nplies_filtered[_eid] = _sorted_counts_ft[_zone]

    actual_total_filtered = sum(
        n * sst.manufacturing_ply_thickness
        for n in elem_nplies_filtered.values()
    )
    final_rel_error_filtered = (
        (actual_total_filtered - fso_total_thickness)
        / fso_total_thickness * 100
    )
    print(f"  Post-trim filtered volume: {actual_total_filtered:.2f} mm "
          f"({final_rel_error_filtered:+.2f}% vs FSO target)")

    # Rebuild the stiffener base-skin overrides from the trimmed grids.
# ── Dn-rule re-check on the delivered (filtered) design ─────────────────────
# The morphology filter moves zone boundaries, so adjacency can change after
# filtering; re-report the rule on the final element assignment (designable
# skin elements — the same field the zone refinement acted on).
if DN_RULE:
    _dn_nviol_f, _dn_worst_f = _dn_rule_check(elem_nplies_filtered)
    if _dn_nviol_f == 0:
        print(f"\n  Dn-rule (final filtered design): PASS "
              f"(max adjacent-zone jump {_dn_worst_f} <= {DN_RULE} plies)")
    else:
        print(f"\n  WARNING: Dn-rule (final filtered design): "
              f"{_dn_nviol_f} adjacent-element jump(s) exceed {DN_RULE} "
              f"plies (worst {_dn_worst_f}). The morphology filter and/or "
              f"stiffener-base zone introduced jumps the zone refinement "
              f"could not prevent.")

# ==============================================================================
# Step 5: Visualize
# ==============================================================================
print("\n" + "=" * 64)
print("  STEP 5: Visualization")
print("=" * 64)

# Build combined arrays (skin + stiffener) so scatter plots show stiffener base skin zones
_all_X, _all_Y, _all_eids = X, Y, eids

# SST table (condensed, symmetric half) and half-stack side view — must be
# plotted BEFORE zone_ply_counts is narrowed to the counts actually used
# below, so they show the SST exactly as built for this run.
plot_sst_table(sst, save_path="sst_table.png")
plot_sst_side_view(sst, save_path="sst_side_view.png")

# SST-only plot
sst.zone_ply_counts = sorted(set(elem_nplies_sst.values()))
plot_sst_zones(
    _all_X, _all_Y, elem_nplies_sst, _all_eids, sst,
    save_path="sst_zones_only.png",
)

# VF compensation evolution plot
if _comp_applied and _post_vc_VF is not None:
    plot_sst_zones_vc_evolution(
        _all_X, _all_Y, _all_eids,
        _pre_vc_elem_nplies, _pre_vc_sst_zpc, _pre_vc_sst_mfg_t, _orig_vol_frac,
        elem_nplies_sst, sorted(set(elem_nplies_sst.values())),
        sst.manufacturing_ply_thickness, _post_vc_VF,
        save_path="sst_zones_vc_evolution.png",
    )

# Filtered plot
sst.zone_ply_counts = sorted(set(elem_nplies_filtered.values()))
plot_sst_zones(
    _all_X, _all_Y, elem_nplies_filtered, _all_eids, sst,
    save_path="sst_zones_filtered.png",
)

# Side-by-side comparison (only meaningful when filtering was actually applied)
if MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0:
    from filter import plot_filtered_comparison
    plot_filtered_comparison(
        _all_X, _all_Y, _all_eids, elem_nplies_sst, elem_nplies_filtered, sst,
        save_path="filtered_zones_comparison.png",
    )

    from filter import plot_zone_grid_manufacturing, plot_fso_vs_filtered, \
        plot_zone_fiber_coverage, plot_ply_shape_differences, elem_nplies_to_grid

    _filtered_plot_counts = sorted(set(elem_nplies_filtered.values()))
    _filtered_grid_plot = elem_nplies_to_grid(
        elem_nplies_filtered,
        eid_grid,
        _filtered_plot_counts,
    )
    _comparison_plot_counts = sorted(
        set(elem_nplies_sst.values()) | set(elem_nplies_filtered.values()))
    _zone_grid_plot = elem_nplies_to_grid(
        elem_nplies_sst,
        eid_grid,
        _comparison_plot_counts,
    )
    _filtered_grid_comp_plot = elem_nplies_to_grid(
        elem_nplies_filtered,
        eid_grid,
        _comparison_plot_counts,
    )
    plot_zone_grid_manufacturing(
        _filtered_grid_plot, ux, uy,
        _filtered_plot_counts,
        sst.manufacturing_ply_thickness,
        save_path="sst_zones_filtered_smoothed.png",
        title=f"SST + MCL filtering (MCL={MIN_CUT_LENGTH:.1f} mm)",
    )

    plot_zone_fiber_coverage(
        filtered_grid, ux, uy, sst,
        tape_width=AFP_TAPE_WIDTH,
        save_prefix="fiber_coverage",
        mcl=MIN_CUT_LENGTH,
        per_angle_zone_grids=_per_angle_out if DIRECTIONAL_FILTERING else None,
    )

    if DIRECTIONAL_FILTERING and _per_angle_out:
        plot_ply_shape_differences(
            _per_angle_out,
            filtered_grid,
            ux,
            uy,
            N_ZONES,
            save_path="ply_shape_differences.png",
            stack_order=list(dict.fromkeys(sst.ply_angles)),
        )

    # before‑filter fiber coverage for comparison
    plot_zone_fiber_coverage(
        zone_grid, ux, uy, sst,
        tape_width=AFP_TAPE_WIDTH,
        save_prefix="fiber_coverage_raw",
        mcl=MIN_CUT_LENGTH,
    )

    plot_fso_vs_filtered(
        _zone_grid_plot, _filtered_grid_comp_plot, ux, uy,
        _comparison_plot_counts,
        sst.manufacturing_ply_thickness,
        save_path="fso_vs_filtered.png",
    )

# ==============================================================================
# Step 6: Export .fem files and run OptiStruct reanalysis
# ==============================================================================
print("\n" + "=" * 64)
print("  STEP 6: Export .fem files and run OptiStruct reanalysis")
print("=" * 64)

# ── 6a: FSO Exact model (unconstrained theoretical optimum) ──────────────────
# Each element gets its own PCOMP with the exact FSO-optimal continuous
# thicknesses, parsed directly from the .sh file.  No manufacturing
# constraints, no zones, no filtering, no SMEAR/SYM — explicit plies only.
print("\n  --- FSO Exact model (unconstrained theoretical optimum) ---")
fso_exact_jobname = "FSO_exact"
fso_exact_filename = fso_exact_jobname + ".fem"

if OPTISTRUCT_AVAILABLE:
    _guide_stack = sst.get_laminate(sst.n_max)
    _n_unique = export_fem_fso_exact(
        source_fem_path=FEM_FILE,
        sh_file_path=SH_FILE,
        output_fem_path=fso_exact_filename,
        guide_stack=_guide_stack,
        preserve_pids=None,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=None,
        verbose=VERBOSE,
    )

    _archive_results(fso_exact_jobname)
    run_solver(filename=fso_exact_filename)
else:
    _n_unique = 0
    raise RuntimeError("OptiStruct is required for FSO_exact generation.")

# The .out "Total Volume" also counts any stiffener shells in the model, but
# we only ever report SKIN volume (DfM is skin-only, like every other run).
fso_exact_volume = read_totalthickness_from_out(jobname=fso_exact_jobname)
if flat_stiffener_pids:
    # Flat-side: the attached stiffener shells are in the FSO_exact model but
    # are not skin, so drop them. The FSO_exact skin volume is exactly the
    # continuous FSO skin, i.e. fso_skin_total.
    fso_exact_volume = fso_skin_total
GuideOptimalCompliance, optimal_displacement_guide = read_results(jobname=fso_exact_jobname)

if VERBOSE:
    print(f"  Unique PCOMP properties: {_n_unique}")
    print(f"  Max. Displacement (-Z): {abs(optimal_displacement_guide)} mm")
print(f"  FSO Exact Compliance: {GuideOptimalCompliance} Nmm")
print(f"  FSO Exact Volume (skin only): {fso_exact_volume:.2f}")

# ── 6a-flat: Uniform panel baseline (same volume, no optimization) ───────────
# A flat panel with the original input skin layup scaled uniformly to match
# the FSO volume. Stiffener elements keep their original local stiffener ply
# on top of the scaled base skin. This gives a fair baseline comparison
# against the optimized SST models.
print("\n  --- Uniform Panel Baseline ---")

# Total thickness of the original input skin (full laminate, excluding
# any stiffener plies).
_original_skin_thickness = sum(t for _, t in initial_plies)

# Scale the whole skin uniformly so total skin volume matches fso_total_thickness
# (skin + stiffener base skin; stiffener plies are excluded from this target).
vol_frac_scaling = (
    fso_total_thickness / (_original_skin_thickness * len(eids))
    if len(eids) > 0 else 1.0
)

if VERBOSE:
    print(f"  Original skin thickness: {_original_skin_thickness:.4f} mm")
    print(f"  Effective FSO Volume Fraction: {vol_frac_scaling:.4f}")

# Build the scaled source-skin layup once and reuse it for every element.
# Use the SST guide-stack sequence (not the raw input layup order) so the
# baseline, FSO_exact and SST all share one stacking and differ only in
# thickness distribution. Total skin stays nominal thickness x VF; the guide
# has the same angle proportions as the input layup, only a different order.
_guide_seq = sst.get_laminate(sst.n_max)
_guide_ply_t = (_original_skin_thickness * vol_frac_scaling) / len(_guide_seq)
guide_source_skin_layup = [
    {"mid": 1, "T": _guide_ply_t, "theta": angle}
    for angle in _guide_seq
]

guide_elem_nplies = {eid: 0 for eid in eids}
guide_sst = copy.deepcopy(sst)
guide_sst.zone_ply_counts = [0]

guide_jobname = "GuidePly_baselineNO"
guide_filename = guide_jobname + ".fem"

if OPTISTRUCT_AVAILABLE:
    export_fem(
        source_fem_path=FEM_FILE,
        output_fem_path=guide_filename,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=guide_sst,
        elem_nplies=guide_elem_nplies,
        mat_id=1,
        source_skin_layup=guide_source_skin_layup,
        verbose=VERBOSE,
    )

    _archive_results(guide_jobname)
    run_solver(filename=guide_filename)
else:
    raise RuntimeError("OptiStruct is required for GuidePly_baselineNO generation.")

_guide_scaled_skin_t = _original_skin_thickness * vol_frac_scaling
guideply_nonoptimized_total_thickness = _guide_scaled_skin_t * len(eids)
compliance_guide_uniform, baseline_displacement_guide = read_results(jobname=guide_jobname)

if VERBOSE:
    print(f"  Scaled skin thickness: {_guide_scaled_skin_t:.4f} mm")
print(f"  Uniform Panel Compliance: {compliance_guide_uniform} Nmm")

# ── 6b: SST-only model ───────────────────────────────────────────────────────
print("\n  --- SST-only model ---")
sst_jobname = "SST_only"
sst_filename = sst_jobname + ".fem"

if OPTISTRUCT_AVAILABLE:
    sst.zone_ply_counts = sorted(set(elem_nplies_sst.values()))
    export_fem(
        source_fem_path=FEM_FILE,
        output_fem_path=sst_filename,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=sst,
        elem_nplies=elem_nplies_sst,
        mat_id=1,
        verbose=VERBOSE,
)

    _archive_results(sst_jobname)
    run_solver(filename=sst_filename)
else:
    raise RuntimeError("OptiStruct is required for SST_only generation.")

compliance_sst, displacement_sst = read_results(jobname=sst_jobname)
if VERBOSE:
    print(f"  Max. Displacement (-Z): {abs(displacement_sst)} mm")
print(f"  Compliance: {compliance_sst} Nmm")

# ── 6c: SST + filtered model ─────────────────────────────────────────────────
print("\n  --- SST + filtered model ---")
filtered_jobname = "SST_filtered"
filtered_filename = filtered_jobname + ".fem"

if OPTISTRUCT_AVAILABLE:
    sst.zone_ply_counts = sorted(set(elem_nplies_filtered.values()))
    export_fem(
        source_fem_path=FEM_FILE,
        output_fem_path=filtered_filename,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=sst,
        elem_nplies=elem_nplies_filtered,
        mat_id=1,
        elem_half_plies=_elem_half_plies,
        verbose=VERBOSE,
    )

    _archive_results(filtered_jobname)
    run_solver(filename=filtered_filename)
else:
    raise RuntimeError("OptiStruct is required for SST_filtered generation.")

compliance_filtered, displacement_filtered = read_results(jobname=filtered_jobname)
if VERBOSE:
    print(f"  Max. Displacement (-Z): {abs(displacement_filtered)} mm")
print(f"  Compliance: {compliance_filtered} Nmm")

# ── 6d: SST-only volume-scaled ───────────────────────────────────────────────
# Scale ply thickness uniformly so total thickness matches FSO. This gives a
# fair compliance comparison at equal volume between SST and guide ply models.
print("\n  --- SST-only (volume-scaled) ---")
sst_scaled_jobname = "SST_only_scaled"
sst_scaled_filename = sst_scaled_jobname + ".fem"

sst_scaled = copy.deepcopy(sst)
scale_factor = fso_total_thickness / actual_total_sst
sst_scaled.manufacturing_ply_thickness *= scale_factor

if VERBOSE:
    print(f"  Volume scaling factor: {scale_factor:.4f} "
          f"(ply thickness {sst.manufacturing_ply_thickness:.4f} -> "
          f"{sst_scaled.manufacturing_ply_thickness:.4f} mm)")

if OPTISTRUCT_AVAILABLE:
    sst_scaled.zone_ply_counts = sorted(set(elem_nplies_sst.values()))
    export_fem(
        source_fem_path=FEM_FILE,
        output_fem_path=sst_scaled_filename,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=sst_scaled,
        elem_nplies=elem_nplies_sst,
        mat_id=1,
        verbose=VERBOSE,
    )

    _archive_results(sst_scaled_jobname)
    run_solver(filename=sst_scaled_filename)
else:
    raise RuntimeError("OptiStruct is required for SST_only_scaled generation.")

compliance_sst_scaled, displacement_sst_scaled = read_results(jobname=sst_scaled_jobname)
if VERBOSE:
    print(f"  Max. Displacement (-Z): {abs(displacement_sst_scaled)} mm")
print(f"  Compliance: {compliance_sst_scaled} Nmm")

plot_sst_zones_scaled(
    _all_X, _all_Y, elem_nplies_sst, _all_eids, sst, sst_scaled,
    save_path="sst_zones_only_scaled.png",
)

# ── 6e: SST + filtered volume-scaled ─────────────────────────────────────────
print("\n  --- SST + filtered (volume-scaled) ---")
filt_scaled_jobname = "SST_filtered_scaled"
filt_scaled_filename = filt_scaled_jobname + ".fem"

filt_scaled = copy.deepcopy(sst)
scale_factor_filt = fso_total_thickness / actual_total_filtered
filt_scaled.manufacturing_ply_thickness *= scale_factor_filt

if VERBOSE:
    print(f"  Volume scaling factor: {scale_factor_filt:.4f} "
          f"(ply thickness {sst.manufacturing_ply_thickness:.4f} -> "
          f"{filt_scaled.manufacturing_ply_thickness:.4f} mm)")

if OPTISTRUCT_AVAILABLE:
    filt_scaled.zone_ply_counts = sorted(set(elem_nplies_filtered.values()))
    export_fem(
        source_fem_path=FEM_FILE,
        output_fem_path=filt_scaled_filename,
        flat_stiffener_pids=(flat_stiffener_pids or None),
        sst=filt_scaled,
        elem_nplies=elem_nplies_filtered,
        mat_id=1,
        elem_half_plies=_elem_half_plies,
        verbose=VERBOSE,
    )

    _archive_results(filt_scaled_jobname)
    run_solver(filename=filt_scaled_filename)
else:
    raise RuntimeError("OptiStruct is required for SST_filtered_scaled generation.")

compliance_filtered_scaled, displacement_filtered_scaled = read_results(jobname=filt_scaled_jobname)
if VERBOSE:
    print(f"  Max. Displacement (-Z): {abs(displacement_filtered_scaled)} mm")
print(f"  Compliance: {compliance_filtered_scaled} Nmm")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 64)
print("  SUMMARY")
print("=" * 64)

# All guide ply models are volume-matched to FSO by construction.
# SST-only and SST+filtered run at their natural volumes (direct assignment).
# _scaled versions have ply thickness scaled to match FSO volume for fair comparison.

print(f"\n  Reference thickness (FSO total): {fso_total_thickness:.2f} mm")
print(f"\n  {'Model':<32} {'Compliance':>12} {'Total Thickness':>12} {'Vol err':>8}")
print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*8}")
print(f"  {'Original FSO Optimum':<32} {OptimalCompliance:>12.2f} {fso_total_thickness:>12.2f} {'':>8}")
print(f"  {'Uniform Panel Baseline':<32} {compliance_guide_uniform:>12.2f} {guideply_nonoptimized_total_thickness:>12.2f} {'':>8}")
_fso_exact_vol_err = ((fso_exact_volume - fso_total_thickness) / fso_total_thickness * 100) if fso_total_thickness else 0
print(f"  {'FSO Exact (Theoretical Optimum)':<32} {GuideOptimalCompliance:>12.2f} {fso_exact_volume:>12.2f} {_fso_exact_vol_err:>+7.2f}%")

print(f"  {'SST-only (actual volume)':<32} {compliance_sst:>12.2f} {actual_total_sst:>12.2f} "
      f"{final_rel_error_sst:>+7.2f}%")
print(f"  {'SST-only (volume-scaled)':<32} {compliance_sst_scaled:>12.2f} {fso_total_thickness:>12.2f} {'':>8}")
print(f"  {'SST + filtered (actual volume)':<32} {compliance_filtered:>12.2f} {actual_total_filtered:>12.2f} "
      f"{final_rel_error_filtered:>+7.2f}%")
print(f"  {'SST + filtered (volume-scaled)':<32} {compliance_filtered_scaled:>12.2f} {fso_total_thickness:>12.2f} {'':>8}")

# Fair comparison: volume-scaled SST vs volume-matched guide ply models
if GuideOptimalCompliance != 0:
    pct_sst = (compliance_sst - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    pct_filtered = (compliance_filtered - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    # if COMPENSATE_METHOD == "sst_only" and VOLUME_COMPENSATE:
    #     pct_sst = (compliance_sst - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    #     pct_filtered = (compliance_filtered_scaled - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    # elif COMPENSATE_METHOD == "sst_filtered" and VOLUME_COMPENSATE:
    #     pct_sst = (compliance_sst_scaled - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    #     pct_filtered = (compliance_filtered - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    # else:
    #     pct_sst = (compliance_sst_scaled - GuideOptimalCompliance) / GuideOptimalCompliance * 100
    #     pct_filtered = (compliance_filtered_scaled - GuideOptimalCompliance) / GuideOptimalCompliance * 100
else:
    pct_sst = pct_filtered = 0.0

if compliance_guide_uniform != 0:
    pct_sst_vs_base = (compliance_sst - compliance_guide_uniform) / compliance_guide_uniform * 100
    pct_filtered_vs_base = (compliance_filtered - compliance_guide_uniform) / compliance_guide_uniform * 100
    # if COMPENSATE_METHOD == "sst_only" and VOLUME_COMPENSATE:
    #     pct_sst_vs_base = (compliance_sst - compliance_guide_uniform) / compliance_guide_uniform * 100
    #     pct_filtered_vs_base = (compliance_filtered_scaled - compliance_guide_uniform) / compliance_guide_uniform * 100
    # elif COMPENSATE_METHOD == "sst_filtered" and VOLUME_COMPENSATE:
    #     pct_sst_vs_base = (compliance_sst_scaled - compliance_guide_uniform) / compliance_guide_uniform * 100
    #     pct_filtered_vs_base = (compliance_filtered - compliance_guide_uniform) / compliance_guide_uniform * 100
    # else:
    #     pct_sst_vs_base = (compliance_sst_scaled - compliance_guide_uniform) / compliance_guide_uniform * 100
    #     pct_filtered_vs_base = (compliance_filtered_scaled - compliance_guide_uniform) / compliance_guide_uniform * 100
else:
    pct_sst_vs_base = pct_filtered_vs_base = 0.0

print(f"\n  Scaled-volume performance comparison (Total thickness = {fso_total_thickness:.2f} mm):")
print(f"  {'-'*32} {'vs FSO Exact':>14} {'vs Baseline':>14}")
print(f"  {'SST-only':<32} {-pct_sst:>+13.2f}% {-pct_sst_vs_base:>+13.2f}%")
print(f"  {'SST + filtered':<32} {-pct_filtered:>+13.2f}% {-pct_filtered_vs_base:>+13.2f}%")
print(f"\n    vs Guide Opt  = performance vs theoretical optimum")
print(f"    vs Baseline   = performance vs non-optimized uniform panel")

if MIN_CUT_LENGTH and MIN_CUT_LENGTH > 0:
    print(f"\n  Morphological filter MCL: {MIN_CUT_LENGTH} mm")
else:
    print(f"\n  Morphological filter: disabled (filtered = SST-only)")

if FILTER_ANALYSIS:
    print("\n" + "=" * 64)
    print("  FILTER ANALYSIS")
    print("=" * 64)
    print(f"  Range: {FILTER_ANALYSIS_MIN:.2f} to {FILTER_ANALYSIS_MAX:.2f} mm (step {FILTER_ANALYSIS_STEP:.2f})")

    from filter import (
        build_grid_mapping,
        zones_to_grid,
        apply_morphology_filter,
        plot_mcl_sweep,
        plot_mcl_sweep_comparison,
        plot_mcl_sweep_volume,
        plot_mcl_sweep_guide_comparison,
        plot_mcl_sweep_directional_comparison,
        plot_mcl_sweep_volume_comparison,
        plot_mcl_sweep_ordering_comparison,
        plot_mcl_sweep_volume_ordering,
        plot_mcl_sweep_volume_all,
        plot_mcl_sweep_volume_dilate_erode,
    )

    mcl_values = _mcl_values(FILTER_ANALYSIS_MIN, FILTER_ANALYSIS_MAX, FILTER_ANALYSIS_STEP)
    eid_grid, elem_to_grid, ux, uy = build_grid_mapping(X, Y, eids)
    element_size = float(np.median(np.diff(ux)))
    panel_size = float(max(ux[-1] - ux[0], uy[-1] - uy[0])) + element_size
    zone_grid = zones_to_grid(cluster_data, centers, elem_to_grid, eid_grid.shape)

    # Build importance grid for volume matching
    importance_grid = np.full(eid_grid.shape, 0.0)
    for eid, thick in thickness_data.items():
        if eid in elem_to_grid:
            iy, ix = elem_to_grid[eid]
            importance_grid[iy, ix] = thick

    effective_mcl_step = 2.0 * element_size
    if FILTER_ANALYSIS_STEP < effective_mcl_step:
        n_unique = len(set(int(round(m / element_size / 2)) for m in mcl_values if m > 0))
        print(f"\n  [WARN] FILTER_ANALYSIS_STEP={FILTER_ANALYSIS_STEP:.2f}mm is below the effective")
        print(f"         resolution of {effective_mcl_step:.2f}mm (2 x element_size={element_size:.2f}mm).")
        print(f"         The SE radius is quantised to integer pixels, so only {n_unique}")
        print(f"         unique structuring elements exist across {len(mcl_values)} sweep points.")
        print(f"         Consider using step >= {effective_mcl_step:.2f}mm.")

    # Extended MCL range for dilate/erode volume plot (up to 20 % of panel)
    max_mcl_ext = max(panel_size * 0.2, FILTER_ANALYSIS_MAX)
    mcl_values_ext = _mcl_values(FILTER_ANALYSIS_MIN, max_mcl_ext, FILTER_ANALYSIS_STEP)

    from filter import print_steiner_verification
    print_steiner_verification(zone_grid, FILTER_ANALYSIS_MAX, element_size, N_ZONES)

    sweep_zone_counts = sorted(set(elem_nplies_sst.values()))

    def _run_filter_method(
        method: str,
        sst_obj,
        prefix: str = "filter_analysis",
        sst_scaled_compliance: float = None,
        sst_actual_total: float = None,
        directional: bool = None,
    ) -> dict[str, list[float]]:
        if sst_scaled_compliance is None:
            sst_scaled_compliance = compliance_sst_scaled
        if sst_actual_total is None:
            sst_actual_total = actual_total_sst
        if directional is None:
            directional = DIRECTIONAL_FILTERING

        print(f"\n  --- {method} (prefix={prefix}, directional={directional}) ---")
        perf_vs_opt = []
        perf_vs_base = []
        vol_delta_pct = []

        per_zone_angles_sweep = {}
        for _z in range(1, N_ZONES):
            _np = sst_obj.zone_ply_counts[min(_z, len(sst_obj.zone_ply_counts) - 1)]
            _lam = sst_obj.get_laminate(_np)
            per_zone_angles_sweep[_z] = sorted(set(_lam))

        for mcl in mcl_values:
            print(f"\n  Sweep point: MCL = {mcl:.2f} mm")

            if abs(mcl) < 1e-12:
                compliance_mcl_scaled = sst_scaled_compliance
                out_exists = compliance_mcl_scaled != 0.0
                vol_delta_pct.append(0.0)
                print("    Using SST-only volume-scaled reference (no filtering).")
            else:
                _sweep_per_angle = {}
                filtered_grid_mcl = apply_morphology_filter(
                    zone_grid, mcl, element_size, N_ZONES,
                    method=method,
                    ply_angles=sst_obj.ply_angles if directional else None,
                    tape_width=AFP_TAPE_WIDTH if directional else None,
                    directional_mode=DIRECTIONAL_MODE if directional else None,
            
                    per_zone_angles=per_zone_angles_sweep,
                    run_fiber_enforcement=False,
            
                    ux=ux, uy=uy,
                    edge_overshoot=EDGE_OVERSHOOT,
                    _out_per_angle=_sweep_per_angle if directional else None,
                )

                elem_nplies_mcl = {}
                for iy in range(filtered_grid_mcl.shape[0]):
                    for ix in range(filtered_grid_mcl.shape[1]):
                        eid = eid_grid[iy, ix]
                        if eid < 0:
                            continue
                        zone = filtered_grid_mcl[iy, ix]
                        zone = max(0, min(zone, len(sweep_zone_counts) - 1))
                        elem_nplies_mcl[eid] = sweep_zone_counts[zone]

                # Per-angle FEM export with ±45 balance (matches single-run pipeline)
                _sweep_half_plies = None
                if directional and _sweep_per_angle:
                    from filter import (
                        balance_angle_grids,
                        clamp_per_angle_zone_grids,
                        combine_per_angle_zone_grids,
                                        compute_per_element_plies,
                    )
                    _sweep_per_angle = clamp_per_angle_zone_grids(
                        _sweep_per_angle, filtered_grid_mcl)
                    _sweep_per_angle = balance_angle_grids(_sweep_per_angle)
                    filtered_grid_mcl = combine_per_angle_zone_grids(
                        _sweep_per_angle,
                        valid_mask=(filtered_grid_mcl >= 0),
                    )
                    _sweep_half_plies = compute_per_element_plies(
                        _sweep_per_angle, eid_grid, sst_obj, N_ZONES,
                        balance_pairs=False,
                    )

                actual_total_mcl = sum(
                    elem_nplies_mcl[eid] * sst_obj.manufacturing_ply_thickness for eid in eids
                )
                if actual_total_mcl <= 0:
                    perf_vs_opt.append(np.nan)
                    perf_vs_base.append(np.nan)
                    vol_delta_pct.append(np.nan)
                    print("    [WARN] Invalid total thickness; skipping this point.")
                    continue

                vol_delta = (actual_total_mcl - sst_actual_total) / sst_actual_total * 100.0
                vol_delta_pct.append(vol_delta)
                scale_factor_mcl = fso_total_thickness / actual_total_mcl
                print(f"    Volume scaling factor: {scale_factor_mcl:.4f}, volume delta: {vol_delta:+.2f}%")

                sst_mcl_scaled = copy.deepcopy(sst_obj)
                sst_mcl_scaled.manufacturing_ply_thickness *= scale_factor_mcl
                sst_mcl_scaled.zone_ply_counts = sweep_zone_counts

                mcl_jobname = f"{prefix}_{method}_mcl{_mcl_tag(mcl)}"
                mcl_filename = mcl_jobname + ".fem"
                mcl_out_path = mcl_jobname + ".out"
                mcl_sh_path = mcl_jobname + ".sh"

                _archive_results(mcl_jobname)
                if FILTER_ANALYSIS_RERUN:
                    _archive_results(mcl_jobname)

                if os.path.exists(mcl_out_path) and not FILTER_ANALYSIS_RERUN:
                    print(f"    Using cached result: {mcl_out_path}")
                else:
                    export_fem(
                        source_fem_path=FEM_FILE,
                        output_fem_path=mcl_filename,
                        flat_stiffener_pids=(flat_stiffener_pids or None),
                        sst=sst_mcl_scaled,
                        elem_nplies=elem_nplies_mcl,
                        mat_id=1,
                        elem_half_plies=_sweep_half_plies,
                    )

                    if OPTISTRUCT_AVAILABLE:
                        run_solver(filename=mcl_filename)
                    else:
                        if not os.path.exists(mcl_out_path):
                            print(f"    [SKIP] No OptiStruct and no cached result: {mcl_out_path}")

                compliance_mcl_scaled, _ = read_results(jobname=mcl_jobname)
                out_exists = os.path.exists(mcl_out_path)

            if compliance_mcl_scaled == 0.0 and not out_exists:
                perf_vs_opt.append(np.nan)
                perf_vs_base.append(np.nan)
                print("    [WARN] No compliance result available for this MCL.")
                continue

            if GuideOptimalCompliance != 0:
                perf_opt = (GuideOptimalCompliance - compliance_mcl_scaled) / GuideOptimalCompliance * 100.0
            else:
                perf_opt = np.nan

            if compliance_guide_uniform != 0:
                perf_base = (compliance_guide_uniform - compliance_mcl_scaled) / compliance_guide_uniform * 100.0
            else:
                perf_base = np.nan

            perf_vs_opt.append(perf_opt)
            perf_vs_base.append(perf_base)
            print(f"    Performance: vs Guide Opt = {perf_opt:+.2f}%, vs Baseline = {perf_base:+.2f}%")

        return {"pct_vs_opt": perf_vs_opt, "pct_vs_baseline": perf_vs_base, "vol_delta_pct": vol_delta_pct}

    def _run_volume_only(method: str, directional: bool,
                         mcl_list: list[float] = None) -> list[float]:
        """Collect volume deltas for a method without running FEM."""
        if mcl_list is None:
            mcl_list = mcl_values
        print(f"\n  --- {method} (volume-only, directional={directional}) ---")
        deltas = []
        for mcl in mcl_list:
            if abs(mcl) < 1e-12:
                deltas.append(0.0)
                continue
            fg = apply_morphology_filter(
                zone_grid, mcl, element_size, N_ZONES,
                method=method,
                ply_angles=sst.ply_angles if directional else None,
                tape_width=AFP_TAPE_WIDTH if directional else None,
                directional_mode=DIRECTIONAL_MODE if directional else None,
        
                per_zone_angles=per_zone_angles,
                run_fiber_enforcement=False,
        
                ux=ux, uy=uy,
                edge_overshoot=EDGE_OVERSHOOT,
            )
            total = 0.0
            for iy in range(fg.shape[0]):
                for ix in range(fg.shape[1]):
                    eid = eid_grid[iy, ix]
                    if eid < 0:
                        continue
                    zone = fg[iy, ix]
                    zone = max(0, min(zone, len(sweep_zone_counts) - 1))
                    total += sweep_zone_counts[zone] * sst.manufacturing_ply_thickness
            vol_delta = (total - actual_total_sst) / actual_total_sst * 100.0
            deltas.append(vol_delta)
            print(f"    MCL={mcl:.1f} mm: volume delta = {vol_delta:+.2f}%")
        return deltas

    # ── Determine which methods to sweep ───────────────────────────────────
    if FILTER_ANALYSIS_ALL:
        _sweep_methods = ["close_only", "open_only", "close_open", "open_close"]
    else:
        _sweep_methods = [FILTERING_METHOD]
        print(f"\n  FILTER_ANALYSIS_ALL=False: sweeping only '{FILTERING_METHOD}'")

    # ── Volume-only pre-computes (only needed for ALL mode plots) ──────────
    if FILTER_ANALYSIS_ALL:
        dilate_vol = _run_volume_only("dilate_only", DIRECTIONAL_FILTERING,
                                      mcl_list=mcl_values_ext)
        erode_vol = _run_volume_only("erode_only", DIRECTIONAL_FILTERING,
                                     mcl_list=mcl_values_ext)
        close_only_ext_vol = _run_volume_only("close_only", DIRECTIONAL_FILTERING,
                                              mcl_list=mcl_values_ext)
        open_only_ext_vol = _run_volume_only("open_only", DIRECTIONAL_FILTERING,
                                             mcl_list=mcl_values_ext)

    # ── FEM sweeps ─────────────────────────────────────────────────────────
    _all_method_results = {}
    for _method in _sweep_methods:
        _all_method_results[_method] = _run_filter_method(_method, sst)

    # Assign named variables for ALL-mode backward compatibility
    if FILTER_ANALYSIS_ALL:
        close_only_results = _all_method_results["close_only"]
        open_only_results = _all_method_results["open_only"]
        close_open_results = _all_method_results["close_open"]
        open_close_results = _all_method_results["open_close"]

    # ── Individual method plots ────────────────────────────────────────────
    for _method in _sweep_methods:
        _mr = _all_method_results[_method]
        plot_mcl_sweep(
            mcl_values=mcl_values,
            pct_vs_opt=_mr["pct_vs_opt"],
            pct_vs_baseline=_mr["pct_vs_baseline"],
            pct_sst_vs_opt=-pct_sst,
            pct_sst_vs_baseline=-pct_sst_vs_base,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_{_method}.png",
            panel_size=panel_size,
        )

    # ── Single-design sweep plot (current design only + SST baseline) ───────
    _current_results = _all_method_results.get(FILTERING_METHOD)
    if _current_results is not None:
        from filter import plot_single_design_sweep
        plot_single_design_sweep(
            mcl_values=mcl_values,
            pct_vs_opt=_current_results["pct_vs_opt"],
            pct_vs_baseline=_current_results["pct_vs_baseline"],
            sst_perf_vs_opt=-pct_sst,
            sst_perf_vs_baseline=-pct_sst_vs_base,
            method=FILTERING_METHOD,
            directional=DIRECTIONAL_FILTERING,
            panel_size=panel_size,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_single_design.png",
        )

    # ── All-mode-only: pairwise, guide, and directional comparisons ────────
    if FILTER_ANALYSIS_ALL:
        # ── Pairwise: close-only vs close-open ──────────────────────────────
        plot_mcl_sweep_comparison(
            mcl_values=mcl_values,
            method_results={"close_only": close_only_results, "close_open": close_open_results},
            pct_sst_vs_opt=-pct_sst,
            pct_sst_vs_baseline=-pct_sst_vs_base,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_close_combined.png",
            panel_size=panel_size,
        )
        plot_mcl_sweep_volume(
            mcl_values=mcl_values,
            method_vol_deltas={
                "close_only": close_only_results["vol_delta_pct"],
                "close_open": close_open_results["vol_delta_pct"],
            },
            save_path=f"{FILTER_ANALYSIS_PREFIX}_close_volume.png",
            panel_size=panel_size,
        )

        # ── Pairwise: open-only vs open-close ───────────────────────────────
        plot_mcl_sweep_comparison(
            mcl_values=mcl_values,
            method_results={"open_only": open_only_results, "open_close": open_close_results},
            pct_sst_vs_opt=-pct_sst,
            pct_sst_vs_baseline=-pct_sst_vs_base,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_open_combined.png",
            panel_size=panel_size,
        )
        plot_mcl_sweep_volume(
            mcl_values=mcl_values,
            method_vol_deltas={
                "open_only": open_only_results["vol_delta_pct"],
                "open_close": open_close_results["vol_delta_pct"],
            },
            save_path=f"{FILTER_ANALYSIS_PREFIX}_open_volume.png",
            panel_size=panel_size,
        )

        # ── Ordering: close-open vs open-close ──────────────────────────────
        plot_mcl_sweep_ordering_comparison(
            mcl_values=mcl_values,
            close_open_results=close_open_results,
            open_close_results=open_close_results,
            pct_sst_vs_opt=-pct_sst,
            pct_sst_vs_baseline=-pct_sst_vs_base,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_ordering_comparison.png",
            panel_size=panel_size,
        )
        plot_mcl_sweep_volume_ordering(
            mcl_values=mcl_values,
            close_open_deltas=close_open_results["vol_delta_pct"],
            open_close_deltas=open_close_results["vol_delta_pct"],
            save_path=f"{FILTER_ANALYSIS_PREFIX}_volume_ordering.png",
            panel_size=panel_size,
        )

        # ── All four methods volume delta ───────────────────────────────────
        plot_mcl_sweep_volume_all(
            mcl_values=mcl_values,
            method_vol_deltas={
                "close_only": close_only_results["vol_delta_pct"],
                "open_only": open_only_results["vol_delta_pct"],
                "close_open": close_open_results["vol_delta_pct"],
                "open_close": open_close_results["vol_delta_pct"],
            },
            save_path=f"{FILTER_ANALYSIS_PREFIX}_volume_all.png",
            panel_size=panel_size,
        )

        plot_mcl_sweep_volume(
            mcl_values=mcl_values_ext,
            method_vol_deltas={"close_only": close_only_ext_vol},
            save_path=f"{FILTER_ANALYSIS_PREFIX}_close_only_volume_ext.png",
            panel_size=panel_size,
        )
        plot_mcl_sweep_volume(
            mcl_values=mcl_values_ext,
            method_vol_deltas={"open_only": open_only_ext_vol},
            save_path=f"{FILTER_ANALYSIS_PREFIX}_open_only_volume_ext.png",
            panel_size=panel_size,
        )

        # ==============================================================
        #  Guide Stack Comparison
        # ==============================================================
        print("\n" + "-" * 64)
        print("  Guide Stack Optimization Comparison")
        print("-" * 64)

        if OPTIMIZE_GUIDE_STACK:
            sst_other = build_sst(
                initial_plies=initial_plies,
                elem_ply_thicknesses=elem_ply_data,
                n_zones=N_ZONES,
                min_ply_thickness=MIN_PLY_THICKNESS,
                covering_plies=COVERING_PLIES,
                max_contiguity=MAX_CONTIGUITY,
                max_disorientation=MAX_DISORIENTATION,
                position_bias=POSITION_BIAS,
                importance_method=IMPORTANCE_METHOD,
                cluster_data=cluster_data,
                cluster_centers=centers,
                doubling_factor_override=DOUBLING_FACTOR,
                optimize_guide_stack=False,
                tolerant_drop_order=SST_TOLERANT_LADDER,
                min_plies=_MIN_PLIES_FSO,
                dn_rule=DN_RULE or 0,
            )
            other_label = "default"
        else:
            sst_other = build_sst(
                initial_plies=initial_plies,
                elem_ply_thicknesses=elem_ply_data,
                n_zones=N_ZONES,
                min_ply_thickness=MIN_PLY_THICKNESS,
                covering_plies=COVERING_PLIES,
                max_contiguity=MAX_CONTIGUITY,
                max_disorientation=MAX_DISORIENTATION,
                position_bias=POSITION_BIAS,
                importance_method=IMPORTANCE_METHOD,
                cluster_data=cluster_data,
                cluster_centers=centers,
                doubling_factor_override=DOUBLING_FACTOR,
                optimize_guide_stack=True,
                tolerant_drop_order=SST_TOLERANT_LADDER,
                min_plies=_MIN_PLIES_FSO,
                dn_rule=DN_RULE or 0,
            )
            other_label = "optimized"

        sst_other.zone_ply_counts = sorted(set(elem_nplies_sst.values()))
        if VERBOSE:
            print(f"\n  Main guide ({'optimized' if OPTIMIZE_GUIDE_STACK else 'default'}): {sst.ply_angles}")
            print(f"  Other guide ({other_label}): {sst_other.ply_angles}")

        other_scaled = copy.deepcopy(sst_other)
        scale_factor_other = fso_total_thickness / actual_total_sst
        other_scaled.manufacturing_ply_thickness *= scale_factor_other
        other_scaled.zone_ply_counts = sweep_zone_counts

        other_sst_jobname = f"SST_only_{other_label}_scaled"
        other_sst_filename = other_sst_jobname + ".fem"

        if OPTISTRUCT_AVAILABLE:
            export_fem(
                source_fem_path=FEM_FILE,
                output_fem_path=other_sst_filename,
                flat_stiffener_pids=(flat_stiffener_pids or None),
                sst=other_scaled,
                elem_nplies=elem_nplies_sst,
                mat_id=1,
                verbose=VERBOSE,
            )
            _archive_results(other_sst_jobname)
            run_solver(filename=other_sst_filename)

        compliance_other_scaled, _ = read_results(jobname=other_sst_jobname)
        if VERBOSE:
            print(f"  Other guide compliance (scaled): {compliance_other_scaled:.2f}")

        if compliance_other_scaled != 0 and GuideOptimalCompliance != 0:
            pct_other_vs_opt = (GuideOptimalCompliance - compliance_other_scaled) / GuideOptimalCompliance * 100.0
        else:
            pct_other_vs_opt = np.nan

        if compliance_other_scaled != 0 and compliance_guide_uniform != 0:
            pct_other_vs_base = (compliance_guide_uniform - compliance_other_scaled) / compliance_guide_uniform * 100.0
        else:
            pct_other_vs_base = np.nan

        other_prefix = f"{FILTER_ANALYSIS_PREFIX}_{other_label}"

        other_close = _run_filter_method(
            "close_only", sst_other, prefix=other_prefix,
            sst_scaled_compliance=compliance_other_scaled,
        )
        other_close_open = _run_filter_method(
            "close_open", sst_other, prefix=other_prefix,
            sst_scaled_compliance=compliance_other_scaled,
        )

        if OPTIMIZE_GUIDE_STACK:
            default_results = {"close_only": other_close, "close_open": other_close_open}
            optimized_results = {"close_only": close_only_results, "close_open": close_open_results}
            pct_sst_default_vs_opt = pct_other_vs_opt
            pct_sst_default_vs_baseline = pct_other_vs_base
            pct_sst_opt_vs_opt = -pct_sst
            pct_sst_opt_vs_baseline = -pct_sst_vs_base
        else:
            default_results = {"close_only": close_only_results, "close_open": close_open_results}
            optimized_results = {"close_only": other_close, "close_open": other_close_open}
            pct_sst_default_vs_opt = -pct_sst
            pct_sst_default_vs_baseline = -pct_sst_vs_base
            pct_sst_opt_vs_opt = pct_other_vs_opt
            pct_sst_opt_vs_baseline = pct_other_vs_base

        plot_mcl_sweep_guide_comparison(
            mcl_values=mcl_values,
            default_results=default_results,
            optimized_results=optimized_results,
            pct_sst_default_vs_opt=pct_sst_default_vs_opt,
            pct_sst_default_vs_baseline=pct_sst_default_vs_baseline,
            pct_sst_opt_vs_opt=pct_sst_opt_vs_opt,
            pct_sst_opt_vs_baseline=pct_sst_opt_vs_baseline,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_guide_comparison.png",
            panel_size=panel_size,
        )

        # ==============================================================
        #  Directional Filtering Comparison
        # ==============================================================
        print("\n" + "-" * 64)
        print("  Directional Filtering Comparison")
        print("-" * 64)

        if DIRECTIONAL_FILTERING:
            dir_close_only = close_only_results
            dir_close_open = close_open_results
            dir_open_only = open_only_results
            dir_open_close = open_close_results
        else:
            iso_close_only = close_only_results
            iso_close_open = close_open_results
            iso_open_only = open_only_results
            iso_open_close = open_close_results

        other_se_directional = not DIRECTIONAL_FILTERING
        other_se_label = "directional" if other_se_directional else "isotropic"
        other_se_prefix = f"{FILTER_ANALYSIS_PREFIX}_{other_se_label}"

        other_se_close = _run_filter_method(
            "close_only", sst, prefix=other_se_prefix, directional=other_se_directional,
        )
        other_se_open = _run_filter_method(
            "open_only", sst, prefix=other_se_prefix, directional=other_se_directional,
        )
        other_se_close_open = _run_filter_method(
            "close_open", sst, prefix=other_se_prefix, directional=other_se_directional,
        )
        other_se_open_close = _run_filter_method(
            "open_close", sst, prefix=other_se_prefix, directional=other_se_directional,
        )

        if DIRECTIONAL_FILTERING:
            iso_close_only = other_se_close
            iso_open_only = other_se_open
            iso_close_open = other_se_close_open
            iso_open_close = other_se_open_close
            dir_dilate_vol = dilate_vol
            dir_erode_vol = erode_vol
            iso_dilate_vol = _run_volume_only("dilate_only", False,
                                                mcl_list=mcl_values_ext)
            iso_erode_vol = _run_volume_only("erode_only", False,
                                              mcl_list=mcl_values_ext)
        else:
            dir_close_only = other_se_close
            dir_open_only = other_se_open
            dir_close_open = other_se_close_open
            dir_open_close = other_se_open_close
            iso_dilate_vol = dilate_vol
            iso_erode_vol = erode_vol
            dir_dilate_vol = _run_volume_only("dilate_only", True,
                                               mcl_list=mcl_values_ext)
            dir_erode_vol = _run_volume_only("erode_only", True,
                                              mcl_list=mcl_values_ext)

        # star_fallback: clip directional performance to iso where worse
        def _fallback(d, i):
            r = copy.deepcopy(d)
            for k in r:
                for idx in range(len(r[k]["pct_vs_opt"])):
                    dp = r[k]["pct_vs_opt"][idx]
                    ip = i[k]["pct_vs_opt"][idx]
                    if not np.isnan(dp) and not np.isnan(ip) and dp < ip:
                        r[k]["pct_vs_opt"][idx] = ip
                    db = r[k]["pct_vs_baseline"][idx]
                    ib = i[k]["pct_vs_baseline"][idx]
                    if not np.isnan(db) and not np.isnan(ib) and db < ib:
                        r[k]["pct_vs_baseline"][idx] = ib
            return r

        if DIRECTIONAL_MODE == "star_fallback":
            dir_close_only = _fallback(dir_close_only, iso_close_only)
            dir_close_open = _fallback(dir_close_open, iso_close_open)
            dir_open_only = _fallback(dir_open_only, iso_open_only)
            dir_open_close = _fallback(dir_open_close, iso_open_close)

        plot_mcl_sweep_directional_comparison(
            mcl_values=mcl_values,
            directional_results={"close_only": dir_close_only, "close_open": dir_close_open},
            isotropic_results={"close_only": iso_close_only, "close_open": iso_close_open},
            pct_sst_vs_opt=-pct_sst,
            pct_sst_vs_baseline=-pct_sst_vs_base,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_directional_comparison.png",
            panel_size=panel_size,
        )

        plot_mcl_sweep_volume_comparison(
            mcl_values=mcl_values,
            directional_deltas={
                "close_only": dir_close_only["vol_delta_pct"],
                "open_only": dir_open_only["vol_delta_pct"],
                "close_open": dir_close_open["vol_delta_pct"],
                "open_close": dir_open_close["vol_delta_pct"],
            },
            isotropic_deltas={
                "close_only": iso_close_only["vol_delta_pct"],
                "open_only": iso_open_only["vol_delta_pct"],
                "close_open": iso_close_open["vol_delta_pct"],
                "open_close": iso_open_close["vol_delta_pct"],
            },
            save_path=f"{FILTER_ANALYSIS_PREFIX}_volume_comparison.png",
            panel_size=panel_size,
        )

        plot_mcl_sweep_volume_dilate_erode(
            mcl_values=mcl_values_ext,
            dilate_deltas=dir_dilate_vol,
            erode_deltas=dir_erode_vol,
            dilate_iso=iso_dilate_vol,
            erode_iso=iso_erode_vol,
            save_path=f"{FILTER_ANALYSIS_PREFIX}_volume_dilate_erode.png",
            panel_size=panel_size,
        )

print("\n  Optimization Completed.")
