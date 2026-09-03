"""
Filter Verification: Morphology filter on a topology optimization case

Applies the same morphology filter used in main.py to a simpler topology
optimization result (element density field, no composites/SST).  Used to
verify the filter's performance against literature benchmarks.

Workflow:
  1. Parse element densities from the topology optimization .sh file
  2. Cluster into 2 zones (binary: void vs. solid) via k-means
  3. Apply morphology filter with configurable radius and method
  4. Export filtered topology as .fem for structural re-evaluation
  5. Run OptiStruct and compare compliance
"""

import os
import numpy as np

from kmeans import k_means1D
from helper import load_mesh_data, parse_total_element_thickness
from filter import (
    build_grid_mapping,
    apply_morphology_filter,
    keep_largest_connected_component_binary,
)
from helper import run_solver, read_results, _mcl_tag

# ==============================================================
# Settings
# ==============================================================

FILE     = "FilterVerificationSigmundNC_2"
FEM_FILE = FILE + ".fem"
SH_FILE  = FILE + ".sh"
OUT_FILE = FILE + ".out"

VERBOSE = False

# Filter radius (mm) — this is the structuring-element disk radius (R_min
# in literature).  The minimum feature size (MCL) = 2 * FILTER_RADIUS.
# Adjust this to match magnitudes used in literature.
FILTER_RADIUS = 3.5    # mm

# Morphological filtering method (same options as main.py):
#   "close_only"  — fill thin holes < 2R.  Material never decreases.
#   "open_only"   — remove thin protrusions < 2R.  Material never increases.
#   "close_open"  — close then open: fill holes AND remove protrusions.
#   "open_close"  — open then close: remove protrusions then fill holes.
#   "dilate_only" — pure dilation.
#   "erode_only"  — pure erosion.
FILTERING_METHOD = "open_close"

# For binary topology verification, branch-preserving regrow can revive small
# detached islands.  Keeping only the largest connected solid component removes
# those artefacts while preserving the main load path.
KEEP_LARGEST_COMPONENT = False

# Edge overshoot (mm): virtually extend the panel boundary before filtering
# so features at the edge are not penalized.  Set to 0 to disable.
EDGE_OVERSHOOT = 0          # FILTER_RADIUS

# Original shell thickness from the PSHELL card (mm)
T_ORIGINAL = 2.0

# Density for void elements (matches MINDENS in the optimizer)
MIN_DENSITY = 0.001

# Target volume fraction (from DCONSTR in the .fem file)
TARGET_VF = 0.5

# Volume-matched scaling: scale solid thickness so total volume equals the
# original optimized volume (VF * panel_volume).  Ensures fair compliance
# comparison by eliminating volume-change effects of the filter.
VOLUME_SCALE = True

# Filter sweep: when True, evaluate rel_obj (filtered/binary compliance)
# across a range of filter radii and plot against literature data.
FILTER_SWEEP  = False
SWEEP_RADII   = [0, 0.31, 0.62, 0.93,
                 1.25, 1.55, 1.875, 2.17,
                 2.5, 2.79, 3.125, 3.41,
                 3.75, 4.03]
# SSWEEP_RADII   = [0, 0.31, 0.62, 0.93,
#                  1.25, 1.55, 1.875, 2.17,
#                  2.5, 2.79, 3.125, 3.41,
#                  3.75, 4.03, 4.375, 4.65, 5]

CORTEZ_RADII = [0, 1.25, 2.5, 3.75, 5]
CORTEZ_REL_OBJ = [1.0000, 1.1154, 1.1679, 1.2179, 1.2431]

HAGG_RADII   = [0, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375]
HAGG_REL_OBJ = [1.000, 1.092, 1.100, 1.106, 1.116, 1.119, 1.125]

OPTISTRUCT_EXE = r"C:\Program Files\Altair\2025.1\hwsolvers\scripts\optistruct.bat"
OPTISTRUCT_AVAILABLE = os.path.exists(OPTISTRUCT_EXE)

import filter as _filter_module
_filter_module.set_verbose(VERBOSE)

if not OPTISTRUCT_AVAILABLE:
    print(f"\n  [INFO] OptiStruct not found at {OPTISTRUCT_EXE}")
    print(f"         Will export .fem files only (no solver re-analysis).\n")


# ==============================================================
# Helper: PSHELL topology .fem export
# ==============================================================

def export_fem_topo(source_fem_path, output_fem_path, elem_zone,
                    t_solid, t_void):
    """
    Write a .fem file for topology re-analysis (PSHELL, no optimization).

    - Removes all optimization cards (DTPL, DRESP1, DCONSTR, DESGLB,
      DESOBJ, DOPTPRM, DCONADD)
    - Creates two PSHELL properties: PID 100 (void) and PID 101 (solid)
    - Updates CQUAD4 PIDs based on the binary zone assignment

    Args:
        source_fem_path: original .fem file (with optimization cards)
        output_fem_path: output .fem file (static analysis)
        elem_zone:       {element_id: 0 (void) or 1 (solid)}
        t_solid:         thickness for solid elements (mm)
        t_void:          thickness for void elements (mm)
    """
    with open(source_fem_path, 'r') as f:
        lines = f.readlines()

    VOID_PID  = 100
    SOLID_PID = 101

    # --- Find original PSHELL to get MID and card template ---
    orig_mid  = 1
    orig_pid  = 1
    orig_raw  = None
    for line in lines:
        s = line.strip()
        if s.startswith("PSHELL"):
            orig_raw = line.rstrip('\r\n').ljust(72)
            try:
                orig_pid = int(orig_raw[8:16].strip())
            except (ValueError, IndexError):
                pass
            try:
                orig_mid = int(orig_raw[16:24].strip())
            except (ValueError, IndexError):
                pass
            break

    def _make_pshell(pid, t):
        """Build a PSHELL line from the original template, swapping PID and T."""
        if orig_raw is not None:
            return (orig_raw[:8]
                    + f"{pid:>8d}"
                    + orig_raw[16:24]
                    + f"{t:>8.4f}"
                    + orig_raw[32:]) + "\n"
        return f"PSHELL  {pid:>8d}{orig_mid:>8d}{t:>8.4f}\n"

    new_pshell_lines = [
        "$$\n",
        "$$  Topology Re-analysis PSHELL Data\n",
        "$$\n",
        _make_pshell(VOID_PID,  t_void),
        _make_pshell(SOLID_PID, t_solid),
        "\n",
    ]

    # --- Element → PID mapping ---
    eid_to_pid = {eid: (SOLID_PID if z >= 1 else VOID_PID)
                  for eid, z in elem_zone.items()}

    # --- Walk source file, removing optim cards and updating PIDs ---
    output_lines   = []
    in_pshell      = False
    pshell_written = False
    skip_cont      = False

    optim_cards = ("DTPL", "DSIZE", "DRESP1", "DCONSTR",
                   "DESGLB", "DCONADD")

    for line in lines:
        raw = line.rstrip('\r\n')

        # --- Skip continuation lines of removed cards ---
        if skip_cont:
            if raw.startswith("+") or (len(raw) > 0 and raw[0] == " "
                                       and raw.strip()):
                continue
            skip_cont = False

        # --- Remove DESOBJ from subcase ---
        if raw.strip().startswith("DESOBJ"):
            continue

        # --- Remove HM optimisation comments ---
        if any(k in raw for k in ("$HMNAME DESVARS",
                                   "$HMNAME OPTICONTROLS",
                                   "$HMNAME OPTICONSTRAINTS")):
            continue

        # --- Remove DOPTPRM ---
        if raw.strip().startswith("DOPTPRM"):
            continue

        # --- Remove optimisation bulk cards ---
        if any(raw.strip().startswith(c) for c in optim_cards):
            skip_cont = True
            continue

        # --- Replace PSHELL block ---
        if raw.strip().startswith("PSHELL"):
            in_pshell = True
            if not pshell_written:
                output_lines.extend(new_pshell_lines)
                pshell_written = True
            continue
        if in_pshell:
            s = raw.strip()
            if s and not s.startswith("$") and \
                    (raw[0] == " " or raw[0] == "+"):
                continue
            else:
                in_pshell = False

        # --- Update CQUAD4 PIDs ---
        if raw.startswith("CQUAD4"):
            padded = raw.ljust(56)
            try:
                eid = int(padded[8:16].strip())
            except ValueError:
                output_lines.append(line)
                continue
            if eid in eid_to_pid:
                new_pid = eid_to_pid[eid]
                output_lines.append(
                    padded[:8] + f"{eid:>8d}" + f"{new_pid:>8d}"
                    + padded[24:] + "\n")
            else:
                output_lines.append(line)
            continue

        # --- Remove original HM property comment / colour for old PID ---
        if "$HMNAME PROP" in raw and f'"property{orig_pid}"' in raw:
            continue
        if "$HWCOLOR PROP" in raw:
            parts = raw.split()
            if len(parts) >= 3 and parts[-2] == str(orig_pid):
                continue

        output_lines.append(line)

    with open(output_fem_path, 'w') as f:
        f.writelines(output_lines)

    n_s = sum(1 for z in elem_zone.values() if z >= 1)
    n_v = len(elem_zone) - n_s
    print(f"    {output_fem_path}:  "
          f"solid={n_s} (PID {SOLID_PID}, T={t_solid:.4f} mm),  "
          f"void={n_v} (PID {VOID_PID}, T={t_void:.6f} mm)")


# ==============================================================
# Step 1: Parse topology densities
# ==============================================================
print("=" * 64)
print("  STEP 1: Parse element densities from topology optimization")
print("=" * 64)

# base_ply_thickness = 1.0 → each element gets its raw density (0..1)
density_data = parse_total_element_thickness(SH_FILE, 1.0)
n_elements   = len(density_data)
densities    = np.array(list(density_data.values()))

print(f"  Elements:     {n_elements}")
print(f"  Density range: [{densities.min():.6f}, {densities.max():.6f}]")
print(f"  Mean density:  {densities.mean():.4f}  "
      f"(target VF = {TARGET_VF})")

# ==============================================================
# Step 2: Parse mesh geometry
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 2: Parse mesh geometry")
print("=" * 64)

X, Y, _ = load_mesh_data(FEM_FILE, density_data)
eids     = list(density_data.keys())
print(f"  {len(eids)} elements loaded from {FEM_FILE}")

# ==============================================================
# Step 3: Cluster into 2 zones (binary: void vs. solid)
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 3: k-means clustering (K = 2, binary topology)")
print("=" * 64)

cluster_data, centers, wcss = k_means1D(density_data, 2)
sorted_centers = sorted(centers)
print(f"  Cluster centres: {sorted_centers[0]:.4f} (void), "
      f"{sorted_centers[1]:.4f} (solid)")

# Build binary zone assignment (0 = void, 1 = solid)
elem_zone = {}
for ci, elements in cluster_data.items():
    rank = 0 if centers[ci] == sorted_centers[0] else 1
    for eid, _ in elements:
        elem_zone[eid] = rank

n_solid = sum(1 for z in elem_zone.values() if z == 1)
n_void  = n_elements - n_solid
print(f"  Solid: {n_solid:>6} ({100*n_solid/n_elements:.1f}%)")
print(f"  Void:  {n_void:>6} ({100*n_void/n_elements:.1f}%)")

# ==============================================================
# Step 4: Build grid and apply morphology filter
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 4: Apply morphology filter")
print("=" * 64)

eid_grid, elem_to_grid, ux, uy = build_grid_mapping(X, Y, eids)
element_size = float(np.median(np.diff(ux)))
print(f"  Grid: {len(ux)} x {len(uy)},  element size: {element_size:.2f} mm")
print(f"  Filter radius (R): {FILTER_RADIUS:.2f} mm  "
      f"({FILTER_RADIUS/element_size:.1f} elements)")
print(f"  MCL (= 2R):        {2*FILTER_RADIUS:.2f} mm")
print(f"  Method:            {FILTERING_METHOD}")
print(f"  Edge overshoot:    {EDGE_OVERSHOOT:.2f} mm")

# Build zone grid (0 = void, 1 = solid)
zone_grid = np.zeros(eid_grid.shape, dtype=int)
for eid, z in elem_zone.items():
    if eid in elem_to_grid:
        iy, ix = elem_to_grid[eid]
        zone_grid[iy, ix] = z

# apply_morphology_filter expects MCL (minimum feature size = diameter)
mcl = 2.0 * FILTER_RADIUS
n_zones = 2

filtered_grid = apply_morphology_filter(
    zone_grid, mcl, element_size, n_zones,
    method=FILTERING_METHOD,
    ply_angles=None,          # isotropic — no directional filtering
    ux=ux, uy=uy,
    edge_overshoot=EDGE_OVERSHOOT,
    verbose=VERBOSE,
)
if KEEP_LARGEST_COMPONENT:
    filtered_grid = keep_largest_connected_component_binary(filtered_grid)

# Build filtered element zone assignment
elem_zone_filtered = {}
for iy in range(filtered_grid.shape[0]):
    for ix in range(filtered_grid.shape[1]):
        eid = eid_grid[iy, ix]
        if eid < 0:
            continue
        elem_zone_filtered[eid] = max(0, min(1, int(filtered_grid[iy, ix])))

n_solid_f = sum(1 for z in elem_zone_filtered.values() if z == 1)
n_void_f  = n_elements - n_solid_f
delta     = n_solid_f - n_solid

print(f"\n  After filtering:")
print(f"    Solid: {n_solid_f:>6} ({100*n_solid_f/n_elements:.1f}%)")
print(f"    Void:  {n_void_f:>6} ({100*n_void_f/n_elements:.1f}%)")
print(f"    Delta: {delta:+d} elements ({100*delta/max(n_solid,1):+.1f}%)")

# ==============================================================
# Step 5: Visualisation
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 5: Visualisation")
print("=" * 64)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
cmap  = mcolors.ListedColormap(["#f0f0f0", "#000000"])
norm  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

panels = [
    (ax1, zone_grid,     "Binary topology (before filter)"),
    (ax2, filtered_grid, f"Filtered (R={FILTER_RADIUS:.1f} mm, {FILTERING_METHOD})"),
]
for ax, grid_data, title in panels:
    ax.pcolormesh(ux, uy, grid_data, cmap=cmap, norm=norm, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

plt.tight_layout()
plot_path = f"{FILE}_topology_comparison.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {plot_path}")

# ==============================================================
# Step 6: Export .fem files
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 6: Export .fem files for re-analysis")
print("=" * 64)

t_void = T_ORIGINAL * MIN_DENSITY
target_volume = TARGET_VF * n_elements * T_ORIGINAL

def _scaled_t(n_solid_elems):
    """Volume-matched solid thickness."""
    if VOLUME_SCALE and n_solid_elems > 0:
        return (target_volume - (n_elements - n_solid_elems) * t_void) / n_solid_elems
    return T_ORIGINAL

# --- Unfiltered binary model ---
binary_jobname = f"{FILE}_binary"
binary_fem     = binary_jobname + ".fem"
t_solid_binary = _scaled_t(n_solid)
print(f"\n  Binary (unfiltered):")
export_fem_topo(FEM_FILE, binary_fem, elem_zone, t_solid_binary, t_void)
if VOLUME_SCALE:
    print(f"    Volume-scaled T_solid = {t_solid_binary:.4f} mm  "
          f"(total vol = {target_volume:.0f} mm²)")

# --- Filtered model ---
r_tag = _mcl_tag(FILTER_RADIUS)
filtered_jobname = f"{FILE}_filtered_R{r_tag}"
filtered_fem     = filtered_jobname + ".fem"
t_solid_filtered = _scaled_t(n_solid_f)
print(f"\n  Filtered:")
export_fem_topo(FEM_FILE, filtered_fem, elem_zone_filtered,
                t_solid_filtered, t_void)
if VOLUME_SCALE:
    print(f"    Volume-scaled T_solid = {t_solid_filtered:.4f} mm  "
          f"(total vol = {target_volume:.0f} mm²)")

# ==============================================================
# Step 7: Run OptiStruct re-analysis
# ==============================================================
print("\n" + "=" * 64)
print("  STEP 7: Run OptiStruct re-analysis")
print("=" * 64)

# --- Read original optimized compliance from the topology .out ---
orig_opt_compliance = None
if os.path.exists(OUT_FILE):
    _, orig_opt_compliance, _ = read_results(
        jobname=FILE, optimize=True,
    )
    if orig_opt_compliance:
        print(f"  Original optimized compliance: {orig_opt_compliance:.4f} Nmm")

# --- Run binary model ---
compliance_binary   = 0.0
compliance_filtered = 0.0

if OPTISTRUCT_AVAILABLE:
    print(f"\n  Running binary model ({binary_fem})...")
    run_solver(filename=binary_fem)
    compliance_binary, _ = read_results(jobname=binary_jobname)
    print(f"  Binary compliance: {compliance_binary:.4f} Nmm")

    print(f"\n  Running filtered model ({filtered_fem})...")
    run_solver(filename=filtered_fem)
    compliance_filtered, _ = read_results(jobname=filtered_jobname)
    print(f"  Filtered compliance: {compliance_filtered:.4f} Nmm")
else:
    print("  [SKIP] OptiStruct not available.")
    if os.path.exists(binary_jobname + ".out"):
        compliance_binary, _ = read_results(jobname=binary_jobname)
        print(f"  Binary compliance (cached): {compliance_binary:.4f} Nmm")
    if os.path.exists(filtered_jobname + ".out"):
        compliance_filtered, _ = read_results(jobname=filtered_jobname)
        print(f"  Filtered compliance (cached): {compliance_filtered:.4f} Nmm")

# ==============================================================
# Summary
# ==============================================================
print("\n" + "=" * 64)
print("  SUMMARY")
print("=" * 64)

mode_str = "volume-matched" if VOLUME_SCALE else "fixed-thickness"
print(f"\n  Mode: {mode_str}")
if VOLUME_SCALE:
    print(f"  Target volume: {target_volume:.0f} mm²  (VF={TARGET_VF}, "
          f"T_orig={T_ORIGINAL:.1f} mm)")

print(f"\n  {'Model':<40} {'Compliance':>12} {'Solid':>8} {'T_solid':>10}")
print(f"  {'-'*40} {'-'*12} {'-'*8} {'-'*10}")
if orig_opt_compliance:
    print(f"  {'Original optimized (continuous)':<40} {orig_opt_compliance:>12.4f} "
          f"{'—':>8} {'—':>10}")
print(f"  {'Binary (unfiltered)':<40} {compliance_binary:>12.4f} "
      f"{n_solid:>8} {t_solid_binary:>10.4f}")
print(f"  {f'Filtered (R={FILTER_RADIUS:.1f}mm, {FILTERING_METHOD})':<40} "
      f"{compliance_filtered:>12.4f} {n_solid_f:>8} {t_solid_filtered:>10.4f}")

if compliance_binary > 0 and compliance_filtered > 0:
    delta_pct = (compliance_filtered - compliance_binary) / compliance_binary * 100
    print(f"\n  Compliance change due to filter: {delta_pct:+.2f}%")

if orig_opt_compliance and compliance_binary > 0:
    gap = (compliance_binary - orig_opt_compliance) / orig_opt_compliance * 100
    print(f"  Binary vs optimized:              {gap:+.2f}%")

if orig_opt_compliance and compliance_filtered > 0:
    gap = (compliance_filtered - orig_opt_compliance) / orig_opt_compliance * 100
    print(f"  Filtered vs optimized:            {gap:+.2f}%")

rel_obj = compliance_filtered / compliance_binary if compliance_binary > 0 else 0.0
print(f"  RelObj:            {rel_obj:.4f}")

# ==============================================================
# Step 8: Filter sweep (optional)
# ==============================================================
if FILTER_SWEEP:
    print("\n" + "=" * 64)
    print("  STEP 8: Filter sweep")
    print("=" * 64)

    sweep_computed = []

    for r in SWEEP_RADII:
        print(f"\n  --- R = {r:.2f} mm ---")

        if r == 0:
            sweep_computed.append(1.0)
            print(f"    No filter: rel_obj = 1.0000")
            continue

        mcl_r = 2.0 * r
        fg = apply_morphology_filter(
            zone_grid, mcl_r, element_size, n_zones,
            method=FILTERING_METHOD,
            ply_angles=None,
            ux=ux, uy=uy,
            edge_overshoot=EDGE_OVERSHOOT,
            verbose=VERBOSE,
        )
        if KEEP_LARGEST_COMPONENT:
            fg = keep_largest_connected_component_binary(fg)

        ez = {}
        for iy in range(fg.shape[0]):
            for ix in range(fg.shape[1]):
                eid = eid_grid[iy, ix]
                if eid < 0:
                    continue
                ez[eid] = max(0, min(1, int(fg[iy, ix])))

        n_solid_r  = sum(1 for z in ez.values() if z == 1)
        t_solid_r  = _scaled_t(n_solid_r)
        r_tag      = _mcl_tag(r)
        jobname    = f"{FILE}_filtered_R{r_tag}"
        fem_path   = jobname + ".fem"

        export_fem_topo(FEM_FILE, fem_path, ez, t_solid_r, t_void)

        if OPTISTRUCT_AVAILABLE:
            run_solver(filename=fem_path)
            comp_r, _ = read_results(jobname=jobname)
        elif os.path.exists(jobname + ".out"):
            comp_r, _ = read_results(jobname=jobname)
        else:
            comp_r = 0.0

        rel_obj_r = comp_r / compliance_binary if compliance_binary > 0 else 0.0
        sweep_computed.append(rel_obj_r)
        print(f"    Solid: {n_solid_r}, compliance: {comp_r:.4f}, "
              f"rel_obj: {rel_obj_r:.4f}")

    print(f"\n  {'Radius (mm)':>12} {'Computed':>10} {'Cortez':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for i, r in enumerate(CORTEZ_RADII):
        if sweep_computed[i] and CORTEZ_REL_OBJ[i]:
            print(f"  {r:>12.2f} {sweep_computed[i]:>10.4f} "
                  f"{CORTEZ_REL_OBJ[i]:>10.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SWEEP_RADII, sweep_computed, 'o-', color='#2166ac',
            linewidth=2, markersize=7, label='Computed')
    ax.plot(CORTEZ_RADII, CORTEZ_REL_OBJ, 's--', color='#d6604d',
            linewidth=2, markersize=7, label='Cortez et al. (TOBS)')
    ax.plot(HAGG_RADII, HAGG_REL_OBJ, 's--', color='#ff69b4',
            linewidth=2, markersize=7, label='Hagg & Wadbro (SIMP*)')
    ax.set_xlabel('Filter radius $R$ (mm)', fontsize=12)
    ax.set_ylabel(
        'RelObj',
        fontsize=12)
    ax.set_title(f'Filter Sweep', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylim(bottom=0.95)
    plt.tight_layout()
    sweep_path = f"{FILE}_filter_sweep.png"
    plt.savefig(sweep_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {sweep_path}")

print("\n  Done.")
