"""
Stacking Sequence Table (SST) construction from Free-Size Optimization results

Builds an SST from FSO per-ply thickness data (Irisarri et al. 2014), enforcing
symmetry, balance, contiguity, disorientation, damtol, covering and internal
continuity by construction. When the layup has too few plies for the requested
number of zones, the guide laminate is expanded by splitting plies into sub-plies
down to the minimum manufacturable thickness.

Ref: Irisarri et al., Composite Structures 107 (2014) 559-569.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from itertools import permutations

_VERBOSE = True


def set_verbose(v: bool):
    global _VERBOSE
    _VERBOSE = v


# ==============================================================================
#  Data structures
# ==============================================================================

@dataclass
class Ply:
    angle: float
    thickness: float
    is_continuous: bool
    drop_rank: Optional[int]
    half_idx: int


@dataclass
class StackingSequenceTable:
    half_plies: list[Ply]
    n_min: int
    n_max: int
    ply_angles: list[float]
    zone_ply_counts: list[int]
    drop_sequence: list[tuple[int, float]]
    base_ply_thickness: float
    manufacturing_ply_thickness: float
    doubling_factor: int
    guideline_report: dict = field(default_factory=dict)
    # Every guideline-valid ply count this guide stack / drop order can reach
    # (before narrowing to the zones actually used). Lets a caller re-select
    # zone levels for a different FSO field without rebuilding the laminate.
    valid_ply_counts: list[int] = field(default_factory=list)

    @property
    def n_half(self) -> int:
        return len(self.half_plies)

    def get_laminate(self, n_plies: int, n_max_override: int = None) -> list[float]:
        # Use n_max_override for scaling ply thickness scenarios (guide ply baseline)
        # Otherwise use self.n_max for normal SST zone construction
        n_max = n_max_override if n_max_override is not None else self.n_max
        # The FULL guide laminate (n_plies == n_max) is always well-defined,
        # even when zone_ply_counts was later narrowed to the zones actually
        # used by the design (which may not include the thickest zone).
        if n_plies != n_max and n_plies not in self.zone_ply_counts:
            raise ValueError(f"n_plies={n_plies} not valid. Choose from {self.zone_ply_counts}")
        # For full laminate (n_plies == n_max), include all plies regardless of drop_rank
        if n_plies == n_max:
            n_drops = 0
        else:
            n_drops = (n_max - n_plies) // 2
        active = [p.angle for p in self.half_plies
                  if p.is_continuous or (p.drop_rank is not None and p.drop_rank > n_drops)]
        return active + active[::-1]

    def get_all_laminates(self) -> dict[int, list[float]]:
        return {n: self.get_laminate(n) for n in self.zone_ply_counts}

    def print_table(self):
        if not _VERBOSE:
            return
        counts = self.zone_ply_counts
        header = "  ".join(f"{n:>4}" for n in counts)
        rule = "  " + "-" * (6 * len(counts) + 14)
        print(f"\n{'SST (half-stack, outer -> mid-plane)':>50}")
        print(f"{'Ply counts:':>14} {header}")
        print(rule)
        for p in self.half_plies:
            row = []
            for n in counts:
                nd = (self.n_max - n) // 2
                if p.is_continuous or (p.drop_rank is not None and p.drop_rank > nd):
                    row.append(f"{p.angle:>4.0f}")
                else:
                    row.append("   .")
            tag = "C" if p.is_continuous else f"D{p.drop_rank}"
            print(f"  [{tag:>3}]  " + "  ".join(row))
        print(rule)
        print(f"{'(mid-plane / symmetric mirror below)':>50}\n")

    def summarize(self):
        print("=" * 64)
        print("  STACKING SEQUENCE TABLE - SUMMARY")
        print("=" * 64)
        print(f"\n  Guide laminate (n_max)     : {self.n_max} plies")
        print(f"  Thinnest laminate (n_min)  : {self.n_min} plies")
        print(f"  Valid ply counts           : {self.zone_ply_counts}")
        print(f"  Original ply thickness     : {self.base_ply_thickness} mm")
        print(f"  Manufacturing ply thickness: {self.manufacturing_ply_thickness} mm")
        print(f"  Doubling factor            : {self.doubling_factor}x")
        print(f"\n  Drop sequence (guide -> thinnest):")
        for rank, (hidx, angle) in enumerate(self.drop_sequence, 1):
            print(f"    Drop {rank}: half-ply {hidx+1} ({angle:+.0f}deg)")
        self.print_table()
        print("  Laminates per zone:")
        for n in self.zone_ply_counts:
            lam = self.get_laminate(n)
            t_mm = len(lam) * self.manufacturing_ply_thickness
            print(f"    {n:>2} plies ({t_mm:.2f} mm): {lam}")
        if self.guideline_report:
            print("\n  Design guideline compliance:")
            for name, passed in self.guideline_report.items():
                marker = "PASS" if passed else "FAIL"
                print(f"    [{marker}] {name}")


# ==============================================================================
#  Laminate constraint checks
# ==============================================================================

def _angle_diff(a: float, b: float) -> float:
    """Minimum angular distance between two fibre directions (period = 180 deg)."""
    return min(abs(a - b), 180.0 - abs(a - b))


def _check_disorientation(seq: list[float], max_diff: float = 45.0) -> bool:
    for i in range(len(seq) - 1):
        if _angle_diff(seq[i], seq[i + 1]) > max_diff:
            return False
    return True


def _check_contiguity(seq: list[float], max_contig: int = 2) -> bool:
    for i in range(len(seq) - max_contig):
        if len(set(seq[i: i + max_contig + 1])) == 1:
            return False
    return True


def _check_10pct(full_seq: list[float], required_angles: list[float]) -> bool:
    n = len(full_seq)
    if n == 0:
        return True
    min_count = max(1, int(np.ceil(n * 0.1)))
    counts = {}
    for a in full_seq:
        counts[a] = counts.get(a, 0) + 1
    for angle in counts:
        if counts[angle] < min_count:
            return False
    return True


def _check_internal_continuity(present_flags: list[bool],
                               max_consecutive_drops: int = 3) -> bool:
    """Internal continuity (guideline 10): no more than max_consecutive_drops
    dropped plies in a row without a surviving one. present_flags is the
    half-stack survival mask; checked on the full symmetric stack so runs that
    merge across the mid-plane count correctly."""
    full = list(present_flags) + list(reversed(present_flags))
    run = 0
    for present in full:
        if present:
            run = 0
        else:
            run += 1
            if run > max_consecutive_drops:
                return False
    return True


def _full_stack_from_indices(half_angles: list[float], present: set[int]) -> list[float]:
    """Build the full symmetric stack from the set of present half-ply indices."""
    h = [half_angles[i] for i in sorted(present)]
    return h + h[::-1]


# ==============================================================================
#  Guide laminate expansion (ply doubling)
# ==============================================================================

def _build_angle_chain(angles: list[float], max_diff: float) -> list[float]:
    """Order angles so consecutive ones differ by <= max_diff, starting from a
    non-0 endpoint where possible (gives a non-0 outer ply, damtol)."""
    if len(angles) <= 1:
        return angles
    adj = {a: [b for b in angles if b != a and _angle_diff(a, b) <= max_diff] for a in angles}
    endpoints = [a for a in angles if len(adj[a]) <= 1]
    if endpoints:
        non_zero = [e for e in endpoints if e != 0.0]
        start = non_zero[0] if non_zero else endpoints[0]
    else:
        non_zero = [a for a in angles if a != 0.0]
        start = non_zero[0] if non_zero else angles[0]
    chain, visited, cur = [start], {start}, start
    while len(chain) < len(angles):
        nbs = [n for n in adj[cur] if n not in visited]
        if not nbs:
            break
        nxt = nbs[0]
        chain.append(nxt)
        visited.add(nxt)
        cur = nxt
    return chain


def _generate_all_chains(angles: list[float], max_diff: float) -> list[list[float]]:
    """All valid angle orderings (Hamiltonian paths) where consecutive angles
    differ by <= max_diff, excluding those starting with 0 (damtol). Sorted for
    reproducibility. For {0,45,-45,90} at 45deg this gives 6 chains."""
    if len(angles) <= 1:
        return [list(angles)]
    adj = {a: [b for b in angles if b != a and _angle_diff(a, b) <= max_diff] for a in angles}

    chains = []

    def _dfs(path, visited):
        if len(path) == len(angles):
            chains.append(list(path))
            return
        cur = path[-1]
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                _dfs(path, visited)
                path.pop()
                visited.discard(nxt)

    for start in sorted(angles):
        if start == 0.0:
            continue
        _dfs([start], {start})

    return sorted(chains, key=lambda c: tuple(c))


def _compute_D_trace(half_angles: list[float], ply_thickness: float) -> float:
    """trace(D) of the CLT bending stiffness matrix for the symmetric laminate
    implied by the outer-to-mid-plane half-stack. Larger = stiffer in bending."""
    E1, E2, G12, nu12 = 141e3, 9.03e3, 4.27e3, 0.32
    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12

    n_half = len(half_angles)
    D = np.zeros((3, 3))
    for i, angle in enumerate(half_angles):
        th = np.radians(angle)
        c, s = np.cos(th), np.sin(th)
        c2, s2, cs = c * c, s * s, c * s

        Qb11 = Q11 * c2**2 + 2 * (Q12 + 2 * Q66) * c2 * s2 + Q22 * s2**2
        Qb22 = Q11 * s2**2 + 2 * (Q12 + 2 * Q66) * c2 * s2 + Q22 * c2**2
        Qb12 = (Q11 + Q22 - 4 * Q66) * c2 * s2 + Q12 * (c2**2 + s2**2)
        Qb66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * c2 * s2 + Q66 * (c2**2 + s2**2)

        z_top = (n_half - i) * ply_thickness
        z_bot = (n_half - i - 1) * ply_thickness
        dz3 = (z_top**3 - z_bot**3) / 3.0

        D[0, 0] += Qb11 * dz3
        D[1, 1] += Qb22 * dz3
        D[2, 2] += Qb66 * dz3

    return D[0, 0] + D[1, 1] + D[2, 2]


def _compute_xi_from_angles_and_thicknesses(
    angles_deg: list[float],
    thicknesses: list[float],
) -> np.ndarray:
    """Bending lamination parameters [xi1_D..xi4_D] (dimensionless, in [-1,1]),
    integrating the stack through-thickness from -t_tot/2 to +t_tot/2."""
    angles = np.deg2rad(angles_deg)
    t_tot = sum(thicknesses)
    if t_tot <= 0.0:
        return np.zeros(4)
    z = -0.5 * t_tot
    xi = np.zeros(4)
    for k in range(len(angles)):
        z_next = z + thicknesses[k]
        dz3 = (z_next ** 3 - z ** 3) / 3.0
        a = angles[k]
        xi += dz3 * np.array([
            np.cos(2.0 * a), np.sin(2.0 * a),
            np.cos(4.0 * a), np.sin(4.0 * a),
        ])
        z = z_next
    return 12.0 * xi / (t_tot ** 3)


def _score_candidate_xi_mismatch(
    half_stack: list[float],
    orig_half_angles: list[float],
    orig_imp: np.ndarray,
    position_bias: float,
    covering_plies: int,
    max_contiguity: int,
    max_disorientation: float,
    n_zones: int,
    fso_xi: dict[int, np.ndarray],
    cluster_data: dict,
    cluster_centers: list[float],
    mfg_t: float,
    tolerant_ladder: bool = False,
    min_plies: int = 0,
) -> float:
    """Score a guide-stack candidate on how well its SST zone laminates match
    the FSO xi^D field (L2 norm over elements). Returns (-avg_mismatch,
    ladder_quality): higher score = better. ladder_quality = (usable counts,
    -largest gap) over counts the design can actually reach (>= min_plies)."""
    n_half = len(half_stack)
    n_max = n_half * 2

    # --- Expanded importance ---
    aim_mean = {}
    for a, im in zip(orig_half_angles, orig_imp):
        aim_mean.setdefault(a, []).append(im)
    aim_mean = {a: np.mean(v) for a, v in aim_mean.items()}

    exp_imp = np.zeros(n_half)
    for i, angle in enumerate(half_stack):
        pos = i / max(n_half - 1, 1)
        exp_imp[i] = aim_mean[angle] * (1.0 + position_bias * (1.0 - pos))

    # --- Drop order ---
    drop_order, forced_cont, extra_cont = determine_drop_order(
        half_stack, exp_imp, covering_plies,
        max_contiguity, max_disorientation, quiet=True,
        tolerant_ladder=tolerant_ladder,
    )

    # --- Build Ply objects ---
    plies = []
    for i in range(n_half):
        if i in drop_order:
            rank = drop_order.index(i) + 1
            is_cont = False
        else:
            rank = None
            is_cont = True
        plies.append(Ply(half_stack[i], 1.0, is_cont, rank, i))

    # --- Valid ply counts ---
    counts = [n_max]
    current = n_max
    i = 0
    while i < len(drop_order):
        idx = drop_order[i]
        a = half_stack[idx]
        if a not in (0.0, 90.0) and (i + 1 < len(drop_order)):
            j = drop_order[i + 1]
            if half_stack[j] == -a:
                current -= 4
                i += 2
                counts.append(current)
                continue
        current -= 2
        i += 1
        counts.append(current)

    all_counts = sorted(set(counts))

    # Ladder quality: only counts the design can actually reach matter, i.e.
    # those at or above the FSO's own minimum. Counting every valid count
    # rewarded stacks with a pile of unusable sub-floor levels and a hole in
    # the middle of the usable band, so a candidate could "win" on raw count
    # while leaving nothing to discretise the thin half of the panel with.
    # Rank on (usable count, -largest gap) so coverage beats quantity.
    _valid_all = _compute_valid_ply_counts(
        n_max, drop_order, half_stack,
        half_plies=plies, max_contiguity=max_contiguity,
        max_disorientation=max_disorientation,
        required_angles=sorted(set(half_stack)),
    )
    _usable = [c for c in sorted(_valid_all) if c >= min_plies]
    if len(_usable) >= 2:
        _max_gap = max(_usable[i + 1] - _usable[i]
                       for i in range(len(_usable) - 1))
        # Reaching the floor matters too: a ladder starting well above the
        # FSO minimum cannot represent the thin zones at all.
        _max_gap = max(_max_gap, _usable[0] - min_plies)
    else:
        _max_gap = n_max
    n_valid = (len(_usable), -_max_gap)

    # --- Select zone ply counts ---
    if n_zones >= len(all_counts):
        zone_counts = all_counts
    else:
        indices = np.round(np.linspace(0, len(all_counts) - 1, n_zones)).astype(int)
        zone_counts = sorted(set(all_counts[i] for i in indices))

    # --- Compute xi^D for each SST zone ---
    zone_xi = {}
    for n in zone_counts:
        nd = (n_max - n) // 2
        active = [p.angle for p in plies
                  if p.is_continuous or (p.drop_rank is not None and p.drop_rank > nd)]
        full_angles_deg = active + active[::-1]
        full_thicknesses = [mfg_t] * len(full_angles_deg)
        zone_xi[n] = _compute_xi_from_angles_and_thicknesses(
            full_angles_deg, full_thicknesses,
        )

    # --- Map cluster centres to zone ply counts ---
    if cluster_data and cluster_centers:
        sorted_ci = sorted(range(len(cluster_centers)),
                           key=lambda i: cluster_centers[i])
        cluster_to_zone_idx = {
            int(ci): min(j, len(zone_counts) - 1)
            for j, ci in enumerate(sorted_ci)
        }
        eid_to_zone = {}
        for ci, elements in cluster_data.items():
            ci_int = int(ci)
            if ci_int not in cluster_to_zone_idx:
                cluster_to_zone_idx[ci_int] = 0
            zi = cluster_to_zone_idx[ci_int]
            for eid, _ in elements:
                eid_to_zone[eid] = zi

        # --- Sum L2 mismatch across all elements ---
        total_mismatch = 0.0
        n_elements = 0
        for eid, fso_val in fso_xi.items():
            if eid not in eid_to_zone:
                continue
            zi = eid_to_zone[eid]
            zone_n = zone_counts[zi]
            sst_val = zone_xi[zone_n]
            diff = np.array(fso_val) - np.array(sst_val)
            total_mismatch += np.sqrt(np.dot(diff, diff))
            n_elements += 1

        if n_elements == 0:
            return 0.0, n_valid
        avg_mismatch = total_mismatch / n_elements
    else:
        # No cluster data: compare FSO average xi^D against each zone, unweighted
        fso_mean = np.zeros(4)
        if fso_xi:
            fso_mean = np.mean(np.array(list(fso_xi.values())), axis=0)

        avg_mismatch = 0.0
        for n in zone_counts:
            sst_val = zone_xi[n]
            diff = fso_mean - sst_val
            avg_mismatch += np.sqrt(np.dot(diff, diff))
        avg_mismatch /= max(len(zone_counts), 1)
    # (xi score, number of guideline-valid ladder counts)
    return -avg_mismatch, n_valid  # higher score = better


# Guide-stack search budget. generate_candidate_half_stacks enumerates
# 2^doubling_factor forward/backward patterns, so the candidate pool grows as
# 3 x 2^DF: 768 at DF 8, but 196,608 at DF 16, where scoring every one costs
# tens of hours and dominates the whole run. The pool is sorted by descending
# trace(D), so truncating it keeps the bending-stiffest candidates and holds the
# search budget constant across ply thicknesses, which makes a ply-thickness
# sweep a comparison of ply thickness rather than of search effort.
# 0 disables the cap.
MAX_GUIDE_CANDIDATES = 768


def generate_candidate_half_stacks(
    original_half_angles: list[float],
    doubling_factor: int,
    max_contiguity: int = 2,
    max_disorientation: float = 45.0,
    optimize: bool = False,
) -> list[tuple[list[float], float]]:
    """Candidate guide half-stacks as (half_stack, trace_D), sorted by
    descending trace(D). All satisfy disorientation/contiguity/damtol by
    construction. optimize=False returns just the default alternating sweep;
    optimize=True enumerates every valid chain x 2^DF forward/backward
    pattern and puts the stiffest first."""
    unique = sorted(set(original_half_angles))

    if optimize:
        chains = _generate_all_chains(unique, max_disorientation)
    else:
        chains = [_build_angle_chain(unique, max_disorientation)]

    n_patterns = 2 ** doubling_factor if optimize else 1

    candidates = []
    seen = set()
    for chain in chains:
        forward = list(chain)
        backward = list(chain[::-1])
        for pat_int in range(n_patterns):
            half = []
            for rep in range(doubling_factor):
                half.extend(forward if (pat_int >> rep) & 1 == 0 else backward)

            if half[0] == 0.0:
                continue
            full = half + half[::-1]
            if not _check_disorientation(half, max_disorientation):
                continue
            if not _check_contiguity(full, max_contiguity):
                continue

            key = tuple(half)
            if key in seen:
                continue
            seen.add(key)
            score = _compute_D_trace(half, 1.0)
            candidates.append((half, score))

    if not candidates:
        print("  WARNING: No damtol-valid candidate found. Using default chain.")
        default_chain = _build_angle_chain(unique, max_disorientation)
        half = []
        for rep in range(doubling_factor):
            half.extend(default_chain if rep % 2 == 0 else default_chain[::-1])
        score = _compute_D_trace(half, 1.0)
        candidates.append((half, score))

    candidates.sort(key=lambda x: -x[1])
    if MAX_GUIDE_CANDIDATES and len(candidates) > MAX_GUIDE_CANDIDATES:
        if _VERBOSE:
            print(f"  Guide stack pool: {len(candidates):,} candidates "
                  f"(3 x 2^DF) truncated to the {MAX_GUIDE_CANDIDATES:,} "
                  f"stiffest by trace(D)")
        candidates = candidates[:MAX_GUIDE_CANDIDATES]
    return candidates


# ==============================================================================
#  FSO importance
# ==============================================================================

def compute_ply_importance_global(
    elem_ply_thicknesses: dict[int, list[float]],
    base_ply_thickness: float,
) -> np.ndarray:
    """Global average retention: mean FSO retention per half-ply across all
    elements equally. Robust default for a smooth thickness field."""
    vals = np.array(list(elem_ply_thicknesses.values()))
    n_plies = vals.shape[1]
    n_half = n_plies // 2
    imp = np.zeros(n_half)
    for i in range(n_half):
        imp[i] = ((vals[:, i] + vals[:, n_plies - 1 - i]) / 2.0).mean() / base_ply_thickness
    return imp


def compute_ply_importance_zone_aware(
    elem_ply_thicknesses: dict[int, list[float]],
    base_ply_thickness: float,
    cluster_data: dict,
    cluster_centers: list[float],
) -> np.ndarray:
    """Zone-aware retention: per-ply retention computed per thickness zone, then
    combined with inverse-thickness weighting (w = 1/zone_center, normalised)
    so the thinnest zones — where plies actually drop — dominate the ranking.
    Captures spatial variation a global average would hide."""
    vals_all = np.array(list(elem_ply_thicknesses.values()))
    eids_all = np.array(list(elem_ply_thicknesses.keys()))
    n_plies = vals_all.shape[1]
    n_half = n_plies // 2

    # Build per-zone retention arrays
    n_zones = len(cluster_centers)
    zone_retention = np.zeros((n_zones, n_half))
    zone_centers = np.zeros(n_zones)
    eid_set_all = {eid: idx for idx, eid in enumerate(eids_all)}

    for ci, elements in cluster_data.items():
        indices = [eid_set_all[eid] for eid, _ in elements if eid in eid_set_all]
        if not indices:
            continue
        cluster_vals = vals_all[indices]
        zone_centers[ci] = cluster_centers[ci]

        for i in range(n_half):
            mirror = n_plies - 1 - i
            zone_retention[ci, i] = (
                (cluster_vals[:, i] + cluster_vals[:, mirror]) / 2.0
            ).mean() / base_ply_thickness

    # Inverse-thickness weighting
    weights = np.where(zone_centers > 0, 1.0 / zone_centers, 0.0)
    weights /= weights.sum() if weights.sum() > 0 else 1.0

    # Weighted importance
    imp = np.zeros(n_half)
    for ci in range(n_zones):
        imp += weights[ci] * zone_retention[ci]

    return imp, zone_retention, zone_centers, weights


def compute_expanded_importance(
    original_half_angles: list[float],
    original_half_importance: np.ndarray,
    expanded_half_stack: list[float],
    position_bias: float = 0.05,
) -> np.ndarray:
    """Map per-angle FSO importance onto the expanded half-stack with a
    positional bias: inner plies get slightly lower importance -> dropped first."""
    aim = {}
    for a, im in zip(original_half_angles, original_half_importance):
        aim.setdefault(a, []).append(im)
    aim_mean = {a: np.mean(v) for a, v in aim.items()}

    n = len(expanded_half_stack)
    importance = np.zeros(n)
    for i, angle in enumerate(expanded_half_stack):
        pos = i / max(n - 1, 1)
        importance[i] = aim_mean[angle] * (1.0 + position_bias * (1.0 - pos))
    return importance


# ==============================================================================
#  SB-cycle identification and drop ordering
# ==============================================================================

def _identify_sb_cycles(
    half_angles: list[float],
    ply_importance: np.ndarray,
    covering_plies: int,
) -> tuple[list[int], list[tuple[float, list[int]]]]:
    """Identify droppable SB-cycles and the plies forced continuous for balance.
    Returns (forced_continuous, cycles) where cycles is
    [(mean_importance, [indices])] per droppable symmetry/balance cycle."""
    n = len(half_angles)
    droppable = np.ones(n, dtype=bool)
    for i in range(min(covering_plies, n)):
        droppable[i] = False

    # Count droppable plies per signed angle (only +-theta, not 0/90)
    angle_indices = defaultdict(list)
    for i in range(n):
        if droppable[i] and half_angles[i] not in (0.0, 90.0):
            angle_indices[half_angles[i]].append(i)

    # Balance: ensure equal +theta/-theta droppable counts per |theta|
    forced_continuous = []
    theta_groups = defaultdict(lambda: {"pos": [], "neg": []})
    for a, idxs in angle_indices.items():
        key = abs(a)
        side = "pos" if a > 0 else "neg"
        theta_groups[key][side].extend(idxs)

    for theta, groups in theta_groups.items():
        pos = sorted(groups["pos"], key=lambda i: -ply_importance[i])
        neg = sorted(groups["neg"], key=lambda i: -ply_importance[i])
        n_pairs = min(len(pos), len(neg))
        for idx in pos[n_pairs:]:
            droppable[idx] = False
            forced_continuous.append(idx)
        for idx in neg[n_pairs:]:
            droppable[idx] = False
            forced_continuous.append(idx)

    # Pair +theta/-theta among droppable plies
    paired = {}
    used = set()
    for i in reversed(range(n)):
        if i in used or not droppable[i] or half_angles[i] in (0.0, 90.0):
            continue
        best_j, best_d = None, float("inf")
        for j in range(n):
            if j in used or j == i or not droppable[j]:
                continue
            if half_angles[j] == -half_angles[i] and abs(i - j) < best_d:
                best_d = abs(i - j)
                best_j = j
        if best_j is not None:
            paired[i] = best_j
            paired[best_j] = i
            used.update([i, best_j])

    # Build SB-cycles
    cycles = []
    processed = set()
    for i in range(n):
        if not droppable[i] or i in processed:
            continue
        if i in paired:
            j = paired[i]
            imp = (ply_importance[i] + ply_importance[j]) / 2.0
            cycles.append((imp, sorted([i, j])))
            processed.update([i, j])
        else:
            cycles.append((ply_importance[i], [i]))
            processed.add(i)

    return forced_continuous, cycles


def _find_disorientation_feasible_order(
    half_angles: list[float],
    cycles: list[tuple[float, list[int]]],
    continuous_indices: set[int],
    max_contiguity: int,
    max_disorientation: float,
    required_angles: list[float] = None,
    target_ply_counts: list[int] = None,
    tolerant: bool = False,
) -> list[tuple[float, list[int]]]:
    """SB-cycle drop ordering giving valid intermediate laminates
    (disorientation + contiguity + 10% rule), best matching FSO importance
    (least important first). Validates every step, or only target_ply_counts
    if given.

    tolerant=True replaces the all-or-nothing veto with a lexicographic score
    (fewest infeasible levels, then importance mismatch). Some bottom levels are
    inherently infeasible (a 4-ply balanced symmetric laminate is always
    [+45/-45]s, a 90deg jump), and under the strict rule that one doomed level
    vetoes every ordering the search always fell back to pure importance order,
    which drops a +-45 pair first and poisons all counts. Tolerant keeps the
    ordering with the fewest bad levels (excluded downstream), recovering
    Irisarri-style 2-ply transitions via self-balancing 0/90 drops."""
    n_cycles = len(cycles)
    all_indices = set(range(len(half_angles)))
    best_order = None
    best_score = float("inf")
    best_bad = None
    best_ic = None

    if required_angles is None:
        required_angles = sorted(set(half_angles))

    sorted_by_imp = sorted(range(n_cycles), key=lambda k: cycles[k][0])
    ideal_rank = {sorted_by_imp[r]: r for r in range(n_cycles)}

    target_set = set(target_ply_counts) if target_ply_counts is not None else None
    n_max = len(half_angles) * 2
    n_half = len(half_angles)

    for perm in permutations(range(n_cycles)):
        present = set(all_indices)
        feasible = True
        n_bad = 0
        n_ic_bad = 0
        n_current = n_max

        for cycle_idx in perm:
            _, idxs = cycles[cycle_idx]
            for idx in idxs:
                present.discard(idx)
            n_current -= 2 * len(idxs)

            if target_set is not None:
                if n_current not in target_set:
                    continue
            lam = _full_stack_from_indices(half_angles, present)
            if not _check_disorientation(lam, max_disorientation) or \
               not _check_contiguity(lam, max_contiguity) or \
               not _check_10pct(lam, required_angles):
                if tolerant:
                    n_bad += 1
                else:
                    feasible = False
                    break
            # Internal continuity scored softly in both modes: at high
            # guide/thin ratios some drop runs exceed 3 for every ordering, so
            # a hard veto would just resurrect the importance-fallback failure.
            if not _check_internal_continuity(
                    [i in present for i in range(n_half)]):
                n_ic_bad += 1

        if feasible:
            score = sum(abs(pos - ideal_rank[cycle_idx])
                        for pos, cycle_idx in enumerate(perm))
            if (n_bad, n_ic_bad, score) < (
                    (best_bad, best_ic, best_score)
                    if best_bad is not None
                    else (float("inf"), float("inf"), float("inf"))):
                best_bad = n_bad
                best_ic = n_ic_bad
                best_score = score
                best_order = perm

    if best_order is None:
        print("  WARNING: No fully feasible ordering found "
              "(disorientation + contiguity + 10% rule).")
        if _VERBOSE:
            print("           Falling back to importance-based ordering.")
        return sorted(cycles, key=lambda x: x[0])

    if tolerant and best_bad and _VERBOSE:
        print(f"  Tolerant ladder: best ordering carries {best_bad} "
              f"infeasible intermediate count(s) (excluded from the valid "
              f"zone counts).")
    if best_ic and _VERBOSE:
        print(f"  Internal continuity: best ordering carries {best_ic} "
              f"intermediate count(s) with >3 consecutive dropped plies "
              f"(guideline 10, scored softly).")

    return [cycles[i] for i in best_order]


def _greedy_feasible_cycle_order(
    half_angles: list[float],
    cycles: list,
    max_contiguity: int,
    max_disorientation: float,
    required_angles: list[float],
    check_10pct: bool = True,
) -> list:
    """Order SB-cycles by repeatedly dropping the least important cycle whose
    removal still leaves a guideline-valid laminate.

    Sorting by importance first and validating afterwards does not work:
    feasibility is path dependent (whether a drop is legal depends on what has
    already gone), so a globally importance-sorted sequence goes illegal early
    and most intermediate counts die. Interleaving the two keeps every level
    valid by construction while importance still picks among the legal moves,
    which is what keeps the ladder at 2-ply granularity instead of collapsing
    to whole-chain unit drops.

    Pure greedy is myopic, though: when importance strongly favours one angle
    (e.g. a compressive case that wants to shed every 0deg ply) it takes the
    cheap drops first and reaches a state where no legal drop is left, and the
    ladder then goes invalid for a long stretch. So each candidate is also
    checked one step ahead, and drops that dead-end the ladder are avoided
    unless nothing else is available."""
    n = len(half_angles)

    def _is_valid(present_set):
        lam = _full_stack_from_indices(half_angles, present_set)
        if not (_check_disorientation(lam, max_disorientation)
                and _check_contiguity(lam, max_contiguity)):
            return False
        return _check_10pct(lam, required_angles) if check_10pct else True


    remaining = list(cycles)
    present = set(range(n))
    order = []
    while remaining:
        feasible = [(imp, k) for k, (imp, idxs) in enumerate(remaining)
                    if _is_valid(present - set(idxs))]
        if feasible:
            # Prefer drops that still leave a legal move afterwards, so the
            # ladder keeps producing valid counts further down.
            safe = []
            for imp, k in feasible:
                if len(remaining) == 1:
                    safe.append((imp, k))
                    continue
                _nxt = present - set(remaining[k][1])
                if any(_is_valid(_nxt - set(idx2))
                       for j, (_, idx2) in enumerate(remaining) if j != k):
                    safe.append((imp, k))
            k = min(safe or feasible, key=lambda t: t[0])[1]
        else:
            # Nothing legal left; take the least important so the ladder still
            # reaches the thinnest laminate (that level is validated later).
            k = min(range(len(remaining)), key=lambda i: remaining[i][0])
        imp, idxs = remaining.pop(k)
        order.append((imp, idxs))
        present -= set(idxs)
    return order


def determine_drop_order(
    half_angles: list[float],
    ply_importance: np.ndarray,
    covering_plies: int,
    max_contiguity: int,
    max_disorientation: float,
    quiet: bool = False,
    allow_sublaminates: bool = False,
    tolerant_ladder: bool = False,
    positional: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Determine the ply drop sequence, verifying every intermediate laminate
    for disorientation/contiguity/10% before committing. Returns
    (drop_order, forced_continuous, extra_continuous). quiet suppresses prints
    (batch scoring); allow_sublaminates skips bridge-ply and 10% enforcement,
    permitting zones with fewer angle groups (e.g. [0/90]_s)."""
    n = len(half_angles)
    required_angles = sorted(set(half_angles))

    # Step 1: identify SB-cycles and forced-continuous for balance
    forced_cont, cycles = _identify_sb_cycles(half_angles, ply_importance, covering_plies)

    continuous = set(range(covering_plies)) | set(forced_cont)

    # Step 2: check if we need additional continuous plies for disorientation.
    all_drop_indices = set()
    for _, idxs in cycles:
        all_drop_indices.update(idxs)

    min_present = set(range(n)) - all_drop_indices
    min_lam = _full_stack_from_indices(half_angles, min_present)
    extra_cont = []

    if not allow_sublaminates and \
       (not _check_disorientation(min_lam, max_disorientation) or
        not _check_contiguity(min_lam, max_contiguity)):
        bridge_found = False
        droppable_indices = sorted(all_drop_indices)

        for idx in sorted(droppable_indices, key=lambda i: -ply_importance[i]):
            test = min_present | {idx}
            lam = _full_stack_from_indices(half_angles, test)
            if _check_disorientation(lam, max_disorientation) and \
               _check_contiguity(lam, max_contiguity):
                extra_cont.append(idx)
                continuous.add(idx)
                min_present.add(idx)
                bridge_found = True
                break

        if not bridge_found:
            from itertools import combinations
            for pair in combinations(
                sorted(droppable_indices, key=lambda i: -ply_importance[i]), 2
            ):
                test = min_present | set(pair)
                lam = _full_stack_from_indices(half_angles, test)
                if _check_disorientation(lam, max_disorientation) and \
                   _check_contiguity(lam, max_contiguity):
                    extra_cont.extend(pair)
                    continuous.update(pair)
                    min_present.update(pair)
                    bridge_found = True
                    break

        if not bridge_found and not quiet:
            print("  WARNING: Could not find bridge plies for disorientation.")
            if _VERBOSE:
                print("           Some intermediate laminates may violate guidelines.")

        updated_cycles = []
        for imp, idxs in cycles:
            remaining = [i for i in idxs if i not in extra_cont]
            if len(remaining) == 0:
                continue
            if len(idxs) == 2 and len(remaining) == 1:
                extra_cont.extend(remaining)
                continuous.update(remaining)
                continue
            new_imp = np.mean([ply_importance[i] for i in remaining])
            updated_cycles.append((new_imp, remaining))
        cycles = updated_cycles

    # Step 2b: enforce 10% rule for the minimum laminate.
    # Force the highest-importance droppable ply of each missing angle continuous.
    all_drop_indices = set()
    for _, idxs in cycles:
        all_drop_indices.update(idxs)
    min_present = set(range(n)) - all_drop_indices
    min_lam = _full_stack_from_indices(half_angles, min_present)

    if not allow_sublaminates and not _check_10pct(min_lam, required_angles):
        min_counts = {}
        for a in min_lam:
            min_counts[a] = min_counts.get(a, 0) + 1
        min_required = max(1, int(np.ceil(len(min_lam) * 0.1)))

        for angle in required_angles:
            if min_counts.get(angle, 0) >= min_required:
                continue
            candidates = sorted(
                [i for i in all_drop_indices if half_angles[i] == angle],
                key=lambda i: -ply_importance[i],
            )
            for cand in candidates:
                test_present = min_present | {cand}
                test_lam = _full_stack_from_indices(half_angles, test_present)
                if _check_disorientation(test_lam, max_disorientation) and \
                   _check_contiguity(test_lam, max_contiguity):
                    extra_cont.append(cand)
                    continuous.add(cand)
                    min_present.add(cand)
                    min_counts[angle] = min_counts.get(angle, 0) + 2
                    break

        # Remove newly forced-continuous plies from their cycles
        updated_cycles = []
        for imp, idxs in cycles:
            remaining = [i for i in idxs if i not in extra_cont]
            if len(remaining) == 0:
                continue
            if len(idxs) == 2 and len(remaining) == 1:
                extra_cont.extend(remaining)
                continuous.update(remaining)
                continue
            new_imp = np.mean([ply_importance[i] for i in remaining])
            updated_cycles.append((new_imp, remaining))
        cycles = updated_cycles

        if extra_cont and not quiet and _VERBOSE:
            print(f"  Forced continuous (10% rule):")
            for idx in extra_cont:
                print(f"    Half-ply {idx} ({half_angles[idx]:+.0f}deg)")

    # Step 3: pick the drop ordering. Tolerant mode always searches: pure
    # importance order drops a +-45 pair first and poisons every count.
    if positional:
        # Position-driven order: always drop the innermost remaining cycle, so
        # what is left is a PREFIX of the repeated angle chain and therefore
        # guideline-valid at (almost) every step. This ignores FSO importance
        # by design; it is the fallback for fields where importance ordering
        # destroys the ladder (e.g. a compressive case whose FSO wants to shed
        # the 0deg plies that are also the disorientation bridges). Cycles are
        # kept intact so +-45 balance still holds.
        ordered_cycles = sorted(cycles, key=lambda x: -max(x[1]))
    elif allow_sublaminates and not tolerant_ladder:
        ordered_cycles = sorted(cycles, key=lambda x: x[0])
    elif len(cycles) <= 8:
        ordered_cycles = _find_disorientation_feasible_order(
            half_angles, cycles, continuous, max_contiguity, max_disorientation,
            required_angles,
            tolerant=tolerant_ladder,
        )
    else:
        # >8 cycles: brute force is too costly, but pure importance order makes
        # nearly every intermediate laminate illegal. Build the ladder greedily
        # instead, taking only legal drops and letting importance choose among
        # them.
        if not quiet and _VERBOSE:
            print(f"  {len(cycles)} SB-cycles (>8): greedy feasible ordering "
                  f"(least-important legal drop at each step).")
        ordered_cycles = _greedy_feasible_cycle_order(
            half_angles, cycles, max_contiguity, max_disorientation,
            required_angles, check_10pct=not allow_sublaminates,
        )

    drop_order = []
    for _, idxs in ordered_cycles:
        drop_order.extend(idxs)

    return drop_order, forced_cont, extra_cont


# ==============================================================================
#  Valid ply count computation
# ==============================================================================

def _compute_valid_ply_counts(
    n_max: int,
    drop_order: list[int],
    half_angles: list[float],
    half_plies: list = None,
    max_contiguity: int = 2,
    max_disorientation: float = 45.0,
    required_angles: list[float] = None,
) -> list[int]:
    """All ply counts reachable by sequential SB-cycle drops. If half_plies is
    given, keeps only those passing disorientation/contiguity/10%."""
    counts = [n_max]
    current = n_max
    i = 0
    while i < len(drop_order):
        idx = drop_order[i]
        a = half_angles[idx]
        if a not in (0.0, 90.0) and (i + 1 < len(drop_order)):
            j = drop_order[i + 1]
            if half_angles[j] == -a:
                current -= 4
                i += 2
                counts.append(current)
                continue
        current -= 2
        i += 1
        counts.append(current)

    if half_plies is None:
        return sorted(set(counts))

    valid = []
    for n in sorted(set(counts)):
        nd = (n_max - n) // 2
        active = [p for p in half_plies
                  if p.is_continuous or (p.drop_rank is not None and p.drop_rank > nd)]
        fs = [p.angle for p in active] + [p.angle for p in reversed(active)]
        present_angles = sorted(set(fs))
        is_sublaminate = required_angles and len(present_angles) < len(required_angles)
        ok = True
        if not is_sublaminate:
            if not _check_disorientation(fs, max_disorientation):
                ok = False
            if ok and not _check_contiguity(fs, max_contiguity):
                ok = False
            if ok and required_angles and not _check_10pct(fs, required_angles):
                ok = False
        else:
            # Sublaminate (whole angle dropped): 10% checked only over present
            # angles, but disorientation still applies — otherwise an
            # infeasible count like [+45,-45,-45,+45] (90deg jump) would be
            # reported valid, selected as a zone, and fail final enforcement.
            if not _check_disorientation(fs, max_disorientation):
                ok = False
            if ok and not _check_contiguity(fs, max_contiguity):
                ok = False
            if ok and not _check_10pct(fs, present_angles):
                ok = False
        if ok:
            valid.append(n)
    return valid


def _precompute_fso_xi(
    elem_ply_thicknesses: dict[int, list[float]],
    base_t: float,
    half_angles: list[float],
) -> dict[int, np.ndarray]:
    """Per-element FSO xi^D vectors, built from the per-half-ply thickness data.
    Precomputed once so candidate mismatch scoring stays cheap."""
    n_half = len(half_angles)
    fso_xi = {}
    for eid, ply_factors in elem_ply_thicknesses.items():
        t_half = np.array(ply_factors) * base_t
        if t_half.sum() <= 0.0:
            fso_xi[eid] = np.zeros(4)
            continue
        angles_deg = list(half_angles) + list(reversed(half_angles))
        thicknesses = list(t_half) + list(t_half[::-1])
        fso_xi[eid] = _compute_xi_from_angles_and_thicknesses(angles_deg, thicknesses)
    return fso_xi


# ==============================================================================
#  SST construction (main entry point)
# ==============================================================================

def build_sst(
    initial_plies: list[tuple[float, float]],
    elem_ply_thicknesses: dict[int, list[float]],
    n_zones: int,
    min_ply_thickness: float = 0.125,
    covering_plies: int = 1,
    max_contiguity: int = 2,
    max_disorientation: float = 45.0,
    position_bias: float = 0.05,
    importance_method: str = "global",
    cluster_data: dict = None,
    cluster_centers: list = None,
    doubling_factor_override: int = None,
    optimize_guide_stack: bool = False,
    allow_sublaminates: bool = True,
    tolerant_drop_order: bool = False,
    min_plies: int = 0,
    dn_rule: int = 0,
    _df_retry: int = 0,
) -> StackingSequenceTable:
    """Build a Stacking Sequence Table from FSO results. Too few plies for the
    requested zones -> the guide is expanded by splitting plies down to
    min_ply_thickness. FSO per-ply data drives the drop order and every
    intermediate laminate is checked for guideline compliance.

    importance_method: "global" (uniform element-average, robust) or
    "zone_aware" (inverse-thickness-weighted per-zone retention; needs
    cluster_data/cluster_centers from k_means1D).
    optimize_guide_stack: search all palindromic sweeps for the stiffest
    (max trace(D)) damtol-valid guide."""
    n_orig = len(initial_plies)
    n_orig_half = n_orig // 2
    base_t = initial_plies[0][1]
    orig_half_angles = [a for a, _ in initial_plies[:n_orig_half]]

    # Doubling factor: split each ply into base_t/min_ply_thickness sub-plies
    if doubling_factor_override is not None:
        doubling_factor = doubling_factor_override
    else:
        doubling_factor = max(1, int(base_t / min_ply_thickness))

    # The guide laminate must always sum to the ORIGINAL laminate thickness, so
    # the ply thickness follows from the guide's ply count rather than being
    # pinned to min_ply_thickness. Pinning it broke the invariant whenever the
    # DF retry changed n_max (e.g. DF 8->9 gave 72 x 0.15625 = 11.25 mm on a
    # 10 mm skin). min_ply_thickness stays the manufacturing floor and is only
    # checked against, never used as the value.
    _skin_total = sum(t for _, t in initial_plies)
    _n_unique_ang = len(set(a for a, _ in initial_plies))
    _n_max_pred = (2 * _n_unique_ang * doubling_factor
                   if doubling_factor > 1 else n_orig)
    mfg_t = _skin_total / _n_max_pred if _n_max_pred else base_t

    if _VERBOSE:
        print(f"\n  Original layup: {n_orig} plies, t_ply = {base_t} mm")
        print(f"  Min manufacturable thickness: {min_ply_thickness} mm")
        print(f"  Doubling factor: {doubling_factor}x (manufacturing t_ply = {mfg_t} mm)")

    # FSO importance (candidate-independent, computed once up front)
    if importance_method == "zone_aware":
        if cluster_data is None or cluster_centers is None:
            raise ValueError("zone_aware method requires cluster_data and cluster_centers")
        orig_imp, zone_ret, zone_ctrs, zone_wts = compute_ply_importance_zone_aware(
            elem_ply_thicknesses, base_t, cluster_data, cluster_centers
        )
        if _VERBOSE:
            print(f"\n  FSO importance method: ZONE-AWARE (inverse-thickness weighted)")
            print(f"\n  Per-zone ply retention (half-stack):")
            header = "  " + f"{'Zone':>6} {'Center':>8} {'Weight':>7} |"
            for a in orig_half_angles:
                header += f" {a:>+6.0f}°"
            print(header)
            print("  " + "-" * len(header))
            for zi in np.argsort(zone_ctrs):
                row = f"  {int(zi+1):>6} {zone_ctrs[zi]:>8.3f} {zone_wts[zi]:>7.3f} |"
                for pi in range(len(orig_half_angles)):
                    row += f" {zone_ret[zi, pi]:>7.4f}"
                print(row)
            print(f"\n  Weighted importance (original half-stack):")
    else:
        orig_imp = compute_ply_importance_global(elem_ply_thicknesses, base_t)
        if _VERBOSE:
            print(f"\n  FSO importance method: GLOBAL (uniform element-average)")
            print(f"\n  Global importance (original half-stack):")

    if _VERBOSE:
        for i, (a, im) in enumerate(zip(orig_half_angles, orig_imp)):
            print(f"    Ply {i+1}: {a:>+6.1f}deg  retention = {im:.4f}")

    # --- Precompute FSO xi^D per element (for candidate scoring) ---
    fso_xi = _precompute_fso_xi(elem_ply_thicknesses, base_t, orig_half_angles)

    # --- Expand guide laminate (with SST-aware scoring) ---
    if doubling_factor > 1:
        candidates = generate_candidate_half_stacks(
            orig_half_angles, doubling_factor, max_contiguity,
            max_disorientation, optimize=optimize_guide_stack,
        )
        if optimize_guide_stack and len(candidates) > 1:
            scored = []
            for half_stack, _ in candidates:
                sst_score, n_valid_cand = _score_candidate_xi_mismatch(
                    half_stack, orig_half_angles, orig_imp,
                    position_bias, covering_plies, max_contiguity,
                    max_disorientation, n_zones, fso_xi,
                    cluster_data, cluster_centers, mfg_t,
                    tolerant_ladder=tolerant_drop_order,
                    min_plies=min_plies,
                )
                scored.append((half_stack, sst_score, n_valid_cand))
            if tolerant_drop_order:
                # Ladder quality first (usable counts, then smallest gap): a
                # stack whose ladder actually covers the FSO range beats a
                # marginally better xi match with a hole in the middle.
                scored.sort(key=lambda x: (-x[2][0], -x[2][1], -x[1]))
            else:
                scored.sort(key=lambda x: -x[1])
            expanded_half, best_score = scored[0][0], scored[0][1]
            n_eval = len(scored)
            worst_score = scored[-1][1]
        else:
            expanded_half, _ = candidates[0]
            n_eval = len(candidates)
            best_score, _ = _score_candidate_xi_mismatch(
                expanded_half, orig_half_angles, orig_imp,
                position_bias, covering_plies, max_contiguity,
                max_disorientation, n_zones, fso_xi,
                cluster_data, cluster_centers, mfg_t,
                tolerant_ladder=tolerant_drop_order,
                min_plies=min_plies,
            )
            worst_score = None

        n_max = len(expanded_half) * 2
        if _VERBOSE:
            print(f"\n  Guide stack search: {n_eval} candidate(s) evaluated")
            print(f"  Selected half-stack ({len(expanded_half)} plies): {expanded_half}")
            print(f"  SST xi^D mismatch score: {best_score:.6f} (higher = better,"
                  f" 0 = perfect match)")
        if worst_score is not None:
            pct = (best_score - worst_score) / abs(worst_score) * 100.0
            print(f"  Worst candidate score:  {worst_score:.6f} "
                  f"({pct:+.1f}% relative improvement)")
        if _VERBOSE:
            print(f"  Guide laminate: {n_max} plies, total = {n_max * mfg_t:.2f} mm")
    else:
        expanded_half = list(orig_half_angles)
        n_max = n_orig
        if _VERBOSE:
            print(f"  No doubling needed")

    # Lock the invariant: guide laminate thickness == original laminate
    # thickness, whatever n_max the guide search ended up with.
    if n_max:
        mfg_t = _skin_total / n_max
    if mfg_t < min_ply_thickness - 1e-9:
        print(f"  WARNING: guide of {n_max} plies needs {mfg_t:.5f} mm plies, "
              f"below the {min_ply_thickness:.5f} mm manufacturing floor. "
              f"Reduce the zone count or the doubling factor.")
    if _VERBOSE:
        print(f"  Guide laminate: {n_max} plies x {mfg_t:.5f} mm = "
              f"{n_max * mfg_t:.3f} mm (original {_skin_total:.3f} mm)")

    full_guide = expanded_half + expanded_half[::-1]

    exp_imp = compute_expanded_importance(orig_half_angles, orig_imp, expanded_half, position_bias)
    if _VERBOSE:
        print(f"\n  Expanded importance (with {position_bias:.0%} position bias):")
        for i, (a, im) in enumerate(zip(expanded_half, exp_imp)):
            print(f"    Half-ply {i+1}: {a:>+6.1f}deg  effective = {im:.4f}")

    drop_order, forced_cont, extra_cont = determine_drop_order(
        expanded_half, exp_imp, covering_plies, max_contiguity, max_disorientation,
        allow_sublaminates=allow_sublaminates,
        tolerant_ladder=tolerant_drop_order,
    )

    if forced_cont and _VERBOSE:
        print(f"\n  Forced continuous (balance):")
        for idx in forced_cont:
            print(f"    Half-ply {idx+1} ({expanded_half[idx]:+.0f}deg)")
    if extra_cont and _VERBOSE:
        print(f"  Forced continuous (disorientation bridge / 10% rule):")
        for idx in extra_cont:
            print(f"    Half-ply {idx+1} ({expanded_half[idx]:+.0f}deg)")

    if _VERBOSE:
        print(f"\n  Drop order:")
        for rank, idx in enumerate(drop_order, 1):
            print(f"    Drop {rank}: half-ply {idx+1} ({expanded_half[idx]:+.0f}deg), "
                  f"importance = {exp_imp[idx]:.4f}")

    all_counts = _compute_valid_ply_counts(n_max, drop_order, expanded_half)
    if _VERBOSE:
        print(f"\n  All achievable ply counts (raw): {all_counts}")

    # --- Build Ply objects (needed for validation) ---
    all_continuous = set(range(covering_plies)) | set(forced_cont) | set(extra_cont)
    plies = []
    for i in range(len(expanded_half)):
        if i in drop_order:
            rank = drop_order.index(i) + 1
            is_cont = False
        else:
            rank = None
            is_cont = True
        plies.append(Ply(expanded_half[i], mfg_t, is_cont, rank, i))

    required_angles = sorted(set(expanded_half))
    all_valid = _compute_valid_ply_counts(
        n_max, drop_order, expanded_half,
        half_plies=plies, max_contiguity=max_contiguity,
        max_disorientation=max_disorientation, required_angles=required_angles,
    )
    if _VERBOSE:
        print(f"  Constraint-valid ply counts: {all_valid}")

    # --- Fall back to a positional drop order if importance wrecked the ladder ---
    # Importance ordering is only worth keeping while it leaves a ladder that
    # can actually discretise the FSO field. When the FSO wants to shed the very
    # plies that bridge disorientation (a compressive case sheds 0deg), the
    # importance-driven order kills them early, every mid-range laminate becomes
    # illegal, and the ladder collapses to a narrow band near the guide -> tens
    # of percent of volume error. A position-driven order gives up the (small)
    # angle-composition fidelity to keep a ladder that spans the range.
    def _ladder_span_quality(_counts):
        """(Dn satisfied, usable levels, -largest gap).

        Dn compliance ranks FIRST: a ladder with more levels but a gap wider
        than dn_rule cannot produce a manufacturable taper, so it loses to a
        slightly sparser ladder that can. Gap includes the distance up from the
        floor, so a ladder that never reaches the thin zones is penalised."""
        _u = sorted(c for c in _counts if c >= min_plies) if min_plies \
            else sorted(_counts)
        if len(_u) < 2:
            return (0, 0, -n_max)
        _g = max(_u[i + 1] - _u[i] for i in range(len(_u) - 1))
        if min_plies:
            _g = max(_g, _u[0] - min_plies)
        return (1 if (dn_rule and _g <= dn_rule) else 0, len(_u), -_g)

    _q_imp = _ladder_span_quality(all_valid)
    if dn_rule and -_q_imp[2] > dn_rule:
        _do_p, _fc_p, _ec_p = determine_drop_order(
            expanded_half, exp_imp, covering_plies, max_contiguity,
            max_disorientation, allow_sublaminates=allow_sublaminates,
            tolerant_ladder=tolerant_drop_order, positional=True,
        )
        _cont_p = set(range(covering_plies)) | set(_fc_p) | set(_ec_p)
        _plies_p = [
            Ply(expanded_half[i], mfg_t,
                i not in _do_p, (_do_p.index(i) + 1) if i in _do_p else None, i)
            for i in range(len(expanded_half))
        ]
        _valid_p = _compute_valid_ply_counts(
            n_max, _do_p, expanded_half,
            half_plies=_plies_p, max_contiguity=max_contiguity,
            max_disorientation=max_disorientation,
            required_angles=required_angles,
        )
        _q_pos = _ladder_span_quality(_valid_p)
        if _q_pos > _q_imp:
            if _VERBOSE:
                print(f"  Importance ladder spans poorly (max gap {-_q_imp[2]} "
                      f"> Dn {dn_rule}); switching to positional drop order.")
                print(f"  Positional ply counts: {_valid_p}")
            drop_order, forced_cont, extra_cont = _do_p, _fc_p, _ec_p
            all_continuous, plies, all_valid = _cont_p, _plies_p, _valid_p
            all_counts = _compute_valid_ply_counts(
                n_max, drop_order, expanded_half)
        elif _VERBOSE:
            print(f"  Positional order is no better "
                  f"(gap {-_q_pos[2]} vs {-_q_imp[2]}); keeping importance order.")

    # Fallback: unit-drop SST when too few CONSTRAINT-VALID counts (raw counts
    # are useless if most violate disorientation). Unit-drop removes whole
    # chain units (one disorientation-safe block per step), so it always gives
    # n_zones manufacturable levels once DF supplies enough units, e.g. the
    # [8,16,24,32,40] ladder at 45 deg.
    _skip_unit_drop = cluster_centers is not None and len(all_valid) >= n_zones
    if (len(all_valid) < 2 or (len(all_valid) < n_zones and len(all_valid) >= 2)) \
            and not _skip_unit_drop:
        if _VERBOSE:
            print(f"\n  Only {len(all_valid)} valid ply counts from individual drops.")
            print(f"  Switching to unit-drop approach (chain-unit removal).")

        chain = _build_angle_chain(sorted(set(expanded_half)), max_disorientation)
        forward = list(chain)
        backward = list(reversed(chain))
        n_units = doubling_factor
        unit_size = len(forward)
        half_n = n_units * unit_size

        if half_n != len(expanded_half):
            if _VERBOSE:
                print(f"  ERROR: unit-drop mismatch (half_n={half_n}, expanded={len(expanded_half)})")
        elif n_units >= n_zones:
            if cluster_centers is not None:
                unit_thickness = unit_size * 2 * mfg_t
                sorted_centers = sorted(cluster_centers)
                units_to_keep = sorted(set(
                    max(1, min(round(c / unit_thickness), n_units))
                    for c in sorted_centers
                ))
                while len(units_to_keep) < n_zones:
                    remaining = [u for u in range(1, n_units + 1) if u not in units_to_keep]
                    if not remaining:
                        break
                    units_to_keep.append(remaining[0])
                    units_to_keep = sorted(units_to_keep)
            else:
                units_to_keep = sorted(set(int(x) for x in np.round(np.linspace(1, n_units, n_zones)).astype(int)))
            zone_counts = sorted([int(k * unit_size * 2) for k in units_to_keep])

            drop_order = []
            forced_cont = []
            extra_cont = []

            for u in reversed(range(n_units)):
                if u == 0:
                    continue
                offset = u * unit_size
                for j in range(unit_size):
                    drop_order.append(offset + j)

            plies = []
            for i in range(len(expanded_half)):
                if i in drop_order:
                    rank = drop_order.index(i) + 1
                    is_cont = False
                else:
                    rank = None
                    is_cont = True
                plies.append(Ply(expanded_half[i], mfg_t, is_cont, rank, i))

            all_valid = zone_counts
            if _VERBOSE:
                print(f"  Unit-drop zones: {zone_counts}")
                print(f"  Units kept: {units_to_keep} out of {n_units}")
        else:
            if _VERBOSE:
                print(f"  Not enough units ({n_units}) for {n_zones} zones.")

    if not all_valid:
        print("  WARNING: No ply counts pass all constraints. Using raw counts.")
        all_valid = all_counts

    # Clamp the ladder to the FSO's own thickness range. The FSO never produced
    # an element thinner than its LAMTHK floor, so levels below it are not part
    # of the design being discretised; offering them lets the rank-based
    # assignment map the thinnest cluster onto a laminate far thinner than
    # anything the FSO asked for (the -30% volume error).
    if min_plies:
        _kept = [c for c in all_valid if c >= min_plies]
        if len(_kept) >= 2:
            if _VERBOSE and len(_kept) < len(all_valid):
                print(f"  Ladder clamped to >= {min_plies} plies (FSO minimum): "
                      f"{all_valid} -> {_kept}")
            all_valid = _kept
        elif _VERBOSE:
            print(f"  WARNING: clamping to >= {min_plies} plies would leave "
                  f"{len(_kept)} level(s); keeping the unclamped ladder.")

    # Select zone counts from the constraint-valid pool, not the raw counts: a
    # count that violates disorientation/contiguity/10% never becomes
    # manufacturable at any DF, so picking it would just spin the DF loop.
    _zc_pool = all_valid if all_valid else all_counts
    if cluster_centers is not None and len(_zc_pool) >= n_zones:
        sorted_centers = sorted(cluster_centers)
        c_max = max(sorted_centers) if sorted_centers else 1.0
        zone_counts = []
        for c in sorted_centers:
            if c_max > 0:
                ideal = round(c / c_max * n_max)
            else:
                ideal = n_max
            ideal = max(_zc_pool[0], min(ideal, n_max))
            candidates = [v for v in _zc_pool if v not in zone_counts]
            if not candidates:
                break
            closest = min(candidates, key=lambda v: abs(v - ideal))
            zone_counts.append(closest)
        while len(zone_counts) < n_zones:
            remaining = [v for v in _zc_pool if v not in zone_counts]
            if not remaining:
                break
            target = np.mean(zone_counts) if zone_counts else _zc_pool[0]
            zone_counts.append(min(remaining, key=lambda v: abs(v - target)))
        zone_counts = sorted(zone_counts)
    elif n_zones >= len(all_valid):
        zone_counts = all_valid
    else:
        indices = np.round(np.linspace(0, len(all_valid) - 1, n_zones)).astype(int)
        zone_counts = sorted(set(all_valid[i] for i in indices))
    # Dn-aware repair: the manufactured taper steps between THICKNESS-ADJACENT
    # ZONES, not between all ladder levels, so a level being available is not
    # enough — the selected ones must be within dn_rule of each other. Fix any
    # oversized step by raising the thinner zone (adds material, never removes)
    # to the lowest pool level that closes the gap, keeping order strict.
    if dn_rule and len(zone_counts) >= 2:
        zone_counts = sorted(zone_counts)
        for _i in range(len(zone_counts) - 2, -1, -1):
            _gap = zone_counts[_i + 1] - zone_counts[_i]
            if _gap <= dn_rule:
                continue
            _need = zone_counts[_i + 1] - dn_rule
            _low = zone_counts[_i - 1] if _i > 0 else 0
            _cand = [v for v in _zc_pool
                     if _need <= v < zone_counts[_i + 1] and v > _low]
            if _cand:
                zone_counts[_i] = min(_cand)
        zone_counts = sorted(set(zone_counts))

    if _VERBOSE:
        print(f"  Selected zone ply counts: {zone_counts} \n")

    # --- Auto-increase DF if not enough zones produced ---
    # A finer DF is only legal while the resulting ply stays at or above the
    # manufacturing floor: the guide must sum to the original laminate, so
    # ply_t = skin / (2 * n_unique * DF) shrinks as DF grows. Without this cap
    # the retry chased an unreachable zone count down to 0.048 mm plies.
    _df_next_ply = (_skin_total / (2 * _n_unique_ang * (doubling_factor + 1))
                    if _n_unique_ang else 0.0)
    _df_can_grow = _df_next_ply >= min_ply_thickness - 1e-9
    if len(zone_counts) < n_zones and _df_retry < 20 and not _df_can_grow:
        print(f"  DF={doubling_factor}: only {len(zone_counts)} of {n_zones} "
              f"zones achievable, and DF {doubling_factor + 1} would need "
              f"{_df_next_ply:.5f} mm plies (floor {min_ply_thickness:.5f}). "
              f"Keeping {len(zone_counts)} zones.")
    if len(zone_counts) < n_zones and _df_retry < 20 and _df_can_grow:
        new_df = doubling_factor + 1
        print(f"  DF={doubling_factor}: only {len(zone_counts)} zones achievable "
              f"(need {n_zones}). Auto-increasing DF to {new_df}...")
        return build_sst(
            initial_plies=initial_plies,
            elem_ply_thicknesses=elem_ply_thicknesses,
            n_zones=n_zones,
            min_ply_thickness=min_ply_thickness,
            covering_plies=covering_plies,
            max_contiguity=max_contiguity,
            max_disorientation=max_disorientation,
            position_bias=position_bias,
            importance_method=importance_method,
            cluster_data=cluster_data,
            cluster_centers=cluster_centers,
            doubling_factor_override=new_df,
            optimize_guide_stack=optimize_guide_stack,
            allow_sublaminates=allow_sublaminates,
            tolerant_drop_order=tolerant_drop_order,
            min_plies=min_plies,
            dn_rule=dn_rule,
            _df_retry=_df_retry + 1,
        )
    elif len(zone_counts) < n_zones:
        print(f"  WARNING: Could not achieve {n_zones} zones after {_df_retry} DF increases "
              f"(DF={doubling_factor}). Using {len(zone_counts)} zones.")

    drop_seq = [(idx, expanded_half[idx]) for idx in drop_order]

    # --- Check guidelines ---
    report = _check_all_guidelines(
        plies, zone_counts, n_max, expanded_half, max_contiguity, max_disorientation
    )

    # Enforce max-disorientation. The drop-order search can silently fall back
    # to importance-only ordering, so a violation used to be reported [FAIL]
    # and accepted anyway. A failure isn't terminal: more sub-plies give the
    # search more room, so auto-increase DF and rebuild (may inflate volume);
    # reject only once the retry budget is spent.
    _diso_fail = [name for name, ok in report.items()
                  if name.startswith("Disorientation") and not ok]
    if _diso_fail:
        if _df_retry < 20:
            new_df = doubling_factor + 1
            print(f"  DF={doubling_factor}: disorientation constraint (max "
                  f"{max_disorientation:g} deg) violated at "
                  f"{', '.join(_diso_fail)}. Auto-increasing DF to {new_df}...")
            return build_sst(
                initial_plies=initial_plies,
                elem_ply_thicknesses=elem_ply_thicknesses,
                n_zones=n_zones,
                min_ply_thickness=min_ply_thickness,
                covering_plies=covering_plies,
                max_contiguity=max_contiguity,
                max_disorientation=max_disorientation,
                position_bias=position_bias,
                importance_method=importance_method,
                cluster_data=cluster_data,
                cluster_centers=cluster_centers,
                doubling_factor_override=new_df,
                optimize_guide_stack=optimize_guide_stack,
                allow_sublaminates=allow_sublaminates,
                tolerant_drop_order=tolerant_drop_order,
                min_plies=min_plies,
                dn_rule=dn_rule,
                _df_retry=_df_retry + 1,
            )
        raise ValueError(
            f"SST disorientation constraint violated (max "
            f"{max_disorientation:g} deg) at {', '.join(_diso_fail)} "
            f"after {_df_retry} DF increases (DF={doubling_factor}). "
            f"Relax max_disorientation or adjust the guide stack; the SST "
            f"discretization is not manufacturable as-is."
        )

    return StackingSequenceTable(
        half_plies=plies, n_min=min(zone_counts), n_max=n_max,
        ply_angles=full_guide, zone_ply_counts=sorted(zone_counts),
        drop_sequence=drop_seq, base_ply_thickness=base_t,
        manufacturing_ply_thickness=mfg_t, doubling_factor=doubling_factor,
        guideline_report=report,
        valid_ply_counts=sorted(_zc_pool),
    )


# ==============================================================================
#  Guideline checking
# ==============================================================================

def _check_all_guidelines(half_plies, zone_ply_counts, n_max, half_angles,
                          max_contiguity, max_disorientation):
    report = {}
    required_angles = sorted(set(p.angle for p in half_plies))

    def _present_at(p, n):
        nd = (n_max - n) // 2
        return p.is_continuous or (p.drop_rank is not None and p.drop_rank > nd)

    for n in zone_ply_counts:
        active = [p for p in half_plies if _present_at(p, n)]
        aa = [p.angle for p in active]
        fs = aa + aa[::-1]
        tag = f"(n={n})"
        report[f"Symmetry {tag}"] = True
        plus = sum(1 for a in aa if a not in (0, 90) and a > 0)
        minus = sum(1 for a in aa if a not in (0, 90) and a < 0)
        report[f"Balance {tag}"] = (plus == minus)
        report[f"Contiguity {tag}"] = _check_contiguity(fs, max_contiguity)
        report[f"Disorientation {tag}"] = _check_disorientation(fs, max_disorientation)
        report[f"Damtol {tag}"] = (fs[0] != 0.0 and fs[-1] != 0.0) if len(fs) >= 2 else True
        report[f"10% rule {tag}"] = _check_10pct(fs, required_angles)
    report["Covering"] = half_plies[0].is_continuous

    # Continuity (guideline 13): each thinner zone's ply set must be a subset
    # of the thicker one's (plies dropped, never cut). True by construction,
    # verified explicitly here.
    scounts = sorted(zone_ply_counts)
    nested = True
    for thin_n, thick_n in zip(scounts, scounts[1:]):
        thin_set = {p.half_idx for p in half_plies if _present_at(p, thin_n)}
        thick_set = {p.half_idx for p in half_plies if _present_at(p, thick_n)}
        if not thin_set <= thick_set:
            nested = False
            break
    report["Continuity (thinner zones nested in thicker)"] = nested

    # Taper guidelines (guideline 12): taper laminates must keep symmetry/
    # balance/contiguity/disorientation, aggregated over the zones above.
    _taper_prefixes = ("Symmetry", "Balance", "Contiguity", "Disorientation")
    report["Taper guidelines (sym/bal/contig/disor at all zones)"] = all(
        v for k, v in report.items() if k.startswith(_taper_prefixes))

    # Internal continuity (guideline 10), checked per real transition (not the
    # full guide, since plies dropped in other tapers aren't in this laminate).
    for thin_n, thick_n in zip(scounts, scounts[1:]):
        flags = [_present_at(p, thin_n)
                 for p in half_plies if _present_at(p, thick_n)]
        report[f"Internal continuity ({thick_n}->{thin_n})"] = \
            _check_internal_continuity(flags)

    return report


# ==============================================================================
#  Zone assignment
# ==============================================================================

def assign_zones(sst, cluster_data, cluster_centers):
    """Simple rank-based zone assignment (thinnest cluster -> thinnest ply count)."""
    sc = sorted(enumerate(cluster_centers), key=lambda x: x[1])
    sp = sorted(sst.zone_ply_counts)
    c2n = {ci: sp[min(rank, len(sp) - 1)] for rank, (ci, _) in enumerate(sc)}
    en = {}
    for ci, elements in cluster_data.items():
        n = c2n[ci]
        for eid, _ in elements:
            en[eid] = n
    return en


def assign_zones_direct(
    sst,
    cluster_data: dict,
    cluster_centers: list[float],
) -> tuple[dict[int, int], tuple[int, ...], float]:
    """Direct rank-based zone assignment: thinnest cluster -> thinnest ply
    count, and so on. No volume matching, so it keeps the requested zone count
    exactly but doesn't match FSO total volume. Returns
    (elem_nplies, ply_counts, relative_error_pct)."""
    mfg_t = sst.manufacturing_ply_thickness
    sp = sorted(sst.zone_ply_counts)
    sc = sorted(enumerate(cluster_centers), key=lambda x: x[1])
    c2n = {ci: sp[min(rank, len(sp) - 1)] for rank, (ci, _) in enumerate(sc)}

    elem_nplies = {}
    for ci, elements in cluster_data.items():
        n = c2n[ci]
        for eid, _ in elements:
            elem_nplies[eid] = n

    eids = list(elem_nplies.keys())
    actual_total = sum(elem_nplies[eid] * mfg_t for eid in eids)
    rel_error = 0.0

    used_counts = tuple(sorted(set(elem_nplies.values())))

    if _VERBOSE:
        print(f"\n  Direct rank-based assignment (no volume matching):")
        print(f"    Cluster -> ply count mapping:")
        for rank, (ci, _) in enumerate(sc):
            n = c2n[ci]
            print(f"      Zone {rank+1} (center={cluster_centers[ci]:.3f} mm, "
                  f"{len(cluster_data[ci])} elems) -> {n} plies ({n * mfg_t:.2f} mm)")
        print(f"    Total thickness: {actual_total:.2f} mm")

    return elem_nplies, used_counts, rel_error


def assign_zones_volume_matched(
    sst,
    cluster_data: dict,
    cluster_centers: list[float],
    target_total_thickness: float,
    zone_element_counts: list[int] = None,
    min_distinct_zones: int = 1,
) -> tuple[dict[int, int], tuple[int, ...], float]:
    """Zone-to-ply-count assignment whose total thickness is closest to
    target_total_thickness (usually the FSO total), searched over all
    monotonic assignments of available counts with >= min_distinct_zones
    distinct values. zone_element_counts overrides the per-zone element counts
    (e.g. post-filter). Returns (elem_nplies, best_assignment, rel_error_pct)."""
    from itertools import product

    sorted_ci = sorted(range(len(cluster_centers)), key=lambda i: cluster_centers[i])

    # Element counts per zone (ordered thinnest to thickest)
    if zone_element_counts is not None:
        counts = zone_element_counts
    else:
        counts = []
        for ci in sorted_ci:
            counts.append(len(cluster_data[ci]))

    n_zones = len(counts)
    available = sorted(set(sst.zone_ply_counts) |
                       set(_compute_valid_ply_counts(
                           sst.n_max,
                           [idx for idx, _ in sst.drop_sequence],
                           [p.angle for p in sst.half_plies],
                       )))
    mfg_t = sst.manufacturing_ply_thickness

    # Search all valid monotonic assignments
    best_error = float("inf")
    best_combo = None

    for combo in product(available, repeat=n_zones):
        if list(combo) != sorted(combo):
            continue
        if len(set(combo)) < min_distinct_zones:
            continue
        total = sum(counts[i] * combo[i] * mfg_t for i in range(n_zones))
        error = abs(total - target_total_thickness)
        if error < best_error:
            best_error = error
            best_combo = combo

    if best_combo is None:
        print(f"  WARNING: No assignment found with >= {min_distinct_zones} distinct zones.")
        if _VERBOSE:
            print(f"           Relaxing constraint by 1.")
        return assign_zones_volume_matched(
            sst, cluster_data, cluster_centers,
            target_total_thickness, zone_element_counts,
            min_distinct_zones=min_distinct_zones - 1,
        )

    # Apply the best assignment
    best_total = sum(counts[i] * best_combo[i] * mfg_t for i in range(n_zones))

    ci_to_nplies = {}
    for rank, ci in enumerate(sorted_ci):
        ci_to_nplies[ci] = best_combo[rank]

    # --- Per-element refinement for exact mass matching ---
    refined_nplies = {}
    gap = target_total_thickness - best_total
    available_sorted = sorted(available)

    if abs(gap) >= mfg_t * 0.5:
        n_refined = _refine_candidates(
            gap, cluster_data, ci_to_nplies, available_sorted, mfg_t, refined_nplies
        )
        if _VERBOSE:
            print(f"  Mass refinement: {n_refined} elements adjusted "
              f"({'promoted' if gap > 0 else 'demoted'})")

    elem_nplies = {}
    for ci, elements in cluster_data.items():
        n = ci_to_nplies[ci]
        for eid, _ in elements:
            elem_nplies[eid] = refined_nplies.get(eid, n)

    actual_total = sum(elem_nplies[eid] * mfg_t for eid in elem_nplies)
    rel_error = (actual_total - target_total_thickness) / target_total_thickness * 100

    return elem_nplies, best_combo, rel_error


def assign_zones_perply(
    sst: StackingSequenceTable,
    elem_ply_thicknesses: dict[int, list[float]],
    original_ply_angles: list[float],
    target_total_thickness: float,
    min_distinct_zones: int = 1,
) -> tuple[dict[int, int], tuple[int, ...], float]:
    """Per-ply FSO-aware zone assignment: matches each element's FSO per-ply
    profile to the SST laminate that best approximates it. Unlike total-
    thickness k-means, two elements with the same total but different ply
    distributions (0-dominated vs 90-dominated) land in different zones.
    Returns (elem_nplies, used_ply_counts, rel_error_pct)."""
    mfg_t = sst.manufacturing_ply_thickness
    available = sorted(set(sst.zone_ply_counts))
    eids = list(elem_ply_thicknesses.keys())
    unique_angles = sorted(set(original_ply_angles))
    n_avail = len(available)

    per_angle_sst = {}
    for n in available:
        lam = sst.get_laminate(n)
        ac = {}
        for a in lam:
            ac[a] = ac.get(a, 0) + 1
        per_angle_sst[n] = {a: ac.get(a, 0) * mfg_t for a in unique_angles}

    per_angle_fso = {}
    for eid in eids:
        pt = elem_ply_thicknesses[eid]
        at = {a: 0.0 for a in unique_angles}
        for i, ang in enumerate(original_ply_angles):
            at[ang] += pt[i]
        per_angle_fso[eid] = at

    scores = {}
    for eid in eids:
        fso = per_angle_fso[eid]
        for idx, n in enumerate(available):
            st = per_angle_sst[n]
            mismatch = sum((st[a] - fso[a]) ** 2 for a in unique_angles)
            scores[(eid, idx)] = mismatch

    elem_idx = {eid: min(range(n_avail), key=lambda i: scores[(eid, i)]) for eid in eids}

    current_total = sum(available[elem_idx[eid]] * mfg_t for eid in eids)
    gap = target_total_thickness - current_total

    if _VERBOSE:
        print(f"\n  Per-ply initial assignment (before mass adjustment):")
        print(f"    Total thickness: {current_total:.2f} mm (target: {target_total_thickness:.2f} mm)")
        print(f"    Gap: {gap:+.2f} mm ({gap / target_total_thickness * 100:+.2f}%)")

    init_dist = {}
    for eid in eids:
        n = available[elem_idx[eid]]
        init_dist[n] = init_dist.get(n, 0) + 1
    if _VERBOSE:
        for n in sorted(init_dist.keys()):
            print(f"    {n:>2} plies: {init_dist[n]:>6} elements")

    total_adjusted = 0
    for iteration in range(n_avail):
        current_total = sum(available[elem_idx[eid]] * mfg_t for eid in eids)
        gap = target_total_thickness - current_total
        if abs(gap) < mfg_t * 0.5:
            break

        if gap > 0:
            cands = []
            for eid in eids:
                ci = elem_idx[eid]
                if ci < n_avail - 1:
                    ni = ci + 1
                    cands.append((scores[(eid, ni)] - scores[(eid, ci)], eid, ci, ni))
            if not cands:
                break
            cands.sort()
            n_moved = 0
            for _, eid, oi, ni in cands:
                if gap < mfg_t * 0.5:
                    break
                gap -= (available[ni] - available[oi]) * mfg_t
                elem_idx[eid] = ni
                n_moved += 1
            total_adjusted += n_moved
            if _VERBOSE:
                print(f"    Iteration {iteration + 1}: promoted {n_moved} elements "
                  f"(remaining gap: {gap:+.2f} mm)")
        else:
            cands = []
            for eid in eids:
                ci = elem_idx[eid]
                if ci > 0:
                    pi = ci - 1
                    cands.append((scores[(eid, pi)] - scores[(eid, ci)], eid, ci, pi))
            if not cands:
                break
            cands.sort()
            n_moved = 0
            for _, eid, oi, ni in cands:
                if gap > -mfg_t * 0.5:
                    break
                gap -= (available[ni] - available[oi]) * mfg_t
                elem_idx[eid] = ni
                n_moved += 1
            total_adjusted += n_moved
            if _VERBOSE:
                print(f"    Iteration {iteration + 1}: demoted {n_moved} elements "
                  f"(remaining gap: {gap:+.2f} mm)")

    if _VERBOSE:
        print(f"    Total elements adjusted: {total_adjusted}")

    elem_nplies = {eid: available[elem_idx[eid]] for eid in eids}
    actual_total = sum(elem_nplies[eid] * mfg_t for eid in eids)
    rel_error = (actual_total - target_total_thickness) / target_total_thickness * 100
    used_counts = tuple(sorted(set(elem_nplies.values())))

    n_distinct = len(set(elem_nplies.values()))
    if n_distinct < min_distinct_zones:
        print(f"  WARNING: Per-ply assignment uses {n_distinct} distinct zones "
              f"(min requested: {min_distinct_zones})")

    return elem_nplies, used_counts, rel_error


def _refine_candidates(gap, cluster_data, ci_to_nplies, available_sorted, mfg_t, refined_nplies):
    candidates = []
    if gap > 0:
        for ci, elements in cluster_data.items():
            current_n = ci_to_nplies[ci]
            idx = available_sorted.index(current_n)
            if idx < len(available_sorted) - 1:
                next_n = available_sorted[idx + 1]
                for eid, fso_t in elements:
                    candidates.append((-fso_t, eid, current_n, next_n))
        candidates.sort()
    else:
        for ci, elements in cluster_data.items():
            current_n = ci_to_nplies[ci]
            idx = available_sorted.index(current_n)
            if idx > 0:
                prev_n = available_sorted[idx - 1]
                for eid, fso_t in elements:
                    candidates.append((fso_t, eid, current_n, prev_n))
        candidates.sort()

    n_refined = 0
    for sort_key, eid, old_n, new_n in candidates:
        if abs(gap) < mfg_t * 0.5:
            break
        refined_nplies[eid] = new_n
        gap -= (new_n - old_n) * mfg_t
        n_refined += 1

    return n_refined


# ==============================================================================
#  Visualization
# ==============================================================================

def _draw_stiffener_outline(ax, stiffener_xy, linewidth=0.8):
    """Draw stiffener boundary as a black contour outline."""
    import numpy as np
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


def plot_sst_zones(X, Y, elem_nplies, eids, sst, save_path="sst_zones.png",
                   stiffener_xy=None, stiffener_base_nplies=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    zc = sorted(sst.zone_ply_counts)
    nz = len(zc)
    c2i = {n: i for i, n in enumerate(zc)}
    colors = [c2i[elem_nplies[eid]] for eid in eids]

    cmap = plt.get_cmap("RdYlBu_r", nz)
    norm = mcolors.BoundaryNorm(np.arange(nz + 1) - 0.5, nz)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(X, Y, c=colors, s=15, cmap=cmap, norm=norm, marker="s", edgecolors="none")
    if stiffener_xy is not None:
        _draw_stiffener_outline(ax, stiffener_xy)
    cbar = plt.colorbar(sc, ticks=np.arange(nz))
    cbar.ax.set_yticklabels([f"{n} plies ({n * sst.manufacturing_ply_thickness:.2f} mm)" for n in zc])
    cbar.set_label("SST Zone", rotation=270, labelpad=20, fontweight="bold")
    ax.set_aspect("equal")
    ax.margins(0)
    ax.set_title(f"SST Zone Distribution ({nz} zones)", fontsize=14, pad=15)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_sst_zones_scaled(X, Y, elem_nplies, eids, sst_original, sst_scaled,
                          save_path="sst_zones_scaled.png",
                          stiffener_xy=None, stiffener_base_nplies=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    zc = sorted(set(elem_nplies.values()))
    nz = len(zc)
    c2i = {n: i for i, n in enumerate(zc)}
    colors = [c2i[elem_nplies[eid]] for eid in eids]

    cmap = plt.get_cmap("RdYlBu_r", nz)
    norm = mcolors.BoundaryNorm(np.arange(nz + 1) - 0.5, nz)

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(X, Y, c=colors, s=15, cmap=cmap, norm=norm, marker="s", edgecolors="none")
    if stiffener_xy is not None:
        _draw_stiffener_outline(ax, stiffener_xy)
    cbar = plt.colorbar(sc, ticks=np.arange(nz))
    cbar.ax.set_yticklabels([
        f"{n} plies ({n * sst_original.manufacturing_ply_thickness:.2f} mm)"
        f" \u2192 {n} plies, scaled ({n * sst_scaled.manufacturing_ply_thickness:.2f} mm)"
        for n in zc
    ])
    cbar.set_label("SST Zone (original \u2192 volume-scaled)", rotation=270, labelpad=20,
                   fontweight="bold")
    ax.set_aspect("equal")
    ax.margins(0)
    ax.set_title(f"SST Zone Distribution \u2014 Volume-Scaled ({nz} zones)", fontsize=14, pad=15)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_sst_zones_vc_evolution(
    X, Y, eids,
    elem_nplies_pre, zone_ply_counts_pre, mfg_t_pre, vf_pre,
    elem_nplies_post, zone_ply_counts_post, mfg_t_post, vf_post,
    save_path="sst_zones_vc_evolution.png",
    stiffener_xy=None, stiffener_base_nplies=None,
):
    """Side-by-side SST zone distribution before vs after VF compensation."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def _plot_panel(ax, elem_nplies, zpc, mfg_t, vf, title):
        zc = sorted(zpc)
        nz = len(zc)
        c2i = {n: i for i, n in enumerate(zc)}
        colors = [c2i.get(elem_nplies.get(eid, zc[0]), 0) for eid in eids]

        cmap = plt.get_cmap("RdYlBu_r", nz)
        norm = mcolors.BoundaryNorm(np.arange(nz + 1) - 0.5, nz)

        sc = ax.scatter(X, Y, c=colors, s=12, cmap=cmap, norm=norm,
                        marker="s", edgecolors="none")
        if stiffener_xy is not None:
            _draw_stiffener_outline(ax, stiffener_xy, linewidth=0.5)
        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(nz), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(
            [f"{n} pl ({n * mfg_t:.2f} mm)" for n in zc], fontsize=8)
        ax.set_aspect("equal")
        ax.margins(0)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, linestyle=":", alpha=0.3)
        return sc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    _plot_panel(
        ax1, elem_nplies_pre, zone_ply_counts_pre, mfg_t_pre, vf_pre,
        f"Before VF Compensation\nVF={vf_pre:.4f}, "
        f"{len(sorted(set(zone_ply_counts_pre)))} zones",
    )
    _plot_panel(
        ax2, elem_nplies_post, zone_ply_counts_post, mfg_t_post, vf_post,
        f"After VF Compensation\nVF={vf_post:.4f}, "
        f"{len(sorted(set(zone_ply_counts_post)))} zones",
    )

    fig.suptitle("SST Zone Evolution \u2014 Volume-Fraction Compensation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


# Fixed fibre-angle colors matching the classic ply-book convention
# (0 = blue, 45 = beige, 90 = green, -45 = red); other angles are assigned
# from tab10 deterministically per plot.
_ANGLE_COLORS = {
    0.0: "#1f63a8",
    45.0: "#e8e4cf",
    90.0: "#2e8b3d",
    -45.0: "#c0392b",
}


def _build_angle_color_map(angles):
    import matplotlib.pyplot as plt
    cmap = dict(_ANGLE_COLORS)
    extras = [a for a in sorted(set(angles)) if a not in cmap]
    tab = plt.get_cmap("tab10")
    for i, a in enumerate(extras):
        cmap[a] = tab(i % 10)
    return cmap


def _text_color_for(bg_hex):
    """Black text on light fills, white on dark ones."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(bg_hex)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"


def _ply_present(p, n, n_max):
    nd = (n_max - n) // 2
    return p.is_continuous or (p.drop_rank is not None and p.drop_rank > nd)


def plot_sst_table(sst, save_path="sst_table.png"):
    """Condensed symmetric SST view (Irisarri Fig. 2, right). Rows = half-stack
    positions (outer top, mid-plane bottom); columns = zone ply counts
    (thinnest left). Present cells colored by fibre angle, dropped ones blank."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    counts = sorted(sst.zone_ply_counts)
    plies = sst.half_plies
    n_rows = len(plies)
    n_cols = len(counts)
    if n_rows == 0 or n_cols == 0:
        return
    color_map = _build_angle_color_map([p.angle for p in plies])

    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 3.2, 0.45 * n_rows + 2.0))

    for r, p in enumerate(plies):
        y = n_rows - 1 - r  # row 0 (outer) at the top
        for c, n in enumerate(counts):
            if _ply_present(p, n, sst.n_max):
                fc = color_map[p.angle]
                ax.add_patch(Rectangle((c, y), 1, 1, facecolor=fc,
                                       edgecolor="black", linewidth=0.6))
                ax.text(c + 0.5, y + 0.5, f"{p.angle:+.0f}\u00b0",
                        ha="center", va="center", fontsize=8,
                        color=_text_color_for(fc))
            else:
                ax.add_patch(Rectangle((c, y), 1, 1, facecolor="white",
                                       edgecolor="lightgray",
                                       linewidth=0.5))
        tag = "C" if p.is_continuous else f"D{p.drop_rank}"
        ax.text(-0.15, y + 0.5, tag, ha="right", va="center", fontsize=8,
                family="monospace")

    for c, n in enumerate(counts):
        ax.text(c + 0.5, n_rows + 0.55, f"{n}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(c + 0.5, n_rows + 0.12,
                f"{n * sst.manufacturing_ply_thickness:.2f}",
                ha="center", va="bottom", fontsize=6.5, color="dimgray")
    ax.text(-0.15, n_rows + 0.12, "t (mm)", ha="right", va="bottom",
            fontsize=6.5, color="dimgray")

    ax.text(n_cols / 2, -0.35, "mid-plane / symmetric mirror below",
            ha="center", va="top", fontsize=8, style="italic",
            color="dimgray")
    ax.plot([0, n_cols], [0, 0], color="black", linewidth=1.4,
            linestyle=(0, (6, 3)))

    handles = [Patch(facecolor=color_map[a], edgecolor="black",
                     label=f"{a:+.0f}\u00b0")
               for a in sorted(color_map) if any(p.angle == a for p in plies)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, title="Fibre angle")

    ax.set_xlim(-1.2, n_cols + 0.2)
    ax.set_ylim(-1.0, n_rows + 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Stacking Sequence Table \u2014 condensed (symmetric half)\n"
        f"guide {sst.n_max} plies, DF={sst.doubling_factor}x, "
        f"t_ply={sst.manufacturing_ply_thickness:.3f} mm",
        fontsize=11, pad=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")


def plot_sst_side_view(sst, save_path="sst_side_view.png"):
    """Side-view of the half-stack across zones, thickest (left) to thinnest
    (right). Flat mid-plane at the bottom, stepped outer surface on top. Bands
    colored by fibre angle; dropped plies taper out in the transition, following
    the SST drop ranks exactly."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Patch

    counts = sorted(sst.zone_ply_counts, reverse=True)  # thickest first
    plies = sst.half_plies
    t = sst.manufacturing_ply_thickness
    n_zones = len(counts)
    if n_zones == 0 or len(plies) == 0:
        return
    color_map = _build_angle_color_map([p.angle for p in plies])

    W, T = 1.0, 0.35  # zone width, taper-transition width (x units)
    x_start = [k * (W + T) for k in range(n_zones)]
    x_end = [x + W for x in x_start]

    def _bottom_y(i, n):
        """y of ply i's bottom face in the zone with count n (mid-plane at
        the bottom: present plies closer to the mid-plane sit below i)."""
        return t * sum(1 for j in range(i + 1, len(plies))
                       if _ply_present(plies[j], n, sst.n_max))

    fig, ax = plt.subplots(figsize=(2.2 * n_zones + 2.5, 4.0))

    for i, p in enumerate(plies):
        # presence is monotone in n: present in the thickest zones, then
        # dropped for good \u2014 find the last zone (in thick->thin order)
        # this ply still exists in.
        last = -1
        for k, n in enumerate(counts):
            if _ply_present(p, n, sst.n_max):
                last = k
            else:
                break
        if last < 0:
            continue

        top_pts, bot_pts = [], []
        for k in range(last + 1):
            b = _bottom_y(i, counts[k])
            top_pts += [(x_start[k], b + t), (x_end[k], b + t)]
            bot_pts += [(x_start[k], b), (x_end[k], b)]

        if last < n_zones - 1:
            # taper wedge: top and bottom edges meet at the ply's virtual
            # seat height in the next (thinner) zone.
            y_meet = _bottom_y(i, counts[last + 1])
            top_pts.append((x_start[last + 1], y_meet))

        ax.add_patch(Polygon(top_pts + bot_pts[::-1], closed=True,
                             facecolor=color_map[p.angle],
                             edgecolor="black", linewidth=0.5))

    for k, n in enumerate(counts):
        ax.text((x_start[k] + x_end[k]) / 2, -0.14 * t * len(plies),
                f"{n} plies\n{n * t:.2f} mm",
                ha="center", va="top", fontsize=9)

    handles = [Patch(facecolor=color_map[a], edgecolor="black",
                     label=f"{a:+.0f}\u00b0")
               for a in sorted(color_map) if any(p.angle == a for p in plies)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9, title="Fibre angle")

    y_max = t * sum(1 for p in plies
                    if _ply_present(p, counts[0], sst.n_max))
    ax.set_xlim(-0.1, x_end[-1] + 0.1)
    ax.set_ylim(-0.32 * t * len(plies), y_max * 1.18)
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(["0", f"{y_max:.2f}"])
    ax.set_ylabel("half-stack thickness (mm)")
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        "SST half-stack side view \u2014 thickest \u2192 thinnest zone\n"
        "(flat mid-plane at bottom, outer surface on top edge)",
        fontsize=11, pad=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if _VERBOSE:
        print(f"  Saved: {save_path}")
