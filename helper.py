import os
import sys
import subprocess
import time
from SST import StackingSequenceTable
import re
import shutil
import numpy as np

# OptiStruct solver executable; absent means analysis-only
OPTISTRUCT_EXE = r"C:\Program Files\Altair\2025.1\hwsolvers\scripts\optistruct.bat"


class _Tee:
    """Duplicate writes to a file and the original stdout"""
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


_VERBOSE = True


def set_verbose(v: bool):
    global _VERBOSE
    _VERBOSE = v


# ==============================================================================
#  OptiStruct .fem / .sh parsing
# ==============================================================================

def _is_smear_pcomp(line: str) -> bool:
    return line.strip().startswith("PCOMP") and "SMEAR" in line.upper()


def _parse_pcomp_ply_line(c: str) -> list[tuple[float, float]]:
    plies = []
    for offset in (16, 48):
        chunk = c[offset:offset + 24] if len(c) >= offset + 24 else c[offset:]
        if not chunk.strip():
            continue
        ts = chunk[0:8].strip()
        an = chunk[8:16].strip() if len(chunk) >= 16 else ""
        if ts and an:
            try:
                plies.append((float(an), float(ts)))
            except ValueError:
                pass
    return plies


def parse_initial_plies(fem_file_path: str) -> list[tuple[float, float]]:
    """Extract ply (angle, thickness) pairs from a .fem, handling both PCOMPG
    and PCOMP SMEAR cards. SMEAR returns the explicit list (not mirrored);
    use symmetrize_smear_plies() for the symmetric equivalent."""
    plies = []
    with open(fem_file_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_smear_pcomp(line):
            i += 1
            while i < len(lines):
                c = lines[i]
                if c.startswith(("$", "*")) or c.strip() == "" or not c[0:8].strip().startswith("+"):
                    break
                ply_batch = _parse_pcomp_ply_line(c)
                if not ply_batch:
                    break
                plies.extend(ply_batch)
                i += 1
            break
        if line.startswith("PCOMPG") or (line.startswith("PCOMP") and "SMEAR" not in line.upper()):
            i += 1
            while i < len(lines):
                c = lines[i]
                if c.startswith(("$", "*")) or c.strip() == "":
                    break
                s = c.lstrip()
                if s and not c[0].isspace() and s[0] not in ("+", "*", "$"):
                    if s.split()[0].isupper() and not s.split()[0][0].isdigit():
                        break
                try:
                    if len(c) >= 40:
                        ts, an = c[24:32].strip(), c[32:40].strip()
                        if ts and an:
                            plies.append((float(an), float(ts)))
                        else:
                            break
                    else:
                        break
                except (ValueError, Exception):
                    break
                i += 1
            break
        i += 1
    return plies


def detect_smear(fem_file_path: str) -> bool:
    with open(fem_file_path, "r") as f:
        for line in f:
            if _is_smear_pcomp(line):
                return True
    return False


def symmetrize_smear_plies(plies: list[tuple[float, float]]) -> list[tuple[float, float]]:
    half = [(a, t / 2.0) for a, t in plies]
    return half + half[::-1]


def expand_smear_thicknesses(
    elem_ply_thicknesses: dict[int, list[float]],
    smeared_angles: list[float],
    base_ply_thickness: float,
) -> dict[int, list[float]]:
    """Map SMEAR per-angle-group thicknesses to the symmetric per-ply list used
    by the rest of the pipeline. SMEAR .sh has one value per unique |angle|
    group (e.g. 3 for {0, +/-45, 90}); output is 2*len(smeared_angles) values
    ordered to match symmetrize_smear_plies()."""
    unique_groups = {}
    for angle in smeared_angles:
        unique_groups.setdefault(abs(angle), []).append(angle)

    # the .sh writes one column per unique |angle| in PCOMP ply order,
    # not in ascending |angle|; sorting here swaps groups whenever the
    # card is not already ordered that way
    sorted_keys = []
    for _a in smeared_angles:
        if abs(_a) not in sorted_keys:
            sorted_keys.append(abs(_a))
    group_to_val_idx = {key: idx for idx, key in enumerate(sorted_keys)}

    result = {}
    for eid, group_vals in elem_ply_thicknesses.items():
        half_vals = [group_vals[group_to_val_idx[abs(angle)]] / 2.0
                     for angle in smeared_angles]
        result[eid] = half_vals + half_vals[::-1]

    return result


def parse_ply_thicknesses(sh_file_path: str, base_ply_thickness: float) -> dict[int, list[float]]:
    """Parse OptiStruct .sh file for per-element, per-ply thicknesses."""
    with open(sh_file_path, "r") as f:
        lines = f.readlines()
    header = lines[1].split()
    n_elements, n_plies = int(header[0]), int(header[-1])
    if _VERBOSE:
        print(f"\n  Parsing .sh: {n_elements} elements, {n_plies} plies/element")
    em = {}
    i = 2
    for _ in range(n_elements):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        eid, nv = int(parts[0]), int(parts[1])
        vals = [float(lines[i + 1 + j].strip()) for j in range(nv)]
        em[eid] = (np.array(vals) * base_ply_thickness).tolist()
        i += nv + 1
    if _VERBOSE:
        print(f"  Parsed {len(em)} elements")
    return em


def load_mesh_data(fem_file_path, thickness_map):
    """
    Reads .fem to find element centers and aligns them with parsed thickness data.
    """
    nodes = {}
    element_centers = []
    final_thickness = []

    with open(fem_file_path, 'r') as f:
        for line in f:
            # Parse GRID (Nodes)
            if line.startswith('GRID'):
                # Handle fixed-width (Small field) format
                nid = int(line[8:16].strip())
                x = clean_optistruct_float(line[24:32].strip())
                y = clean_optistruct_float(line[32:40].strip())
                nodes[nid] = [x, y]

            # Parse CQUAD4 (Elements)
            elif line.startswith('CQUAD4'):
                eid = int(line[8:16].strip())
                # Only process if we have thickness data for this element
                if eid in thickness_map:
                    # Get IDs of the 4 nodes
                    nids = [int(line[24:32].strip()), int(line[32:40].strip()),
                            int(line[40:48].strip()), int(line[48:56].strip())]

                    # Calculate center (average of nodes)
                    coords = [nodes[n] for n in nids if n in nodes]
                    if len(coords) == 4:
                        center = np.mean(coords, axis=0)
                        element_centers.append(center)
                        final_thickness.append(thickness_map[eid])

    centers = np.array(element_centers)
    return centers[:, 0], centers[:, 1], final_thickness


def clean_optistruct_float(val_str):
    """Converts OptiStruct 'short-form' scientific notation to standard python floats."""
    val_str = val_str.strip()
    if not val_str:
        return 0.0

    cleaned = re.sub(r'(?<=\d)(?=[+-])', 'E', val_str)

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_total_element_thickness(sh_file_path, base_ply_thickness):
    """
    Parse an OptiStruct .sh (shape/property) file from free-size optimization.

    The .sh file contains per-element, per-ply thickness multipliers (0..1).
    Structure:
        Line 1: version string
        Line 2: header (n_elements, n_iterations, ..., n_plies_per_element)
        Then for each element: a line with (element_id, n_values),
        followed by n_values lines of thickness multipliers.

    Returns:
        dict: {element_id: [ply1_mult, ply2_mult, ...], ...}
              Multipliers in range [0, 1] where 1 = full nominal ply thickness.
    """
    with open(sh_file_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = lines[1].split()
    n_elements = int(header[0])
    n_plies = int(header[-1])

    elem_thickness_map = {}
    i = 2  # start after header
    for _ in range(n_elements):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        eid = int(parts[0])
        n_vals = int(parts[1])
        values = []
        for j in range(n_vals):
            values.append(float(lines[i + 1 + j].strip()))
        values = float(sum(values))*base_ply_thickness
        elem_thickness_map[eid] = values
        i += n_vals + 1

    return elem_thickness_map


def parse_per_angle_thickness(
    sh_file_path: str,
    ply_angles: list[float],
    base_ply_thickness: float,
    n_elements: int,
) -> dict[float, dict[int, float]]:
    """Parse .sh file and return thickness per element grouped by angle.

    Returns:
        {angle: {element_id: thickness_at_that_angle}}
    """
    unique_angles = sorted(set(round(a) for a in ply_angles))
    angle_to_ply_indices = {ua: [] for ua in unique_angles}
    for i, a in enumerate(ply_angles):
        key = round(a)
        angle_to_ply_indices[key].append(i)

    with open(sh_file_path, "r") as f:
        lines = f.readlines()

    per_angle = {ua: {} for ua in unique_angles}
    idx = 2
    for _ in range(n_elements):
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        eid, nv = int(parts[0]), int(parts[1])
        mults = [float(lines[idx + 1 + j].strip()) for j in range(nv)]
        for ua in unique_angles:
            t = sum(mults[pi] for pi in angle_to_ply_indices[ua] if pi < nv)
            per_angle[ua][eid] = t * base_ply_thickness
        idx += nv + 1

    return per_angle


def _savefig_locked_retry(plt_mod, save_path, retries=4, **kwargs):
    """savefig robust to transient Windows file locks (viewer / AV / sync).

    A PNG left open in an image viewer while the pipeline regenerates it
    raises OSError(22) and would otherwise kill a multi-hour run at the
    plotting stage.  Retry briefly, then warn and skip the plot rather
    than crash.
    """
    import time as _time
    for attempt in range(retries):
        try:
            plt_mod.savefig(save_path, **kwargs)
            return True
        except OSError:
            _time.sleep(1.0 + attempt)
    print(f"  WARNING: could not save '{save_path}' (file locked by another "
          f"process, e.g. an open image viewer) - plot skipped.")
    return False


def setup_run_log(log_path="run_log.txt"):
    """Redirect stdout to both the console and a log file."""
    global _run_log_tee
    _run_log_tee = _Tee(log_path)
    sys.stdout = _run_log_tee

# ==============================================================================
#  .fem export
# ==============================================================================

def _build_stiffener_skin_plies(
        sst: StackingSequenceTable,
        mat_id: int,
        source_skin_layup: list[dict] | None = None,
        stiffener_base_nplies: int | None = None,
        ply_thickness_override: float | None = None,
) -> list[tuple[int, float, float]]:
    """Build replacement skin plies for stiffener PCOMPG cards.

    GuidePly mode (source_skin_layup provided):
        Angles/ply-count from the flat skin, but thickness scaled so
        the total equals the current DfM stiffener-base thickness scaled by
        the same factor as the flat skin.
    SST mode (source_skin_layup is None):
        Full laminate from sst.get_laminate(stiffener_base_nplies)
        with manufacturing ply thickness, or *ply_thickness_override* when
        given (e.g. the FSO_exact model has no minimum ply thickness, so it
        uses the ideal unclamped sub-ply thickness base_t / DF).

    Returns list of (mid, thickness, angle).
    """
    if source_skin_layup is not None:
        source_total = sum(p["T"] for p in source_skin_layup)
        if source_total <= 0:
            return []
        base_nplies = stiffener_base_nplies or sst.n_max
        scale = (base_nplies / sst.n_max) if sst.n_max > 0 else 1.0
        return [(int(p["mid"]), float(p["T"]) * scale, float(p["theta"]))
                for p in source_skin_layup]
    n = stiffener_base_nplies or max(sst.zone_ply_counts)
    lam = sst.get_laminate(n)
    t_ply = (ply_thickness_override if ply_thickness_override
             else sst.manufacturing_ply_thickness)
    return [(mat_id, t_ply, angle) for angle in lam]


def _format_pcompg_skin_ply(gplyid: int, mid: int, t: float, theta: float) -> str:
    """Format a skin-ply continuation line for a PCOMPG card."""
    t_str = f"{t:.4f}" if t < 10 else f"{t:.3f}"
    return (f"        {gplyid:>8d}{mid:>8d}{t_str:>8s}"
            f"{theta:>8.1f}{'YES':>8s}\n")


def _parse_pcompg_ply(raw: str) -> dict | None:
    """Parse one small-field PCOMPG continuation ply.

    Uses fixed-width columns first so previously generated lines like
    ``1       10.125`` are still interpreted as MID=1, T=0.125 instead of
    a single malformed whitespace token. Falls back to whitespace parsing
    for legacy/free-field-like lines.
    """
    if len(raw) >= 40:
        gplyid_s = raw[8:16].strip()
        mid_s = raw[16:24].strip()
        t_s = raw[24:32].strip()
        theta_s = raw[32:40].strip()
        if gplyid_s and mid_s and t_s and theta_s:
            try:
                return {
                    "gplyid": int(gplyid_s),
                    "mid": int(mid_s),
                    "T": float(t_s),
                    "theta": float(theta_s),
                }
            except ValueError:
                pass

    tokens = raw.strip().split()
    if tokens and tokens[0] == "+":
        tokens = tokens[1:]
    if len(tokens) < 4:
        return None
    try:
        return {
            "gplyid": int(tokens[0]),
            "mid": int(tokens[1]),
            "T": float(tokens[2]),
            "theta": float(tokens[3]),
        }
    except ValueError:
        return None


def _collect_stiffener_ply_lines(
        lines: list[str], i: int, mat_id: int,
) -> tuple[list[str], int, bool]:
    """Collect non-skin continuation lines from a stiffener PCOMPG card.

    Returns (stiffener_ply_lines, next_index, had_skin) where
    *stiffener_ply_lines* are the verbatim continuation lines whose
    MID != mat_id, *next_index* is the index of the first line after the
    card, and *had_skin* says whether the card contained any skin plies
    (MID == mat_id).  Cards without skin plies (flat-side attached
    stiffener shells) must be preserved verbatim — no skin injection.
    """
    stiffener_ply_lines = []
    had_skin = False
    j = i + 1
    while j < len(lines):
        raw = lines[j].rstrip("\r\n")
        stripped = raw.strip()
        if stripped and not stripped.startswith("$") and \
                (raw[0] == " " or raw[0] == "+"):
            ply = _parse_pcompg_ply(raw)
            if ply is None:
                raise ValueError(f"Could not parse PCOMPG continuation line: {raw!r}")
            if ply["mid"] != mat_id:
                stiffener_ply_lines.append(lines[j])
            else:
                had_skin = True
            j += 1
        else:
            break
    return stiffener_ply_lines, j, had_skin


def export_fem(
        source_fem_path: str,
        output_fem_path: str,
        sst: StackingSequenceTable,
        elem_nplies: dict[int, int] | None = None,
        mat_id: int = 1,
        optimize: bool = False,
        elem_half_plies: dict[int, tuple[int, ...]] | None = None,
        stiffener_eids: set[int] | None = None,
        stiffener_base_nplies: int | None = None,
        stiffener_ply_info: dict[int, list[dict]] | None = None,
        stiffener_skin_overrides: dict[int, list[tuple[int, float, float]]] | None = None,
        source_skin_layup: list[dict] | None = None,
        flat_stiffener_pids: set[int] | None = None,
        verbose: bool = False,
):
    """Write a .fem with SST-discretized PCOMP properties for static analysis.

    flat_stiffener_pids: PIDs of flat-side attached stiffener shells. Their
    PCOMPG cards are already stripped to stiffener plies only, so they are
    copied verbatim (no skin detection/injection). This is required because the
    material-based skin test misfires when the stiffener ply shares the skin
    material.

    optimize=False: strip the optimization cards, replace the PCOMPG with one
    PCOMP per SST zone (PIDs from 100), and re-point each CQUAD4 to its zone.
    optimize=True: replace the PCOMPG with a single full-layup PCOMP (sst.n_max)
    and keep the optimization cards and CQUAD4 PIDs. Everything else (GRIDs,
    loads, SPCs, MATs, ...) is preserved.

    elem_half_plies, when given, exports one PCOMP per unique ply-subset for
    per-angle directional shapes; otherwise elem_nplies gives the zone counts."""
    with open(source_fem_path, "r") as f:
        lines = f.readlines()

    original_pid = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("PCOMPG") or stripped.startswith("PCOMP"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    original_pid = int(parts[1])
                except ValueError:
                    pass
            break

    if optimize:
        # --- OPTIMIZE = TRUE BRANCH ---
        # Replace PCOMPG/PCOMP with a single PCOMP representing the full SST layup

        n_plies = sst.n_max
        pid = original_pid if original_pid is not None else 2
        lam = sst.get_laminate(n_plies)
        t_ply = sst.manufacturing_ply_thickness
        t_total = n_plies * t_ply
        half_lam = lam[:len(lam) // 2]

        if verbose:
            print(f"\n  Optimize Mode: Generating single PCOMP for full SST layup")
            print(f"    {n_plies} plies ({t_total:.2f} mm) -> PID {pid}: {lam}")

        pcomp_lines = []
        pcomp_lines.append(f"$$ SST Full Layup: {n_plies} plies (SYM), "
                           f"t = {t_total:.3f} mm\n")
        pcomp_lines.append(_format_pcomp_header(pid, sym=True))

        for angle in half_lam:
            pcomp_lines.append(_format_pcomp_ply(mat_id, t_ply, angle))

        pcomp_lines.append("\n")

        output_lines = []
        in_pcompg = False
        pcomp_inserted = False

        i = 0
        while i < len(lines):
            line = lines[i]
            raw = line.rstrip("\r\n")

            # --- Replace PCOMPG or PCOMP block ---
            if raw.startswith("PCOMPG") or raw.startswith("PCOMP"):
                in_pcompg = True
                if not pcomp_inserted:
                    output_lines.append("$$\n")
                    output_lines.append("$$  PCOMP Data (SST full layup)\n")
                    output_lines.append("$$\n")
                    output_lines.extend(pcomp_lines)
                    pcomp_inserted = True
                i += 1
                continue

            if in_pcompg:
                stripped = raw.strip()
                if stripped and not stripped.startswith("$") and \
                        (raw[0] == " " or raw[0] == "+"):
                    i += 1
                    continue
                else:
                    in_pcompg = False

            # --- Pass through everything else exactly as-is ---
            output_lines.append(line)
            i += 1

        with open(output_fem_path, "w") as f:
            f.writelines(output_lines)

        if verbose:
            print(f"\n  Written: {output_fem_path}")
            print(f"  Replaced PCOMPG with single PCOMP (PID {pid})")

    else:
        # --- OPTIMIZE = FALSE BRANCH ---
        # Discretize into zones and strip optimization cards

        pid_base = 100
        t_ply = sst.manufacturing_ply_thickness

        if elem_half_plies is not None:
            # ==========================================================
            # PER-ANGLE MODE: each unique half-ply subset gets its own
            # PCOMP property.  Allows different angles to have different
            # ply shapes.
            # ==========================================================
            unique_subsets = sorted(set(elem_half_plies.values()),
                                    key=lambda s: (len(s), s))
            subset_pid = {}
            eid_to_pid = {}
            pcomp_lines = []

            for i, subset in enumerate(unique_subsets):
                pid = pid_base + i
                subset_pid[subset] = pid

                angles = [sst.half_plies[idx].angle for idx in subset]
                n_total = len(angles) * 2  # SYM doubles the half-stack
                t_total = n_total * t_ply

                pcomp_lines.append(
                    f"$$ Directional Zone {i+1}: {n_total} plies (SYM), "
                    f"t = {t_total:.3f} mm\n"
                )
                pcomp_lines.append(_format_pcomp_header(pid, sym=True))
                for angle in angles:
                    pcomp_lines.append(_format_pcomp_ply(mat_id, t_ply, angle))
                pcomp_lines.append("\n")

            for eid, subset in elem_half_plies.items():
                eid_to_pid[eid] = subset_pid[subset]

            if verbose:
                print(f"\n  Directional Zone -> Property ID mapping:")
                for subset, pid in sorted(subset_pid.items(), key=lambda x: x[1]):
                    angles = [sst.half_plies[idx].angle for idx in subset]
                    n_total = len(angles) * 2
                    print(f"    PID {pid}: {n_total} plies, angles={angles}")

        else:
            # ==========================================================
            # ZONE MODE: original zone-based export
            # ==========================================================
            if source_skin_layup is not None:
                # Guide baseline mode: uniform scaled original skin layup
                # for every non-stiffener element.
                zone_pid = {0: pid_base}

                if verbose:
                    total_t = sum(p["T"] for p in source_skin_layup)
                    print(f"\n  Zone -> Property ID mapping:")
                    print(f"    Guide baseline skin: {len(source_skin_layup)} plies, "
                          f"t = {total_t:.3f} mm -> PID {pid_base}")

                pcomp_lines = []
                total_base_T = sum(p["T"] for p in source_skin_layup)
                pcomp_lines.append(
                    f"$$ Guide baseline skin: {len(source_skin_layup)} plies, "
                    f"t = {total_base_T:.3f} mm\n"
                )
                pcomp_lines.append(_format_pcomp_header(pid_base, sym=False))
                for ply in source_skin_layup:
                    pcomp_lines.append(
                        _format_pcomp_ply(ply["mid"], ply["T"], ply["theta"]))
                pcomp_lines.append("\n")

                eid_to_pid = {}
                for eid, n in elem_nplies.items():
                    eid_to_pid[eid] = pid_base
            else:
                zone_pid = {}
                for i, n in enumerate(sorted(sst.zone_ply_counts)):
                    zone_pid[n] = pid_base + i

                if verbose:
                    print(f"\n  Zone -> Property ID mapping:")
                    for n in sorted(zone_pid):
                        pid = zone_pid[n]
                        lam = sst.get_laminate(n, n_max_override=sst.n_max)
                        t_mm = n * t_ply
                        print(f"    {n:>2} plies ({t_mm:.2f} mm) -> PID {pid}: {lam}")

                pcomp_lines = []
                for n_plies in sorted(zone_pid):
                    pid = zone_pid[n_plies]
                    lam = sst.get_laminate(n_plies)
                    half_lam = lam[:len(lam) // 2]

                    pcomp_lines.append(f"$$ SST Zone: {n_plies} plies (SYM), "
                                       f"t = {n_plies * t_ply:.3f} mm\n")
                    pcomp_lines.append(_format_pcomp_header(pid, sym=True))

                    for angle in half_lam:
                        pcomp_lines.append(_format_pcomp_ply(mat_id, t_ply, angle))

                    pcomp_lines.append("\n")

                eid_to_pid = {}
                for eid, n in elem_nplies.items():
                    if stiffener_eids and eid in stiffener_eids:
                        continue
                    eid_to_pid[eid] = zone_pid[n]

        # Stiffener PCOMPG rewrite: keep the source cards but replace their skin
        # plies (MID == mat_id) with the correct base-skin layup (GuidePly ->
        # VF-scaled flat skin, SST -> max-zone laminate). Stiffener plies kept.
        new_stiffener_skin = _build_stiffener_skin_plies(
            sst, mat_id, source_skin_layup, stiffener_base_nplies)

        # In flat-side mode the stiffener PCOMPGs are copied verbatim, so this
        # base skin is computed but never applied — don't report it.
        if verbose and new_stiffener_skin and not flat_stiffener_pids:
            _total_t = sum(p[1] for p in new_stiffener_skin)
            _angles = [p[2] for p in new_stiffener_skin]
            print(f"  Stiffener base skin: {len(new_stiffener_skin)} plies, "
                  f"t = {_total_t:.3f} mm, angles={_angles}")

        _preserved_stiffener_pcompgs = 0

        # --- Process the source file ---
        output_lines = []
        skip_until_blank = False  # for multi-line card removal
        in_pcompg = False
        removed_pcompg = False

        # Cards to strip entirely (optimization-specific)
        optim_cards = ("DSIZE", "DRESP1", "DCONSTR", "DESGLB")

        i = 0
        while i < len(lines):
            line = lines[i]
            raw = line.rstrip("\r\n")

            # --- Skip continuation lines ---
            if skip_until_blank:
                if raw.startswith("+") or raw.startswith("        "):
                    i += 1
                    continue
                else:
                    skip_until_blank = False

            # --- Remove skin PCOMPG/PCOMP block, preserve stiffener PCOMPGs ---
            if raw.startswith("PCOMPG") or raw.startswith("PCOMP"):
                _card_pid = None
                try:
                    _card_pid = int(raw.split()[1])
                except (ValueError, IndexError):
                    pass
                if (_card_pid is not None and original_pid is not None
                        and _card_pid != original_pid):
                    _preserved_stiffener_pcompgs += 1
                    # Flat-side stiffener shells are already stripped to stiffener
                    # plies only -> copy the whole card verbatim.
                    if flat_stiffener_pids and _card_pid in flat_stiffener_pids:
                        output_lines.append(line)
                        j = i + 1
                        while j < len(lines):
                            r = lines[j].rstrip("\r\n")
                            s = r.strip()
                            if not s or s.startswith("$") or r[0] not in " +":
                                break
                            output_lines.append(lines[j])
                            j += 1
                        i = j
                        continue
                    stiffener_plies, j, _had_skin = _collect_stiffener_ply_lines(
                        lines, i, mat_id)
                    if _had_skin:
                        _card_skin = (
                            stiffener_skin_overrides.get(_card_pid,
                                                         new_stiffener_skin)
                            if stiffener_skin_overrides else new_stiffener_skin
                        )
                    else:
                        # Flat-side attached stiffener shell: the card never
                        # had skin plies — preserve it verbatim.
                        _card_skin = []
                    output_lines.append(line)
                    for _gp, (_smid, _st, _sang) in enumerate(
                            _card_skin, 1):
                        output_lines.append(
                            _format_pcompg_skin_ply(_gp, _smid, _st, _sang))
                    output_lines.extend(stiffener_plies)
                    i = j
                    continue
                in_pcompg = True
                i += 1
                continue
            if in_pcompg:
                stripped = raw.strip()
                if stripped and not stripped.startswith("$") and \
                        (raw[0] == " " or raw[0] == "+"):
                    i += 1
                    continue
                else:
                    in_pcompg = False
                    if not removed_pcompg:
                        output_lines.append("$$\n")
                        output_lines.append("$$  PCOMP Data (SST-discretized zones)\n")
                        output_lines.append("$$\n")
                        output_lines.extend(pcomp_lines)
                        removed_pcompg = True

            # --- Remove optimization cards ---
            if any(raw.startswith(card) for card in optim_cards):
                skip_until_blank = True
                i += 1
                continue

            # --- Remove DESOBJ from subcase ---
            if raw.strip().startswith("DESOBJ"):
                i += 1
                continue

            # --- Remove HM optimization comments ---
            if "$HMNAME DESVARS" in raw or "$HMNAME OPTICONSTRAINTS" in raw:
                i += 1
                continue
            if "OPTIRESPONSES" in raw or "OPTICONSTRAINTS" in raw:
                i += 1
                continue

            # --- Update CQUAD4 property references ---
            if raw.startswith("CQUAD4"):
                new_line = _update_cquad4_pid_direct(raw, eid_to_pid)
                output_lines.append(new_line + "\n")
                i += 1
                continue

            # --- Remove original property HM comments ---
            if "$HMNAME PROP" in raw:
                if original_pid is not None and f'"property{original_pid}"' in raw:
                    i += 1
                    continue
                elif '"property2"' in raw:
                    i += 1
                    continue
            if "$HWCOLOR PROP" in raw and raw.strip().endswith("5"):
                if "2" in raw.split():
                    i += 1
                    continue

            # --- Pass through everything else ---
            output_lines.append(line)
            i += 1

        with open(output_fem_path, "w") as f:
            f.writelines(output_lines)

        if verbose:
            print(f"\n  Written: {output_fem_path}")
            print(f"  Total PCOMP zones: {len(eid_to_pid) and len(set(eid_to_pid.values()))}")
            if _preserved_stiffener_pcompgs:
                print(f"  Preserved stiffener PCOMPG cards: "
                      f"{_preserved_stiffener_pcompgs}")


def _parse_pcomp_plies_detail(fem_path: str) -> list[dict] | None:
    """Parse PCOMP/PCOMPG plies from .fem, returning list of {mid, T, theta}."""
    with open(fem_path, "r") as f:
        lines = f.readlines()

    def _is_cont(l):
        return l and l[0] in (" ", "\t", "+", "*")

    for i, line in enumerate(lines):
        if line.startswith("PCOMPG"):
            plies = []
            j = i + 1
            while j < len(lines) and _is_cont(lines[j]):
                cl = lines[j]
                if len(cl) >= 32 and cl.strip().startswith("+"):
                    mid = cl[16:24].strip()
                    t_str = cl[24:32].strip()
                    theta = cl[32:40].strip()
                    if t_str:
                        try:
                            plies.append({"mid": mid, "T": float(t_str), "theta": theta})
                        except ValueError:
                            pass
                j += 1
            return plies if plies else None

        if line.startswith("PCOMP") and "PCOMPG" not in line:
            plies = []
            j = i + 1
            while j < len(lines) and _is_cont(lines[j]):
                cl = lines[j]
                if len(cl) < 24:
                    j += 1
                    continue
                for off in (8, 40):
                    mid = cl[off:off + 8].strip() if len(cl) > off else ""
                    t_str = cl[off + 8:off + 16].strip() if len(cl) > off + 8 else ""
                    theta = cl[off + 16:off + 24].strip() if len(cl) > off + 16 else ""
                    if t_str:
                        try:
                            plies.append({"mid": mid, "T": float(t_str), "theta": theta})
                        except ValueError:
                            pass
                j += 1
            return plies if plies else None
    return None


def _parse_dsize_angle_to_dv(fem_path: str) -> dict[str, int] | None:
    """Parse DSIZE BALANCE constraints to build angle to DV-index mapping."""
    with open(fem_path, "r") as f:
        lines = f.readlines()

    balanced_pairs = []
    ply_angles_list = []
    dsize_ref_type = None
    dsize_ref_id = None

    for i, line in enumerate(lines):
        if line.startswith("DSIZE"):
            dsize_ref_type = line[16:24].strip()
            dsize_ref_id = line[24:32].strip()
            j = i + 1
            while j < len(lines):
                cl = lines[j]
                if not cl.strip() or cl.strip()[0] not in ("+", "*"):
                    break
                if "BALANCE" in cl:
                    toks = cl.split()
                    try:
                        bi = toks.index("BALANCE")
                        balanced_pairs.append((toks[bi + 1], toks[bi + 2]))
                    except (ValueError, IndexError):
                        pass
                j += 1
            break

    if dsize_ref_type is None:
        return None

    if dsize_ref_type == "STACK":
        ply_angles = {}
        for line in lines:
            if line.startswith("PLY"):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        ply_angles[int(parts[1])] = parts[4].strip()
                    except (ValueError, IndexError):
                        pass
        stack_ids = []
        for i, line in enumerate(lines):
            if line.startswith("STACK"):
                rest = line[16:]
                for tok in rest.split():
                    try:
                        stack_ids.append(int(tok))
                    except ValueError:
                        pass
                j = i + 1
                while j < len(lines):
                    cl = lines[j]
                    if not cl.strip() or cl.strip()[0] not in ("+", "*"):
                        break
                    for tok in cl.strip()[1:].split():
                        try:
                            stack_ids.append(int(tok))
                        except ValueError:
                            pass
                    j += 1
                break
        for pid in stack_ids:
            a = ply_angles.get(pid)
            if a is not None:
                ply_angles_list.append(a)

    elif dsize_ref_type == "PCOMP":
        for i, line in enumerate(lines):
            if line.startswith("PCOMP") and not line.startswith("PCOMPG"):
                if line[8:16].strip() == dsize_ref_id:
                    j = i + 1
                    while j < len(lines) and lines[j] and \
                            lines[j][0] in (" ", "\t", "+", "*"):
                        cl = lines[j]
                        for off in (8, 40):
                            t_str = cl[off + 8:off + 16].strip() \
                                if len(cl) > off + 8 else ""
                            theta = cl[off + 16:off + 24].strip() \
                                if len(cl) > off + 16 else ""
                            if t_str and theta:
                                ply_angles_list.append(theta)
                        j += 1
                    break

    if not ply_angles_list:
        return None

    angle_to_dv = {}
    dv_idx = 0
    seen_pairs = set()
    for angle in ply_angles_list:
        matched = False
        for a1, a2 in balanced_pairs:
            if angle == a1 or angle == a2:
                key_pair = tuple(sorted([a1, a2]))
                if key_pair not in seen_pairs:
                    seen_pairs.add(key_pair)
                    angle_to_dv[a1] = dv_idx
                    angle_to_dv[a2] = dv_idx
                    dv_idx += 1
                matched = True
                break
        if not matched and angle not in angle_to_dv:
            angle_to_dv[angle] = dv_idx
            dv_idx += 1

    return angle_to_dv if angle_to_dv else None


def detect_stiffener_elements(fem_path: str) -> tuple[int, set[int]]:
    """
    Detect stiffener (non-design) elements from the .fem file.

    Parses the DSIZE/DTPL card to find the design property ID, then scans
    CQUAD4 entries.  Elements whose PID != design PID are stiffeners.

    Returns:
        (design_pid, stiffener_eids): design property ID and set of stiffener
        element IDs.  If no DSIZE/DTPL found, returns (0, set()).
    """
    with open(fem_path, "r") as f:
        lines = f.readlines()

    design_pid = None
    for line in lines:
        if line.startswith("DSIZE") or line.startswith("DTPL"):
            ref_type = line[16:24].strip()
            ref_id = line[24:32].strip()
            if ref_type in ("PCOMP", "PSHELL") and ref_id:
                try:
                    design_pid = int(ref_id)
                except ValueError:
                    pass
            break

    if design_pid is None:
        return 0, set()

    stiffener_eids = set()
    for line in lines:
        if line.startswith("CQUAD4"):
            try:
                pid = int(line[16:24].strip())
                if pid != design_pid:
                    eid = int(line[8:16].strip())
                    stiffener_eids.add(eid)
            except (ValueError, IndexError):
                pass

    return design_pid, stiffener_eids


def _design_skin_ply_count(lines: list[str], design_pid: int):
    """Number of plies in the design-skin property. Inside a stiffener PCOMPG the
    skin base is exactly this many plies, listed first (bottom of the stack),
    followed by the stiffener ply(s). That count/position is the only invariant
    that separates skin base from stiffener across all models: it holds even when
    skin and stiffener share a material (numex1symm), when the stiffener ply is
    thinner than a skin ply (a stiffener run-out), and when the skin base under a
    stiffener differs in thickness from the free skin. Returns None if the design
    property isn't found. Handles a PCOMP design skin (up to two plies per
    continuation line) and a PCOMPG design skin (one ply per line)."""
    for i, line in enumerate(lines):
        head = line[:8].strip()
        if head not in ("PCOMP", "PCOMPG"):
            continue
        try:
            if int(line[8:16]) != design_pid:
                continue
        except (ValueError, IndexError):
            continue
        count, j = 0, i + 1
        while j < len(lines):
            r = lines[j].rstrip("\r\n")
            if not r or r[0] not in " +" or r.strip().startswith("$"):
                break
            if head == "PCOMP":
                for off in (8, 40):        # thickness at cols 16/48
                    if r[off + 8:off + 16].strip():
                        count += 1
            else:                          # PCOMPG: one ply per line
                count += 1
            j += 1
        return count
    return None


def convert_stiffeners_to_flat_side(
        source_fem_path: str,
        output_fem_path: str,
        skin_mat_id: int = 1,
        eid_offset: int = 1000000,
        include_stiffeners: bool = True,
        verbose: bool = False,
) -> tuple[int, int]:
    """Convert a smeared-stiffener model to a flat-side-stiffener model. In the
    source, each stiffened CQUAD4 has a PCOMPG with both skin plies and the
    smeared stiffener ply, and being a non-design property the FSO can't size
    the skin under it. The conversion:

    * re-points every stiffened CQUAD4 to the design property (whole skin
      designable);
    * adds a duplicate CQUAD4 (EID + eid_offset) on the same nodes referencing
      the original stiffener PID, rigidly attached via the shared nodes;
    * strips each stiffener PCOMPG to only its stiffener plies, so the attached
      element carries exactly the original stiffener thickness.

    include_stiffeners=False omits the duplicates and stiffener PCOMPGs, giving
    the bare all-designable panel for the FSO input (keeping stiffener stiffness
    out of the FSO, else the optimizer strips the redundant skin under it).
    Everything else passes through verbatim. Returns
    (n_converted_elements, n_stripped_pcompgs)."""
    design_pid, stiffener_eids = detect_stiffener_elements(source_fem_path)
    if design_pid == 0 or not stiffener_eids:
        raise ValueError(
            f"No stiffener elements found in {source_fem_path}; "
            f"flat-side conversion is not applicable.")

    with open(source_fem_path, "r") as f:
        lines = f.readlines()

    # PIDs referenced by stiffened elements (their PCOMPGs get stripped).
    stiffener_pids = set()
    for line in lines:
        if line.startswith("CQUAD4"):
            try:
                if int(line[8:16]) in stiffener_eids:
                    stiffener_pids.add(int(line[16:24]))
            except (ValueError, IndexError):
                pass

    # Inside a stiffener PCOMPG the skin base is the first k plies (k = design-
    # skin ply count), followed by the stiffener ply(s). Strip those first k
    # plies and keep the rest. Count/position is the only robust discriminator:
    # neither material nor thickness works in general (skin and stiffener can
    # share a material, and a stiffener run-out ply can be thinner than a skin
    # ply), but the skin base always replicates the full design layout first.
    _skin_ply_k = _design_skin_ply_count(lines, design_pid)
    n_empty_guard = 0

    out = []
    n_conv = 0
    n_stripped = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        raw = line.rstrip("\r\n")

        if raw.startswith("CQUAD4"):
            try:
                eid = int(raw[8:16])
                pid = int(raw[16:24])
            except (ValueError, IndexError):
                out.append(line)
                i += 1
                continue
            if eid in stiffener_eids and pid != design_pid:
                # Skin element joins the design property...
                out.append(raw[:16] + f"{design_pid:>8d}" + raw[24:] + "\n")
                # ...and the stiffener becomes an attached coincident shell
                # (same nodes, same theta) on the original stiffener PID —
                # unless this is the bare FSO variant.
                if include_stiffeners:
                    out.append("CQUAD4  " + f"{eid + eid_offset:>8d}"
                               + f"{pid:>8d}" + raw[24:] + "\n")
                n_conv += 1
            else:
                out.append(line)
            i += 1
            continue

        if raw.startswith("PCOMPG"):
            try:
                pid = int(raw.split()[1])
            except (ValueError, IndexError):
                pid = None
            if pid in stiffener_pids:
                # Keep the header and ONLY the stiffener plies (drop the first k
                # skin-base plies); in the bare FSO variant drop the card body.
                if include_stiffeners:
                    out.append(line)
                j = i + 1
                ply_lines = []
                while j < n:
                    r = lines[j].rstrip("\r\n")
                    s = r.strip()
                    if not s or s.startswith("$") or r[0] not in " +":
                        break
                    ply_lines.append(lines[j])
                    j += 1
                if include_stiffeners:
                    if _skin_ply_k is not None and len(ply_lines) > _skin_ply_k:
                        keep_lines = ply_lines[_skin_ply_k:]   # strip skin base
                    else:
                        # Too few plies to strip safely: keep all so the card is
                        # never left empty (a zero-thickness PCOMPG is fatal).
                        keep_lines = ply_lines
                        n_empty_guard += 1
                    out.extend(keep_lines)
                n_stripped += 1
                i = j
                continue
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1

    with open(output_fem_path, "w") as f:
        f.writelines(out)

    if verbose:
        if include_stiffeners:
            print(f"  Flat-side conversion: {n_conv} stiffened elements "
                  f"re-pointed to design PID {design_pid}; {n_conv} attached "
                  f"stiffener shells created (EID + {eid_offset}); "
                  f"{n_stripped} PCOMPGs stripped to stiffener plies only "
                  f"(k={_skin_ply_k} skin base plies each).")
            if n_empty_guard:
                print(f"  WARNING: {n_empty_guard} stiffener PCOMPG(s) had "
                      f"<= k plies; kept intact to avoid a zero-thickness card. "
                      f"Check the design-skin ply count ({_skin_ply_k}).")
        else:
            print(f"  Flat-side FSO variant: {n_conv} stiffened elements "
                  f"re-pointed to design PID {design_pid}; stiffeners "
                  f"omitted entirely (bare all-designable panel).")
        print(f"  Written: {output_fem_path}")

    return n_conv, n_stripped


def _parse_sh_raw(sh_path: str) -> dict[int, list[float]]:
    """Parse .sh file, returning raw multipliers (0..1) per element."""
    with open(sh_path, "r") as f:
        lines = f.readlines()
    header = lines[1].split()
    n_elements = int(header[0])
    result = {}
    i = 2
    for _ in range(n_elements):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        eid = int(parts[0])
        nv = int(parts[1])
        vals = [float(lines[i + 1 + j].strip()) for j in range(nv)]
        result[eid] = vals
        i += nv + 1
    return result


def export_fem_fso_exact(
        source_fem_path: str,
        sh_file_path: str,
        output_fem_path: str,
        guide_stack: list[float] | None = None,
        preserve_pids: set[int] | None = None,
        sst: StackingSequenceTable | None = None,
        stiffener_base_nplies: int | None = None,
        flat_stiffener_pids: set[int] | None = None,
        verbose: bool = False,
) -> int:
    """Export a .fem with per-element PCOMP properties straight from FSO .sh
    data. The stack uses the SST guide angles; each element's per-angle total
    (.sh multiplier x nominal_T) is split equally across that angle's guide
    plies. Pure CLPT, structural analysis only (no SMEAR/SYM, optimization cards
    removed). preserve_pids keeps stiffener PCOMPGs (skin replaced by the FSO-
    exact base). flat_stiffener_pids marks flat-side attached shells, already
    stripped to stiffener plies only, which are copied verbatim (the material-
    based skin test misfires when stiffener and skin share a material). Returns
    the number of unique PCOMP properties created."""
    nominal_plies = _parse_pcomp_plies_detail(source_fem_path)
    if nominal_plies is None:
        raise ValueError("No PCOMP/PCOMPG found in source .fem")

    angle_to_dv = _parse_dsize_angle_to_dv(source_fem_path)
    sh_mults = _parse_sh_raw(sh_file_path)

    if guide_stack is None:
        guide_stack = [float(p["theta"]) for p in nominal_plies]

    mat_id = nominal_plies[0]["mid"]
    mat_id_int = int(mat_id)
    nominal_t = nominal_plies[0]["T"]

    # FSO_exact is the unconstrained optimum (no min ply thickness), so its
    # stiffener base uses the ideal sub-ply thickness base_t/DF, keeping it at
    # the thickest-zone scale (40 x 0.05 = 2.0 mm) rather than the min-ply-
    # clamped 40 x 0.125 = 5.0 mm.
    _t_ideal = (sst.base_ply_thickness / sst.doubling_factor
                if sst is not None and sst.doubling_factor else None)
    new_stiffener_skin = _build_stiffener_skin_plies(
        sst, mat_id_int, None, stiffener_base_nplies,
        ply_thickness_override=_t_ideal) \
        if sst is not None and stiffener_base_nplies else []

    # Flat-side stiffener shells keep their own cards, so this base skin is
    # computed but never applied — don't report it.
    if verbose and new_stiffener_skin and not flat_stiffener_pids:
        _total_t = sum(p[1] for p in new_stiffener_skin)
        _angles = [p[2] for p in new_stiffener_skin]
        print(f"  Stiffener base skin: {len(new_stiffener_skin)} plies, "
              f"t = {_total_t:.3f} mm, angles={_angles}")

    guide_angles_str = [f"{a:.1f}" for a in guide_stack]

    angle_counts = {}
    for a_str in guide_angles_str:
        angle_counts[a_str] = angle_counts.get(a_str, 0) + 1

    if verbose:
        print(f"  Guide stack: {guide_stack}")
        print(f"  Guide angles (str): {guide_angles_str}")
        print(f"  Angle counts: {angle_counts}")
        print(f"  Nominal T: {nominal_t:.4f}")
        print(f"  Angle-to-DV: {angle_to_dv}")
        print(f"  Elements in .sh: {len(sh_mults)}")

    def _angle_total_mult(eid_mults, a_str):
        """Return the .sh multiplier for a given angle string."""
        if angle_to_dv is None or not angle_to_dv:
            idx = guide_angles_str.index(a_str)
            return eid_mults[idx] if idx < len(eid_mults) else 1.0
        dv = angle_to_dv.get(a_str)
        return eid_mults[dv] if dv is not None and dv < len(eid_mults) else 1.0

    elem_key = {}
    for eid, raw_m in sh_mults.items():
        per_angle_total = {}
        for a_str in angle_counts:
            mult = _angle_total_mult(raw_m, a_str)
            per_angle_total[a_str] = mult * nominal_t

        ply_ts = []
        for a_str in guide_angles_str:
            t = per_angle_total[a_str] / angle_counts[a_str]
            ply_ts.append(round(t, 4))
        elem_key[eid] = tuple(ply_ts)

    unique_keys = sorted(set(elem_key.values()))
    pid_base = 100
    key_to_pid = {key: pid_base + i for i, key in enumerate(unique_keys)}

    pcomp_lines = []
    for key in unique_keys:
        pid = key_to_pid[key]
        pcomp_lines.append(f"$$ FSO Exact PID {pid}\n")
        pcomp_lines.append(_format_pcomp_header(pid, sym=False, smear=False))
        for j in range(0, len(guide_angles_str), 2):
            a1 = guide_angles_str[j]
            t1 = key[j]
            if j + 1 < len(guide_angles_str):
                a2 = guide_angles_str[j + 1]
                t2 = key[j + 1]
                line = (f"+       {mat_id:>8}{t1:<8.4f}"
                        f"{a1:>8}{'YES':>8}"
                        f"{mat_id:>8}{t2:<8.4f}"
                        f"{a2:>8}{'YES':>8}\n")
            else:
                line = (f"+       {mat_id:>8}{t1:<8.4f}"
                        f"{a1:>8}{'YES':>8}\n")
            pcomp_lines.append(line)
        pcomp_lines.append("\n")

    eid_to_pid = {eid: key_to_pid[key] for eid, key in elem_key.items()}

    with open(source_fem_path, "r") as f:
        lines = f.readlines()

    original_pid = None
    for line in lines:
        s = line.strip()
        if s.startswith("PCOMPG") or s.startswith("PCOMP"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    original_pid = int(parts[1])
                except ValueError:
                    pass
            break

    output_lines = []
    skip_until_blank = False
    in_pcompg = False
    removed_pcompg = False
    optim_cards = ("DSIZE", "DRESP1", "DCONSTR", "DESGLB")

    i = 0
    while i < len(lines):
        line = lines[i]
        raw = line.rstrip("\r\n")

        if skip_until_blank:
            if raw.startswith("+") or raw.startswith("        "):
                i += 1
                continue
            else:
                skip_until_blank = False

        if raw.startswith("PCOMPG"):
            if preserve_pids or flat_stiffener_pids:
                try:
                    _pid = int(raw.split()[1])
                except (ValueError, IndexError):
                    _pid = None
                # Flat-side attached shell: already stripped -> copy verbatim.
                if flat_stiffener_pids and _pid in flat_stiffener_pids:
                    output_lines.append(line)
                    _next = i + 1
                    while _next < len(lines):
                        r = lines[_next].rstrip("\r\n")
                        s = r.strip()
                        if not s or s.startswith("$") or r[0] not in " +":
                            break
                        output_lines.append(lines[_next])
                        _next += 1
                    i = _next
                    continue
                if preserve_pids and _pid is not None and _pid in preserve_pids:
                    stiffener_plies, _next, _had_skin = \
                        _collect_stiffener_ply_lines(lines, i, mat_id_int)
                    output_lines.append(line)
                    # Cards without skin plies (flat-side attached stiffener
                    # shells) are preserved verbatim — no skin injection.
                    _card_skin = new_stiffener_skin if _had_skin else []
                    for _gp, (_smid, _st, _sang) in enumerate(
                            _card_skin, 1):
                        output_lines.append(
                            _format_pcompg_skin_ply(_gp, _smid, _st, _sang))
                    output_lines.extend(stiffener_plies)
                    i = _next
                    continue
            in_pcompg = True
            i += 1
            continue
        if raw.startswith("PCOMP"):
            in_pcompg = True
            i += 1
            continue
        if in_pcompg:
            stripped = raw.strip()
            if stripped and not stripped.startswith("$") and \
                    (raw[0] == " " or raw[0] == "+"):
                i += 1
                continue
            else:
                in_pcompg = False
                if not removed_pcompg:
                    output_lines.append("$$\n")
                    output_lines.append("$$  PCOMP Data (FSO exact, per-element)\n")
                    output_lines.append("$$\n")
                    output_lines.extend(pcomp_lines)
                    removed_pcompg = True

        if any(raw.startswith(card) for card in optim_cards):
            skip_until_blank = True
            i += 1
            continue

        if raw.strip().startswith("DESOBJ"):
            i += 1
            continue

        if "$HMNAME DESVARS" in raw or "$HMNAME OPTICONSTRAINTS" in raw:
            i += 1
            continue
        if "OPTIRESPONSES" in raw or "OPTICONSTRAINTS" in raw:
            i += 1
            continue

        if raw.startswith("CQUAD4"):
            new_line = _update_cquad4_pid_direct(raw, eid_to_pid)
            output_lines.append(new_line + "\n")
            i += 1
            continue

        if "$HMNAME PROP" in raw:
            if original_pid is not None and f'"property{original_pid}"' in raw:
                i += 1
                continue
            elif '"property2"' in raw:
                i += 1
                continue
        if "$HWCOLOR PROP" in raw and raw.strip().endswith("5"):
            if "2" in raw.split():
                i += 1
                continue

        output_lines.append(line)
        i += 1

    with open(output_fem_path, "w") as f:
        f.writelines(output_lines)

    if verbose:
        print(f"\n  Written: {output_fem_path}")
        print(f"  Unique PCOMP properties: {len(unique_keys)}")

    return len(unique_keys)

def _format_pcomp_header(pid: int, sym: bool = False, smear: bool = False) -> str:
    if smear:
        flag = "SMEAR"
    elif sym:
        flag = "SYM"
    else:
        flag = ""
    return f"PCOMP   {pid:<8d}" + " " * 48 + flag + "\n"


def _format_pcomp_ply(mid: int, thickness: float, angle: float) -> str:
    t_str = f"{thickness:.4f}" if thickness < 10 else f"{thickness:.3f}"
    a_str = f"{angle:.1f}"
    return f"        {mid:<8d}{t_str:<8s}{a_str:<8s}YES\n"

# AS4/3501-6 Carbon/Epoxy unidirectional lamina
CFRP_AS4_3501_6 = {
    "E1": 135.0, "E2": 9.0, "NU12": 0.3,
    "G12": 5.0, "G1Z": 5.0, "G2Z": 5.0,
}


def _fmt8(v: float) -> str:
    """Format a number to fit an 8-column Nastran small field, always keeping a
    decimal point so it is read as a real (Nastran rejects bare integers here)."""
    s = f"{v:.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    if len(s) > 8:
        s = f"{v:.6g}"
        if "." not in s and "E" not in s and "e" not in s:
            s = (s + ".")[:8]
    return f"{s:<8.8s}"


def _format_mat8(mid: int, p: dict) -> str:
    return ("MAT8    " + f"{mid:<8d}"
            + _fmt8(p["E1"]) + _fmt8(p["E2"]) + _fmt8(p["NU12"])
            + _fmt8(p["G12"]) + _fmt8(p["G1Z"]) + _fmt8(p["G2Z"]) + "\n")


def _format_pcomp_smear_plies(mid: int, plies) -> list[str]:
    """Format plies as PCOMP '+' continuation lines, two plies per line, in the
    standard MID/T/THETA/SOUT column layout (thickness at cols 16 & 48, angle at
    24 & 56) so both OptiStruct and parse_initial_plies read them."""
    out = []
    for k in range(0, len(plies), 2):
        row = "+       "
        for t, a in plies[k:k + 2]:
            row += f"{mid:>8d}" + f"{t:<8.4f}"[:8] + f"{a:<8.1f}"[:8] + "YES     "
        out.append(row + "\n")
    return out


def _clean_float(s: str) -> float:
    """Parse a Nastran short-form float (e.g. '1.0-5' -> 1.0e-5)."""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(re.sub(r"(?<=\d)(?=[+-])", "E", s))
    except ValueError:
        return 0.0


def _parse_pcomp_block(lines: list[str], hdr_idx: int):
    """From a PCOMP header at hdr_idx, consume its '+' continuation lines.
    Returns (end_idx, total_thickness, first_ply_material_mid)."""
    total, mid, j = 0.0, None, hdr_idx + 1
    while j < len(lines) and lines[j].startswith("+"):
        for off in (8, 40):  # MID at cols 8/40, thickness at 16/48
            ms, ts = lines[j][off:off + 8].strip(), lines[j][off + 8:off + 16].strip()
            if ms and ts:
                try:
                    m, t = int(float(ms)), _clean_float(lines[j][off + 8:off + 16])
                except ValueError:
                    continue
                if t > 0:
                    total += t
                    if mid is None:
                        mid = m
        j += 1
    return j, total, mid


def _material_is_isotropic(lines: list[str], mid) -> bool:
    """True if material *mid* is isotropic: a MAT1, or a MAT8 with E1 == E2."""
    if mid is None:
        return False
    for line in lines:
        head = line[:8].strip()
        if head not in ("MAT1", "MAT2", "MAT8", "MAT9"):
            continue
        try:
            if int(line[8:16]) != mid:
                continue
        except (ValueError, IndexError):
            continue
        if head == "MAT1":
            return True
        if head == "MAT8":
            e1, e2 = _clean_float(line[16:24]), _clean_float(line[24:32])
            return abs(e1 - e2) <= 1e-6 * max(abs(e1), abs(e2), 1.0)
        return False
    return False


def convert_noncomposite_skin_to_composite(
        source_fem_path: str,
        output_fem_path: str,
        angles=(0.0, 45.0, -45.0, 90.0),
        material: dict | None = None,
        verbose: bool = False) -> bool:
    """Rebuild a non-composite design skin as a PCOMP LAM=SMEAR of equal-thickness
    plies at *angles* (summing to the original skin thickness) on a new carbon-
    fibre MAT8, so the composite DfM pipeline can run. Applies to:
      * a PSHELL design skin, and
      * a PCOMP/PCOMPG skin whose plies use an isotropic material (MAT1, or a
        MAT8 with E1 == E2) — this also normalises a malformed LAM flag.
    A skin that is already a genuine composite (orthotropic MAT8) is left
    unchanged. The DSIZE is not touched (it must reference the skin PID as a
    composite free-size). Returns True only if a conversion was performed."""
    if material is None:
        material = CFRP_AS4_3501_6
    design_pid, _ = detect_stiffener_elements(source_fem_path)
    with open(source_fem_path, "r") as f:
        lines = f.readlines()

    def _copy():
        with open(output_fem_path, "w") as f:
            f.writelines(lines)

    if design_pid == 0:
        if verbose:
            print("  Skin conversion: no DSIZE design property found; skipped.")
        _copy()
        return False

    prop_idx = prop_kind = skin_t = block_end = mat_mid = None
    for i, line in enumerate(lines):
        head = line[:8].strip()
        try:
            pid = int(line[8:16]) if len(line) >= 16 else None
        except ValueError:
            pid = None
        if pid != design_pid:
            continue
        if head == "PSHELL":
            prop_idx, prop_kind = i, "PSHELL"
            skin_t, block_end = _clean_float(line[24:32]), i + 1
            break
        if head in ("PCOMP", "PCOMPG"):
            prop_idx, prop_kind = i, head
            block_end, skin_t, mat_mid = _parse_pcomp_block(lines, i)
            break

    if prop_idx is None or not skin_t or skin_t <= 0:
        if verbose:
            print(f"  Skin conversion: design PID {design_pid} property not found "
                  f"or unreadable; skipped.")
        _copy()
        return False

    if prop_kind != "PSHELL" and not _material_is_isotropic(lines, mat_mid):
        if verbose:
            print(f"  Skin conversion: design PID {design_pid} is already a "
                  f"composite {prop_kind} (orthotropic material); no change.")
        _copy()
        return False

    used_mids = set()
    for line in lines:
        if line[:8].strip() in ("MAT1", "MAT2", "MAT8", "MAT9"):
            try:
                used_mids.add(int(line[8:16]))
            except (ValueError, IndexError):
                pass
    new_mid = (max(used_mids) + 1) if used_mids else 1

    t_ply = skin_t / len(angles)
    block = [_format_pcomp_header(design_pid, smear=True)]
    block += _format_pcomp_smear_plies(new_mid, [(t_ply, a) for a in angles])
    block.append(_format_mat8(new_mid, material))

    out = lines[:prop_idx] + block + lines[block_end:]
    with open(output_fem_path, "w") as f:
        f.writelines(out)

    if verbose:
        _src = ("PSHELL" if prop_kind == "PSHELL"
                else f"isotropic {prop_kind}")
        print(f"  Skin conversion: {_src} (PID {design_pid}, t={skin_t:.4f} mm) "
              f"-> PCOMP SMEAR, {len(angles)} plies "
              f"{[f'{a:g}' for a in angles]} @ {t_ply:.4f} mm each, on new "
              f"carbon-fibre MAT8 (MID {new_mid}).")
    return True


def sanitize_pcomp_lam_flags(fem_path: str) -> bool:
    """Blank malformed non-numeric tokens in the GE field of PCOMP/PCOMPG header
    cards. Some HyperMesh exports emit a garbled 'SYSMEAR' (SYM+SMEAR mashed
    together) into the GE column, which OptiStruct rejects with ERROR #1485
    ('unrecognized/unexpected label'). GE must be a real number or blank; the
    real laminate flag lives in the LAM field (next column) and is left intact,
    so only the offending token is cleared. Returns True if the file changed."""
    def _is_numeric_field(s: str) -> bool:
        s = s.strip()
        if not s:
            return True
        try:
            float(re.sub(r'(?<=\d)(?=[+-])', 'E', s))
            return True
        except ValueError:
            return False

    with open(fem_path, "r") as f:
        lines = f.readlines()

    changed = False
    for i, line in enumerate(lines):
        if line[:8].strip() not in ("PCOMP", "PCOMPG"):
            continue
        raw = line.rstrip("\r\n").ljust(72)
        if _is_numeric_field(raw[56:64]):
            continue
        lines[i] = (raw[:56] + " " * 8 + raw[64:]).rstrip() + "\n"
        changed = True

    if changed:
        with open(fem_path, "w") as f:
            f.writelines(lines)
    return changed


def _update_cquad4_pid_direct(line: str, eid_to_pid: dict) -> str:
    raw = line.rstrip("\r\n")
    raw = raw.ljust(56)

    eid = int(raw[8:16].strip())

    if eid in eid_to_pid:
        new_pid = eid_to_pid[eid]
    else:
        return line.rstrip("\r\n")

    return raw[:8] + f"{eid:>8d}" + f"{new_pid:>8d}" + raw[24:]


def run_solver(work_dir=".", exec_path=OPTISTRUCT_EXE, filename=".",
               optimize=False):
    """Calls OptiStruct. Returns True if solver ran, False if skipped (executable not found)."""
    if not os.path.exists(exec_path):
        print(f"  [SKIP] OptiStruct executable not found: {exec_path}")
        print(f"         Solver run skipped for '{filename}'. Results will use cached data or default to 0.")
        return False

    input_path = os.path.abspath(os.path.join(work_dir, filename))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Can't find file at {input_path}")

    if optimize:
        cmd = [exec_path, input_path, "-nt", "18"]
    else:
        cmd = [exec_path, input_path, "-optskip", "-nt", "18"]

    result = subprocess.run(
        cmd,
        cwd=os.path.abspath(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        # OptiStruct on Windows sometimes crashes writing to the console
        # (forrtl severe(38), CONOUT$ "pipe is being closed") at the very end
        # of a job, AFTER a complete valid .out is already on disk, returning a
        # non-zero exit code even though the solve succeeded. The .out was
        # archived before this run, so a parseable compliance means this solve
        # produced it. Only abort when no usable result exists.
        out_path = os.path.splitext(input_path)[0] + ".out"
        have_result = False
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8", errors="replace") as _f:
                _lines = _f.readlines()
            for _i, _ln in enumerate(_lines):
                if "Subcase" in _ln and "Compliance" in _ln:
                    for _sl in _lines[_i + 1:_i + 10]:
                        _p = _sl.split()
                        if len(_p) >= 2 and _p[0].isdigit():
                            try:
                                have_result = float(_p[1]) != 0.0
                            except ValueError:
                                pass
                            break
                if have_result:
                    break
        if have_result:
            print(f"  WARNING: OptiStruct exited with code {result.returncode} "
                  f"for '{filename}' but wrote a complete result; continuing "
                  f"(likely a Windows console-write crash, not a solve failure).")
            return True
        print(f"OptiStruct Error Output:\n{result.stdout}")
        raise RuntimeError("OptiStruct finished with an error code.")

    return True


def read_results(work_dir=".", jobname=".", optimize=False):
    """Parses .out for Compliance and .pch for Strain Energies."""
    out_path = os.path.join(work_dir, f"{jobname}.out")
    pch_path = os.path.join(work_dir, f"{jobname}.pch")

    compliance = 0.0
    optimal_compliance = 0.0
    disp_z = 0.0

    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(reversed(lines)):
                if "ITERATION" in line and "1" in line:
                    optimize = True

            for i, line in enumerate(lines):
                if "Subcase" in line and "Compliance" in line:
                    for sub_line in lines[i + 1: i + 10]:
                        parts = sub_line.split()
                        if len(parts) >= 2 and parts[0].isdigit():
                            try:
                                compliance = float(parts[1])
                                break
                            except:
                                continue
                        if compliance != 0.0: break
            if optimize:
                lines = lines[::-1]
                for i, line in enumerate(lines):
                    if "Note:" and "Epsilon" in line:
                        for sub_line in lines[i + 1: i + 5]:
                            parts = sub_line.split()
                            if len(parts) >= 2 and parts[0].isdigit():
                                try:
                                    optimal_compliance = float(parts[1])
                                    break
                                except:
                                    continue
                            if optimal_compliance != 0.0: break
        if os.path.exists(pch_path):
            with open(pch_path, 'r') as f:
                capture = False
                for line in f:
                    if "$DISPLACEMENTS" in line:
                        capture = True
                        continue

                    if capture and line.startswith("$") and "SUBCASE" not in line and "REAL" not in line:
                        capture = False
                        continue

                    if capture and not line.startswith("$") and "-CONT-" not in line:
                        try:
                            data_str = line[10:].strip()
                            components = data_str.split()
                            disp_z = float(components[3])
                            break
                        except Exception:
                            continue
    if optimize:
        return compliance, optimal_compliance, disp_z
    else:
        return compliance, disp_z


def _render_fem_with_volume_fraction(fem_path: str, new_vf: float) -> str:
    """Return the .fem contents with DCONSTR volume fraction updated in-memory"""
    with open(fem_path, "r") as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DCONSTR"):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    val = float(parts[-1])
                except (ValueError, IndexError):
                    continue
                if 0.0 < val < 1.0:
                    old_str = parts[-1]
                    new_str = f"{new_vf:.4f}"
                    raw = line.rstrip("\n")
                    idx = raw.rfind(old_str)
                    if idx >= 0:
                        lines[i] = raw[:idx] + new_str + raw[idx + len(old_str):] + "\n"
                        modified = True
                        break

    if not modified:
        raise ValueError(f"No DCONSTR volume fraction found in {fem_path}")

    return "".join(lines)


def _read_text_if_exists(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()


def _can_reuse_fso_job(jobname: str, expected_fem_text: str | None = None) -> bool:
    """Return True when an existing FSO job can be reused safely."""
    _sh = jobname + ".sh"
    _out = jobname + ".out"
    if not (os.path.exists(_sh) and os.path.exists(_out)):
        return False
    if expected_fem_text is None:
        return True
    _fem = jobname + ".fem"
    _existing = _read_text_if_exists(_fem)
    return _existing == expected_fem_text


def _archive_results(jobname):
    """Archive all existing solver outputs, then remove originals."""
    import shutil
    extensions = [".out", ".sh", ".pch", ".stat", ".res", ".hgdata", ".hmascii", ".report"]
    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    counter = 0
    for ext in extensions:
        src = f"{jobname}{ext}"
        if os.path.exists(src):
            i = 1
            dst = os.path.join(archive_dir, f"{jobname}{ext}")
            while os.path.exists(f"{dst}.{i:03d}"):
                i += 1
            shutil.move(src, f"{dst}.{i:03d}")
            counter += 1
    if counter:
        print(f"  Archived {counter} old files for '{jobname}'")

def read_totalthickness_from_out(work_dir=".", jobname="."):
    out_path = os.path.join(work_dir, f"{jobname}.out")
    t_tot = 0.0

    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if "Total" and "Volume" in line:
                    parts = line.split()
                    if len(parts) > 6:
                        try:
                            t_tot = float(parts[6])
                            break
                        except:
                            continue
                    else:
                        if "Volume" in line:
                            parts = line.split()
                            try:
                                t_tot = float(parts[2])
                                break
                            except:
                                continue
    return t_tot

def read_volfrac_from_out(work_dir=".", jobname="."):
    out_path = os.path.join(work_dir, f"{jobname}.out")
    vol_frac = 1.0

    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if "Initial" and "Material" and "Fraction" in line:
                    parts = line.split()
                    try:
                        vol_frac = float(parts[5])
                        break
                    except:
                        continue
    return vol_frac


def get_volume_fraction(fem_path: str) -> float | None:
    """Read the volume fraction from a .fem file's DCONSTR card.

    Returns the fractional value (0 < val < 1) found on the DCONSTR line,
    or None if not found.
    """
    with open(fem_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("DCONSTR"):
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        val = float(parts[-1])
                        if 0.0 < val < 1.0:
                            return val
                    except (ValueError, IndexError):
                        continue
    return None


def ensure_desglb_global(fem_path: str) -> bool:
    """Move any DESGLB card sitting inside the SUBCASE section (between the first
    SUBCASE and BEGIN BULK) up to the global case-control level, just above the
    first SUBCASE. OptiStruct rejects DESGLB inside a SUBCASE (fatal ERROR 1039);
    HyperMesh sometimes exports it there. Returns True if the file was changed."""
    with open(fem_path, "r") as f:
        lines = f.readlines()

    first_subcase = begin_bulk = None
    for i, line in enumerate(lines):
        s = line.strip().upper()
        if first_subcase is None and s.startswith("SUBCASE"):
            first_subcase = i
        if s.startswith("BEGIN") and "BULK" in s:
            begin_bulk = i
            break
    if first_subcase is None or begin_bulk is None:
        return False

    moved = [i for i in range(first_subcase + 1, begin_bulk)
             if lines[i].strip().upper().startswith("DESGLB")]
    if not moved:
        return False

    desglb = [lines[i] for i in moved]
    kept = [l for j, l in enumerate(lines) if j not in moved]
    out = kept[:first_subcase] + desglb + kept[first_subcase:]
    with open(fem_path, "w") as f:
        f.writelines(out)
    return True


def read_optistruct_error(out_path: str) -> str:
    """Return the first OptiStruct '*** ERROR ...' block from a .out (one-line
    summary), or a generic message if none is found."""
    if not os.path.exists(out_path):
        return f"No .out file at {out_path}."
    lines = open(out_path, encoding="utf-8", errors="replace").read().splitlines()
    for i, l in enumerate(lines):
        if "ERROR" in l and "***" in l:
            detail = next((lines[j].strip() for j in range(i + 1, min(i + 4, len(lines)))
                           if lines[j].strip()), "")
            return f"{l.strip()} {detail}".strip()
    return "See the .out file for details."


def get_min_laminate_thickness(fem_path: str) -> float | None:
    """Read the DSIZE 'COMP LAMTHK <t>' minimum-laminate-thickness, or None."""
    with open(fem_path, "r") as f:
        for line in f:
            k = line.find("LAMTHK")
            if k >= 0:
                return _clean_float(line[k + 8:k + 16])
    return None


def set_min_laminate_thickness(fem_path: str, new_t: float) -> bool:
    """Update the DSIZE 'COMP LAMTHK <t>' value. The thickness sits in the
    8-column field immediately after the LAMTHK keyword field."""
    with open(fem_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        k = line.find("LAMTHK")
        if k < 0:
            continue
        vs = k + 8
        raw = line.rstrip("\n").ljust(vs + 8)
        lines[i] = raw[:vs] + _fmt8(new_t) + raw[vs + 8:].rstrip() + "\n"
        with open(fem_path, "w") as f:
            f.writelines(lines)
        return True
    return False


def _apply_lamthk_to_text(text: str, new_t: float) -> str:
    """Set the DSIZE 'COMP LAMTHK <t>' value in an in-memory .fem text."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        k = line.find("LAMTHK")
        if k < 0:
            continue
        vs = k + 8
        raw = line.rstrip("\n").ljust(vs + 8)
        lines[i] = raw[:vs] + _fmt8(new_t) + raw[vs + 8:].rstrip() + "\n"
        break
    return "".join(lines)


def set_volume_fraction(fem_path: str, new_vf: float):
    """Update the volume fraction upper bound in a .fem file's DCONSTR card.

    Finds the DCONSTR line that contains a fractional value (0 < val < 1) and
    replaces it with *new_vf*.  Prints the change for confirmation.
    """
    with open(fem_path, 'r') as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DCONSTR"):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    val = float(parts[-1])
                    if 0.0 < val < 1.0:
                        old_str = parts[-1]
                        new_str = f"{new_vf:.4f}"
                        raw = line.rstrip("\n")
                        idx = raw.rfind(old_str)
                        if idx >= 0:
                            lines[i] = raw[:idx] + new_str + raw[idx + len(old_str):] + "\n"
                        modified = True
                        print(f"  DCONSTR VF: {val:.4f} -> {new_vf:.4f}")
                        break
                except (ValueError, IndexError):
                    continue

    if modified:
        with open(fem_path, 'w') as f:
            f.writelines(lines)
    else:
        print(f"  WARNING: No DCONSTR VF found in {fem_path}")

    return modified


def _archive_results(jobname):
    """Archive all existing solver outputs for a jobname to archive/ folder.

    Robust against transient Windows file locks (WinError 32): another
    process (a lingering solver, an open viewer, antivirus) briefly holding
    an output file must not kill a multi-hour pipeline.  Locked files are
    retried, then copied instead of moved, and as a last resort skipped
    with a warning.
    """
    extensions = [
        ".out", ".sh", ".pch", ".stat", ".res", ".hgdata", ".hmascii",
        ".report", ".h3d",
    ]
    suffixes = ["_des.h3d", "_s1.h3d"]
    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    counter = 0
    for ext in extensions + suffixes:
        src = f"{jobname}{ext}"
        if not os.path.exists(src):
            continue
        i = 1
        dst = os.path.join(archive_dir, f"{jobname}{ext}")
        while os.path.exists(f"{dst}.{i:03d}"):
            i += 1
        dst = f"{dst}.{i:03d}"
        moved = False
        for attempt in range(4):
            try:
                shutil.move(src, dst)
                moved = True
                break
            except (PermissionError, OSError):
                time.sleep(1.0 + attempt)
        if not moved:
            # Copy so the archive snapshot exists; the solver will overwrite
            # the (still locked) original when it re-runs the job.
            try:
                shutil.copy2(src, dst)
                print(f"  WARNING: '{src}' is locked by another process - "
                      f"archived a copy instead of moving it.")
                moved = True
            except (PermissionError, OSError) as exc:
                print(f"  WARNING: could not archive '{src}' ({exc}) - "
                      f"skipping it.")
        if moved:
            counter += 1
    if counter:
        print(f"  Archived {counter} old files for '{jobname}'")


def _mcl_tag(value: float) -> str:
    """Format MCL value as a filename-safe tag (e.g. 10.0 -> 10p00)."""
    return f"{value:.2f}".replace(".", "p")


def _mcl_values(start: float, stop: float, step: float) -> list[float]:
    """Inclusive floating range for MCL sweep."""
    if step <= 0:
        raise ValueError("Sweep step must be > 0")
    if stop < start:
        raise ValueError("Sweep max must be >= sweep min")
    n = int(round((stop - start) / step))
    values = [round(start + i * step, 6) for i in range(n + 1)]
    if not values or values[-1] < stop - 1e-9:
        values.append(round(stop, 6))
    return values


def _cleanup_junk():
    """Delete OptiStruct junk, temp files, and old experiment artifacts."""
    workdir = os.path.dirname(os.path.abspath(__file__))

    junk_extensions = {".mvw", ".oss", ".hgdata", ".hist", ".hm"}
    junk_files = {
        "hwsolver.mesg", "optistruct.msg", "os_ncpus.txt", "desktop.ini",
    }
    old_filenames = {
        "FSOraw_plyswapped.fem", "FSOraw_plyswapped.sh",
        "FSOraw_plyswapped.out", "FSOraw_plyswapped.pch",
        "FSOraw_plyswapped.h3d", "FSOraw_plyswapped_des.h3d",
        "FSOraw_plyswapped_s1.h3d", "FSOraw_plyswapped.stat",
        "FSOraw_plyswapped.hgdata", "FSOraw_plyswapped.hist",
        "FSOraw_plyswapped.hm", "FSOraw_plyswapped.oss",
        "FSOraw_plyswapped.mvw", "FSOraw_plyswapped.HM.ent.tcl",
        "SSTfiltered.fem", "SSTfiltered.h3d", "SSTfiltered.mvw",
        "SSTfiltered.out", "SSTfiltered.pch", "SSTfiltered.stat",
        "FSO1Dstiff.fem", "FSO1Dstiff.sh",
        "FSOforMMC.fem", "FSOforMMC.sh",
        "SST_reanalysis.fem",
        "clustering.zip", "code.zip",
        "command1.tcl",
        "elbow_formatted.png", "sst_zones.png",
    }
    numbered_pattern = re.compile(r"_\d{3}\.(out|stat)$")
    err_pattern = re.compile(r"\.err_\d+\.out$")
    pycache_dir = os.path.join(workdir, "__pycache__")

    deleted = 0
    skipped = 0

    

    for f in os.listdir(workdir):
        fp = os.path.join(workdir, f)
        if os.path.isdir(fp):
            continue
        _, ext = os.path.splitext(f)
        should_delete = False
        if ext.lower() in junk_extensions:
            should_delete = True
        elif f in junk_files or f in old_filenames:
            should_delete = True
        elif f.endswith(".HM.ent.tcl"):
            should_delete = True
        elif f.startswith("OS") and f.endswith("_inc.txt"):
            should_delete = True
        elif f.startswith("stdout_"):
            should_delete = True
        elif numbered_pattern.search(f):
            should_delete = True
        elif err_pattern.search(f):
            should_delete = True
        if should_delete:
            try:
                os.remove(fp)
                deleted += 1
            except PermissionError:
                skipped += 1

    if os.path.isdir(pycache_dir):
        shutil.rmtree(pycache_dir)
        deleted += 1

    print(f"  Cleaned up {deleted} junk files"
          + (f" ({skipped} locked, skipped)" if skipped else ""))
