# DfM-Skin

Design-for-manufacturing (DfM) code for the variable-thickness composite panel skin, developed
as part of the MSc thesis *Novel Framework for the Optimization and Manufacturable Design of
Stiffened Variable-Thickness Composite Panels* by Flavio Claudio Padua, Faculty of Aerospace
Engineering, Delft University of Technology, in collaboration with Collins Aerospace (academic supervisor: Daniël Peeters; company
supervisor: Ismet Baran).

The code takes a continuous Free-Size Optimization (FSO) thickness result for the panel skin and
turns it into a manufacturable, discrete-zone laminate design for Automated Fiber Placement
(AFP), then quantifies the structural cost of doing so. The pipeline follows the DfM phase of the
thesis methodology (chapter 3, sections 3.5–3.8):

1. **Thickness clustering (`kmeans.py`, section 3.5)** — the continuous FSO thickness field is
   reduced to a scalar total thickness per element, then clustered in one dimension into a finite
   set of constant-thickness zones via Lloyd's k-means algorithm, with an elbow-method sweep to
   select the zone count.
2. **SST discretization (`SST.py`, section 3.6)** — each thickness zone is assigned a
   manufacturable, discrete-ply laminate via a Stacking Sequence Table (SST), built following a
   simplified version of the SST methodology of Irisarri et al. (*Composite Structures* 107,
   2014, 559–569). A guide laminate is expanded and then reduced ply-by-ply along a ply-drop
   ladder, with every intermediate ply count checked against the symmetry, balance, contiguity,
   disorientation, damage-tolerance, covering and internal-continuity guidelines before being
   considered usable.
3. **Morphological filtering (`filter.py`, section 3.7)** — the in-plane zone boundaries produced
   by clustering are filtered so they respect the AFP Minimum Cut Length (MCL), using a
   morphology-based black-and-white filter adapted from Sigmund (*Structural and Multidisciplinary
   Optimization* 33, 2007, 401–424). Zone layouts with more than two levels are handled via
   threshold decomposition into N−1 binary sub-problems, each filtered with the open/close
   morphological operators and reassembled by superposition.
4. **Volume compensation (`main.py`, section 3.8)** — because both discretization and filtering
   change the skin volume relative to the FSO target, the requested volume fraction driving the
   FSO is corrected in closed loop: a perturbed FSO run gives a finite-difference sensitivity of
   final volume to requested volume fraction, from which a corrected volume fraction is
   extrapolated, re-run, and adopted only if it improves the volume match.
5. **Filter verification (`filterverification.py`, section 3.7.2)** — applies the same
   morphological filter, independently of the full skin pipeline, to a simpler topology
   optimization benchmark (element density field, no composites/SST), to check its behaviour
   against literature benchmarks.

`main.py` orchestrates steps 1–4 end-to-end and calls out to Altair OptiStruct for the FSO and
re-analysis runs; `filterverification.py` is a standalone script for step 5.

## Repository contents

| File | Purpose |
|---|---|
| `main.py` | Orchestrates the full FSO → clustering → SST → filtering → volume compensation pipeline. |
| `kmeans.py` | 1D k-means thickness clustering and elbow-method zone-count selection (section 3.5). |
| `SST.py` | Stacking Sequence Table construction from the FSO ply data (section 3.6). |
| `filter.py` | Morphology-based minimum-feature-size filter for AFP manufacturability (section 3.7). |
| `filterverification.py` | Standalone filter verification against a topology-optimization benchmark (section 3.7.2). |
| `helper.py` | Shared I/O and OptiStruct interfacing utilities (`.fem`/`.out`/`.pch` parsing, solver invocation). |
| `bulkhead_simple.fem`, `bulkhead_compress.fem`, `bulkhead_asymm2s.fem`, `MMConly2mm.fem`, `MMConly20mm.fem` | FSO input models used by `main.py` (selected via the `FILE` constant near the top of the script). |
| `FilterVerificationSigmundNC_2.fem`, `FilterVerificationSigmundNC_2.sh` | Benchmark input used by `filterverification.py`. |

## Requirements

```
pip install -r requirements.txt
```

Python 3.9+ with the packages in `requirements.txt` (`numpy`, `scipy`, `matplotlib`,
`scikit-image`) is sufficient to run the clustering, SST construction and filtering steps, and to
export `.fem` files.

**Structural re-analysis and the FSO/volume-compensation loop require a local, licensed
installation of Altair OptiStruct.** The solver path is set via `OPTISTRUCT_EXE` at the top of
`helper.py` (`C:\Program Files\Altair\2025.1\hwsolvers\scripts\optistruct.bat` by default — edit
this to match your installation). If OptiStruct is not found at that path, the pipeline still
runs and exports all intermediate/filtered `.fem` files; it skips the solver re-analysis step and
reports this explicitly rather than failing.

## Usage

Both entry points are plain scripts (no CLI arguments) — configuration is done by editing the
constants near the top of the file, then running the script directly:

```
python main.py                 # full pipeline; set FILE = "bulkhead_simple" (or one of the
                                # other provided models) near the top to choose the input
python filterverification.py   # standalone filter verification against the benchmark case
```

## References

Irisarri, F.-X., Laurin, F., Leroy, F.-H., Maire, J.-F. (2014). Computational strategy for
multiobjective optimization of composite stiffened panels. *Composite Structures*, 107, 559–569.

Sigmund, O. (2007). Morphology-based black and white filters for topology optimization.
*Structural and Multidisciplinary Optimization*, 33, 401–424.
