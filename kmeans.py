import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

class DesignProblem:
    """A simple container to satisfy the plot_clustered_thickness function"""

    def __init__(self, X, Y, skin_thickness):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.skin_thickness = np.array(skin_thickness)


def k_means1D(thickness_data, n_clusters):
    rng = np.random.default_rng(seed=42)
    t_max = np.max(list(thickness_data.values()))
    t_min = np.min(list(thickness_data.values()))
    center_data = rng.choice(np.linspace(t_min, t_max, len(list(thickness_data.values()))), n_clusters, replace=False)

    difference = 1
    while difference > 0.0001:
        cluster_data, wcss = create_1Dclusters(thickness_data, center_data)
        updated_centers = update_cluster_centers(cluster_data, n_clusters)
        center_data_previous = center_data
        center_data = updated_centers
        difference = np.mean(np.abs(np.array(center_data) - np.array(center_data_previous)))

    return cluster_data, center_data, wcss

def create_1Dclusters(thickness_data, cluster_centers):
    cluster_assignment = {}
    wcss = 0
    cluster_data = {i: [] for i in range(len(cluster_centers))}
    for eid, t_val in thickness_data.items():
        distances = []
        for center in cluster_centers:
            distances.append(float(np.abs(t_val - center)))
        min_dist = min(distances)
        best_cluster = distances.index(min_dist)
        cluster_assignment[eid] = best_cluster
        wcss += min_dist ** 2
        cluster_data[best_cluster].append((eid, t_val))

    return cluster_data, wcss

def update_cluster_centers(cluster_data, n_clusters):
    cluster_centers = []
    for i in range(n_clusters):
        values = cluster_data[i]
        if len(values) > 0:
            t_sum = sum(item[1] for item in values)
            cluster_centers.append(t_sum / len(values))
        else:
            cluster_centers.append(0.0)

    return cluster_centers


def plot_elbow(max_n_clusters, thickness_data, wcss_curve=None,
               elbow_k=None, save_path="elbow_formatted.png"):
    """Save the elbow (WCSS vs k) plot. Pass wcss_curve to reuse an already
    computed curve (aligned to k=1..max_n_clusters); otherwise it is computed
    here. elbow_k, if given, marks the selected number of zones."""
    ks = list(range(1, max_n_clusters + 1))
    if wcss_curve is None:
        wcss_array = []
        for i in ks:
            _, _, wcss = k_means1D(thickness_data, i)
            wcss_array.append(wcss)
    else:
        wcss_array = list(wcss_curve)

    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(8, 6))
        plt.plot(ks, wcss_array, marker='o', markersize=8, linestyle='-',
                 linewidth=2, color='#2c3e50', label='$WCSS$')
        if elbow_k is not None and 1 <= elbow_k <= len(wcss_array):
            plt.axvline(elbow_k, color='#c0392b', linestyle='--', linewidth=1.5,
                        alpha=0.8)
            plt.scatter([elbow_k], [wcss_array[elbow_k - 1]], s=120,
                        facecolors='none', edgecolors='#c0392b', linewidths=2,
                        zorder=5, label=f'Selected $k={elbow_k}$')
        plt.title('Elbow Method for Optimal Number of Clusters ($k$)',
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Number of Clusters ($k$)', fontsize=12)
        plt.ylabel('Within-Cluster Sum of Squares ($WCSS$)', fontsize=12)
        plt.xticks(ks)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    return save_path


def plot_clustered_thickness(problem, cluster_centers, save_path=None,
                             thickness_scale=1.0):
    """
    Plots discretized thickness zones with equally spaced colorbar entries
    using a Red-to-Blue (RdYlBu) colormap.

    thickness_scale multiplies the cluster centers for the colorbar labels so
    they read as the actual full-laminate thickness in mm. For a SMEAR input
    the clustered values are half-stack thicknesses, so pass 2; otherwise 1.
    """
    X = problem.X
    Y = problem.Y
    skin_T = problem.skin_thickness

    # 1. Sort centers so that Index 0 is the thinnest and Index N is the thickest
    sorted_centers = np.sort(np.array(cluster_centers))
    n_clusters = len(sorted_centers)

    # 2. Map each element to the INDEX of the closest center (0, 1, 2...)
    # This is the secret to getting equal spacing on the colorbar
    distances = np.abs(skin_T[:, np.newaxis] - sorted_centers)
    cluster_indices = np.argmin(distances, axis=1)

    plt.figure(figsize=(11, 8))

    # 3. Define the Colormap (Red-Yellow-Blue reversed: Blue=Low, Red=High)
    # Using 'RdYlBu_r' for a standard engineering heatmap
    cmap = plt.get_cmap('RdYlBu_r', n_clusters)

    # Define normalization so colors map exactly to indices 0, 1, 2...
    norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)

    # Plot using cluster_indices for the color data 'c'
    sc = plt.scatter(X, Y, c=cluster_indices, s=15, cmap=cmap, norm=norm,
                     marker='s', edgecolors='none')

    # 4. Create Colorbar with specific ticks
    cbar = plt.colorbar(sc, ticks=np.arange(n_clusters))

    # Replace the index labels (0, 1, 2) with the actual thickness values
    cbar.ax.set_yticklabels([f"{z * thickness_scale:.3f} mm" for z in sorted_centers])
    cbar.set_label("Discretized Thickness Zones", rotation=270, labelpad=20, fontweight='bold')

    plt.axis('equal')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title(f"Clustered Thickness Distribution ($k={n_clusters}$)", fontsize=14, pad=15)

    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linestyle=':', alpha=0.3)

    output_name = save_path or f"clustered_thickness_k{n_clusters}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()


def plot_threshold_masks(problem, cluster_centers, save_prefix="filter_threshold",
                         thickness_scale=1.0):
    """One black/white plot per binary threshold used in the morphological
    filter's threshold decomposition (k zones -> k-1 thresholds). At threshold
    t, an element is black when its zone is >= t (i.e. at least the t-th
    thickness level), white otherwise — exactly the (pre-filter) mask
    (zone_grid >= t) the filter operates on. Elements are drawn as solid filled
    cells (pcolormesh), so black regions read as continuous material. No
    colorbar. thickness_scale rescales the thickness in the title (pass 2 for
    SMEAR half-stack values). Returns the list of saved file paths."""
    X = np.asarray(problem.X, dtype=float)
    Y = np.asarray(problem.Y, dtype=float)
    skin_T = np.asarray(problem.skin_thickness, dtype=float)

    # Same thinnest->thickest indexing as plot_clustered_thickness / zones_to_grid
    sorted_centers = np.sort(np.array(cluster_centers))
    n_clusters = len(sorted_centers)
    distances = np.abs(skin_T[:, np.newaxis] - sorted_centers)
    zone_index = np.argmin(distances, axis=1)

    # Reconstruct the structured mesh grid so each element fills its cell.
    Xr, Yr = np.round(X, 2), np.round(Y, 2)
    ux = np.sort(np.unique(Xr))
    uy = np.sort(np.unique(Yr))
    x_to_ix = {v: i for i, v in enumerate(ux)}
    y_to_iy = {v: i for i, v in enumerate(uy)}
    ix = np.array([x_to_ix[v] for v in Xr])
    iy = np.array([y_to_iy[v] for v in Yr])

    def _cell_edges(c):
        c = np.asarray(c, dtype=float)
        if len(c) == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        mids = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[c[0] - 0.5 * (c[1] - c[0])], mids,
                               [c[-1] + 0.5 * (c[-1] - c[-2])]])

    x_edges, y_edges = _cell_edges(ux), _cell_edges(uy)

    # 0 -> white (below threshold), 1 -> black (at/above threshold); cells with
    # no element are masked (transparent).
    bw_cmap = mcolors.ListedColormap(["white", "black"])
    bw_cmap.set_bad(alpha=0.0)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2)

    # Panel outline (all cells that hold an element) — keeps white below-threshold
    # regions distinguishable from outside the panel.
    domain = np.zeros((len(uy), len(ux)))
    domain[iy, ix] = 1.0

    saved = []
    for t in range(1, n_clusters):
        above = (zone_index >= t).astype(float)
        grid = np.full((len(uy), len(ux)), np.nan)
        grid[iy, ix] = above
        # At threshold t the black cells are zones >= t, so the thinnest zone
        # shown is zone t; label it with that zone's thickness.
        t_cut = sorted_centers[t] * thickness_scale

        plt.figure(figsize=(11, 8))
        plt.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid),
                       cmap=bw_cmap, norm=norm, shading='flat')
        plt.contour(ux, uy, domain, levels=[0.5], colors='0.6', linewidths=1.0)

        plt.axis('equal')
        plt.xlim(x_edges.min(), x_edges.max())
        plt.ylim(y_edges.min(), y_edges.max())
        plt.title(f"Filter threshold {t} of {n_clusters - 1}: "
                  rf"zones $\geq$ {t}  ($t \geq$ {t_cut:.3f} mm)",
                  fontsize=14, pad=15)
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.grid(True, linestyle=':', alpha=0.3)

        path = f"{save_prefix}_{t}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(path)

    return saved