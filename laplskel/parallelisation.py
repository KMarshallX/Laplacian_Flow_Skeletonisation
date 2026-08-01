import os

import numpy as np
from joblib import delayed
from scipy.ndimage import find_objects
from scipy.spatial import KDTree
from tqdm_joblib import ParallelPbar

from .contraction import laplacian_graph_contraction
from .graph import compute_sparse_adjacency_matrix


def _process_single_label(
    label_id,
    cropped_label,
    offset_origin,
    use_edt,
    use_anisotropic,
    enforce_containment,
    beta_edt,
    w_L,
    w_H_base,
    tol,
    max_distance,
    decimate_every,
    min_edge_length,
    num_features,
    solver,
):
    """
    Worker function to process a single connected component label.

    Parameters
    ----------
    label_id : int
        ID of current label
    cropped_label : np.ndarray
        Segmentation labelled with scipy's label and cropped with find_objects.
    offset_origin : list
        cropped offset.
    use_edt : bool
        Enables boundary tracking potential constraints using Euclidean Distance Transforms.
    use_anisotropic : bool
        Enables anisotropic geometry handling to penalize internal longitudinal
        shortening vectors.
    enforce_containment : bool
        If True, applies a hard projection constraint to force nodes drifting out of the
        foreground mask onto the closest inner boundary shell surface voxel.
    beta_edt : float
        Scaling modulation weight assigned to boundary energy calculation properties.
    w_L : float
        Contraction weight step modifier targeting structural local geometric collapse.
        This should be alpha in Damseh 2021.
    w_H_base : float
        Baseline structural node anchor positional persistence value metric.
        This should be equivalent to beta in Damseh 2021.
    tol : float
        Convergence tolerance limit evaluated against mean vertex displacement.
        This should be the equivalent of gamma in Damseh 2021 (not sure).
    max_distance : float
        Maximum distance to consider when making the sparse adjacency matrix.
    decimate_every : int
        Frequency cadence interval defining how many contraction loop steps occur before
        triggering an edge-collapse decimation execution.
    min_edge_length : float
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, i.e. the isotropic voxel size of the grid used for
        decimation.
    num_features : int
        Number of extracted labels.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it far faster, but may require a tad more memory.

    Returns
    -------
    label_id : int
        ID of current label (For tracking)
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    """
    X_init_local = np.argwhere(cropped_label).astype(np.uint16)
    tree = KDTree(X_init_local)

    # Skip small noise components
    if len(X_init_local) <= 3:
        adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

        X_init_global = X_init_local + np.array(offset_origin, dtype=np.float32)
        return label_id, X_init_global, adj_sparse

    print(
        f'\n--- Processing Label {label_id}/{num_features} ({X_init_local.sum()} voxels) ---'
    )

    print('Computing proximity network coordinates...')
    adj_sparse = compute_sparse_adjacency_matrix(tree, max_distance)

    # Run contraction on this label's component mask
    label_X_local, label_adj = laplacian_graph_contraction(
        X_init_local,
        adj_sparse,
        binary_segmentation=cropped_label,
        use_edt=use_edt,
        use_anisotropic=use_anisotropic,
        enforce_containment=enforce_containment,
        beta_edt=beta_edt,
        w_L=w_L,
        w_H_base=w_H_base,
        tol=tol,
        decimate_every=decimate_every,
        min_edge_length=min_edge_length,
        solver=solver,
    )

    label_X_global = label_X_local + np.array(offset_origin, dtype=np.float32)

    return label_id, label_X_global, label_adj


def process_components(
    labeled_volume,
    num_features,
    use_edt,
    use_anisotropic,
    enforce_containment,
    beta_edt,
    w_L,
    w_H_base,
    tol,
    max_distance,
    decimate_every,
    min_edge_length,
    n_jobs,
    solver,
):
    """Process labeled segmentation components in parallel."""
    total_cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs <= 0:
        n_workers = max(1, int(np.floor(0.30 * total_cores)))
    else:
        n_workers = min(n_jobs, total_cores)

    print(
        f'Processing {num_features} components in parallel using {n_workers} worker(s) '
        f'on {total_cores} CPU cores detected.'
    )

    slices_list = find_objects(labeled_volume)

    tasks = []
    for label_id in range(1, num_features + 1):
        bbox_slice = slices_list[label_id - 1]

        if bbox_slice is None:
            continue

        # Extract cropped boolean mask for ONLY this label
        cropped_label = labeled_volume[bbox_slice] == label_id

        # Offset origin (min_x, min_y, min_z) used to map back to original volume
        offset_origin = (
            bbox_slice[0].start,
            bbox_slice[1].start,
            bbox_slice[2].start,
        )

        tasks.append(
            delayed(_process_single_label)(
                label_id,
                cropped_label,
                offset_origin,
                use_edt,
                use_anisotropic,
                enforce_containment,
                beta_edt,
                w_L,
                w_H_base,
                tol,
                max_distance,
                decimate_every,
                min_edge_length,
                num_features,
                solver,
            )
        )

    results = ParallelPbar('Skeletonising')(n_jobs=n_workers, batch_size=1)(tasks)

    return results
