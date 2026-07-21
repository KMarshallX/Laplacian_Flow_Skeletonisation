#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from nigsp import io
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser(
        description="Configurable EDT-Guided Laplacian Graph Contraction Pipeline.",
        add_help=False,
    )

    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "--input",
        "-i",
        dest="nifti_path",
        type=str,
        required=True,
        help="Path pointing toward the input .nii or .nii.gz file volume.",
    )

    optional = parser.add_argument_group("Other Optional Arguments")
    optional.add_argument(
        "--output",
        "-o",
        dest="out_path",
        type=str,
        default=None,
        help="Path destination for the generated skeleton arrays.",
    )
    optional.add_argument(
        "--use_edt",
        action="store_true",
        help=(
            "Use distance transform potential constraints on top of classic uniform "
            "retention mapping."
        ),
    )
    optional.add_argument(
        "--use_anisotropic",
        action="store_true",
        help=(
            "Flag to disable anisotropic constraints and fall back to standard "
            "isotropic Laplacian matrix operations."
        ),
    )
    optional.add_argument(
        "--beta_edt",
        type=float,
        default=1.0,
        help=(
            "Custom scaling weight assigned to modulate the EDT boundary energy "
            "constraints."
        ),
    )
    optional.add_argument(
        "--w_L",
        type=float,
        default=0.5,
        help="Contraction weight scalar multiplier variable.",
    )
    optional.add_argument(
        "--w_H",
        dest="w_H_base",
        type=float,
        default=0.5,
        help="Baseline structural anchor retention weight variable.",
    )
    optional.add_argument(
        "--decimate_every",
        dest="decimate_every",
        type=int,
        default=2,
        help="Decimate nodes every N steps [Default=2].",
    )
    optional.add_argument(
        "--downsample",
        action="store_true",
        help="Downsample the original matrix to preserve RAM.",
    )
    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    return parser


def compute_laplacian_matrix(
    X, adjacency_matrix, use_anisotropic=True, alpha_norm=1.5, alpha_tang=0.1
):
    """
    Compute the Graph Laplacian Matrix L = D - W.

    Supports toggling between
    -------------------------
    1. Standard (Isotropic) Laplacian: Purely reciprocal Euclidean distance weights.
    2. Anisotropic Laplacian: Multiplies distance affinity by directional alignment factors
       to penalize longitudinal shrinkage while favoring radial cross-sectional collapse.

    Parameters
    ----------
    X : numpy.ndarray
        An (N, 3) array containing the continuous 3D spatial coordinates of the
        graph vertices/nodes.
    adjacency_matrix : scipy.sparse.spmatrix
        A sparse binary adjacency matrix of shape (N, N) defining the structural
        connectivity profile between the vertices.
    use_anisotropic : bool, optional
        If True, modulates affinity weights using localized directional alignment vectors
        to encourage radial over longitudinal contraction. If False, defaults to classic
        isotropic Euclidean distance reciprocals. Default is True.
    alpha_norm : float, optional
        The scaling coefficient penalty assigned to normal (cross-sectional radial)
        displacement components when `use_anisotropic` is active. Default is 1.5.
    alpha_tang : float, optional
        The scaling coefficient penalty assigned to tangential (longitudinal direction)
        displacement components when `use_anisotropic` is active. Default is 0.1.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The calculated sparse Graph Laplacian Matrix of shape (N, N) governed
        by the equation L = D - W.
    """
    n_vertices = X.shape[0]

    # Get row and col indices from the sparse adjacency matrix
    rows, cols = adjacency_matrix.nonzero()

    # 1. Compute spatial difference vectors and Euclidean distances
    diffs = X[rows] - X[cols]
    distances = np.linalg.norm(diffs, axis=1)
    distances = np.maximum(distances, 1e-6)  # Prevent division by zero

    if use_anisotropic:
        # Estimate local structural tangents using local neighborhood PCA proxy
        tangents = np.zeros_like(X)
        for i in range(n_vertices):
            neighbors = cols[rows == i]
            if len(neighbors) > 1:
                cov = np.cov(X[neighbors].T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                tangents[i] = eigvecs[:, -1]  # Principal directional eigenvector
            else:
                tangents[i] = np.array([1.0, 0.0, 0.0])

        t_i = tangents[rows]
        dot_products = np.sum(diffs * t_i, axis=1)

        # Decompose into longitudinal and cross-sectional components
        tangential_comps = np.abs(dot_products)
        normal_comps = np.linalg.norm(diffs - (dot_products[:, None] * t_i), axis=1)

        # Scale the affinity weights using the anisotropy parameters
        aniso_mod = (alpha_norm * normal_comps) + (alpha_tang * tangential_comps)
        weights = aniso_mod / distances
    else:
        # Standard Isotropic Weights
        weights = 1.0 / distances

    # Assemble sparse operators
    W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = sparse.csr_matrix(
        (degree_values, (range(n_vertices), range(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return D - W


def edge_collapse_decimation(X, adjacency_matrix, min_edge_length):
    """
    Perform structural decimation (E-collapse).

    Merges vertices connected by edges shorter than min_edge_length to maintain
    clean topology and prevent node crowding during graph contraction.

    Parameters
    ----------
    X : numpy.ndarray
        An (N, 3) float array containing the 3D spatial coordinates of the
        graph's vertices, where N is the number of vertices.
    adjacency_matrix : scipy.sparse.spmatrix
        A square, sparse adjacency matrix (e.g., CSR or COO format) of shape (N, N)
        representing the structural connectivity between nodes.
    min_edge_length : float
        The structural distance threshold. Any edge with a Euclidean length shorter
        than this value will be collapsed.

    Returns
    -------
    new_X : numpy.ndarray
        A (M, 3) float array containing the updated spatial coordinates of the remaining
        M unique vertices after simplification.
    new_adj : scipy.sparse.csr_matrix
        A simplified sparse CSR adjacency matrix of shape (M, M) with self-loops
        and duplicate edges removed.
    """
    n_vertices = X.shape[0]
    rows, cols = adjacency_matrix.nonzero()

    # Keep track of which vertices are mapped/merged to which
    vertex_map = np.arange(n_vertices)

    for u, v in zip(rows, cols):
        if u >= v:
            continue  # Only check each unique undirected edge once

        # Check if the edge is shorter than the allowed threshold
        dist = np.linalg.norm(X[u] - X[v])
        if dist < min_edge_length:
            root_u = vertex_map[u]
            root_v = vertex_map[v]
            if root_u != root_v:
                # Merge v into u: update positions to their average
                X[root_u] = (X[root_u] + X[root_v]) / 2.0
                vertex_map[vertex_map == root_v] = root_u

    # Remap unique remaining vertices
    unique_verts, inverse_indices = np.unique(vertex_map, return_inverse=True)
    new_X = X[unique_verts]

    # Rebuild the simplified adjacency matrix
    new_rows = inverse_indices[rows]
    new_cols = inverse_indices[cols]

    # Remove self-loops and duplicates
    valid_mask = new_rows != new_cols
    new_rows = new_rows[valid_mask]
    new_cols = new_cols[valid_mask]

    new_data = np.ones(len(new_rows), dtype=bool)
    new_adj = sparse.csr_matrix(
        (new_data, (new_rows, new_cols)), shape=(len(unique_verts), len(unique_verts))
    )

    return new_X, new_adj


def laplacian_graph_contraction_edt(
    X_init,
    adj_init,
    binary_segmentation=None,
    use_edt=True,
    use_anisotropic=True,
    enforce_containment=False,
    w_L=0.5,
    w_H_base=0.5,
    beta_edt=1.0,
    delta=0.5,
    max_iter=2000,
    tol=0.05,
    decimate_every=2,
    min_edge_length=0.5,
    alpha_norm=1.5,
    alpha_tang=0.1,
):
    """
    Carry out Laplacian Flow Dynamics.

    It uses optimization using 3D Euclidean Distance Transform (EDT) to dynamically
    scale retention forces and optionally enforces hard-voxel mask boundary containment.

    Parameters
    ----------
    X_init : numpy.ndarray
        Initial 3D coordinates of the graph vertices as an (N, 3) array.
    adj_init : scipy.sparse.csr_matrix
        Boolean sparse adjacency matrix representing initial network connectivity of shape (N, N).
    binary_segmentation : numpy.ndarray, optional
        The binary segmentation mask volume used to calculate the EDT profile. Default is None.
    use_edt : bool, optional
        If True, enables the Euclidean Distance Transform boundary potential constraint to prevent
        implosive collapse beyond true anatomy boundaries. Default is True.
    use_anisotropic : bool, optional
        If True, applies directionally weighted affinity rules prioritizing cross-sectional
        radial contraction over structural longitudinal shrinkage. Default is True.
    enforce_containment : bool, optional
        If True, applies a hard projection constraint to force nodes drifting out of the
        foreground mask onto the closest inner boundary shell surface voxel. Default is False.
    w_L : float, optional
        Contraction weight coefficient forcing nodes toward localized neighborhood geometric centers.
        Default is 0.5.
    w_H_base : float, optional
        The baseline structural positional anchor retention weight coefficient. Default is 0.5.
    beta_edt : float, optional
        Scaling parameter modulate exponent behavior of the EDT boundary attraction potential.
        Default is 1.0.
    delta : float, optional
        A smoothing stabilizer parameter added to the denominator to avoid division-by-zero errors
        at exact boundary contours. Default is 0.5.
    max_iter : int, optional
        Maximum allowed iteration steps for the contraction flow solver. Default is 2000.
    tol : float, optional
        Convergence tolerance limit evaluated against mean vertex displacement. Default is 1e-3.
    decimate_every : int, optional
        Frequency cadence interval defining how many contraction loop steps occur before triggering
        an edge-collapse decimation execution. Default is 2.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo structural merging.
        Default is 0.5.
    alpha_norm : float, optional
        The normal/cross-sectional penalty parameter used during anisotropic calculation phases.
        Default is 1.5.
    alpha_tang : float, optional
        The tangential/longitudinal orientation penalty parameter used during anisotropic calculation phases.
        Default is 0.1.

    Returns
    -------
    X : numpy.ndarray
        Contracted centerline coordinates as an (M, 3) matrix.
    adj : scipy.sparse.csr_matrix
        Decimated skeleton topology graph connectivity representation of shape (M, M).
    """
    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    # Conditional 3D EDT & Hard-Voxel Constraint Lookup Precomputation
    edt_volume = None
    closest_vessels_indices = None

    if (use_edt or enforce_containment) and binary_segmentation is not None:
        print("Computing 3D EDT Map and boundary projection lookup tensors...")
        # Inverse transform tells background voxels how far they are from the foreground target mask
        background_edt, nearest_indices = ndimage.distance_transform_edt(
            binary_segmentation == 0, return_indices=True
        )
        edt_volume = ndimage.distance_transform_edt(binary_segmentation)
        closest_vessels_indices = nearest_indices
        vol_shape = binary_segmentation.shape
    elif (use_edt or enforce_containment) and binary_segmentation is None:
        print(
            "Warning: No segmentation mask provided. Falling back to classic approach."
        )
        use_edt = False
        enforce_containment = False

    print(
        f"Starting contraction [Anisotropic={use_anisotropic}, EDT={use_edt}, Hard-Containment={enforce_containment}] "
        f"with {X.shape[0]} nodes..."
    )

    for i in range(max_iter):
        n_vertices = X.shape[0]

        # 1. Compute chosen Laplacian variant
        L = compute_laplacian_matrix(
            X,
            adj,
            use_anisotropic=use_anisotropic,
            alpha_norm=alpha_norm,
            alpha_tang=alpha_tang,
        )
        L_squared = L.dot(L)

        # 2. Extract localized retention matrix mapping
        max_pull = ""
        ix = np.clip(np.round(X[:, 0]).astype(int), 0, vol_shape[0] - 1)
        iy = np.clip(np.round(X[:, 1]).astype(int), 0, vol_shape[1] - 1)
        iz = np.clip(np.round(X[:, 2]).astype(int), 0, vol_shape[2] - 1)

        if use_edt:
            node_distances = edt_volume[ix, iy, iz]
            w_H_per_node = w_H_base * np.exp(beta_edt / (node_distances + delta))
            W_H_sq = sparse.diags(w_H_per_node**2, format="csr")
            max_pull = f" - Max EDT w_H Pull: {np.max(w_H_per_node):.4f}"
        else:
            W_H_sq = sparse.eye(n_vertices, format="csr") * (w_H_base**2)

        # 3. Solve Implicit Update System equations
        A = (w_L**2) * L_squared + W_H_sq
        B = W_H_sq.dot(X)

        X_next = np.zeros_like(X)
        for dim in range(3):
            X_next[:, dim] = spsolve(A, B[:, dim])

        # 4. Explicit Hard-Voxel Containment Constraint Projection
        if enforce_containment:
            # Re-discretize positions to evaluate mask containment state
            ix_next = np.clip(np.round(X_next[:, 0]).astype(int), 0, vol_shape[0] - 1)
            iy_next = np.clip(np.round(X_next[:, 1]).astype(int), 0, vol_shape[1] - 1)
            iz_next = np.clip(np.round(X_next[:, 2]).astype(int), 0, vol_shape[2] - 1)

            # Find points that fell outside the vessel grid (mask == 0)
            escaped_mask = binary_segmentation[ix_next, iy_next, iz_next] == 0
            escaped_count = np.sum(escaped_mask)

            if escaped_count > 0:
                # Extract precomputed closest coordinate index maps for escaped nodes
                proj_x = closest_vessels_indices[0][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]
                proj_y = closest_vessels_indices[1][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]
                proj_z = closest_vessels_indices[2][
                    ix_next[escaped_mask], iy_next[escaped_mask], iz_next[escaped_mask]
                ]

                # Project continuous coordinates onto the target boundary shell voxels
                X_next[escaped_mask] = np.stack(
                    [proj_x, proj_y, proj_z], axis=1
                ).astype(float)
                max_pull += f" [Projected: {escaped_count} escaped nodes]"

        displacement = np.mean(np.linalg.norm(X_next - X, axis=1))
        X = X_next

        print(
            f"Iter {i + 1}/{max_iter} - Remaining Nodes: {X.shape[0]} - "
            f"Error Drift: {displacement:.5f}{max_pull}"
        )

        if displacement < tol:
            print("Convergence criteria reached.")
            break

        if (i + 1) % decimate_every == 0:
            X, adj = edge_collapse_decimation(X, adj, min_edge_length)

    return X, adj


def graph_to_dense_3d(X, adjacency_matrix, target_shape):
    """
    Rasterizes an abstract graph topology into a dense 3D binary volume.

    Parameters
    ----------
    X : ndarray of shape (N, 3)
        The 3D coordinates of the vertices/nodes.
    adjacency_matrix : csr_matrix of shape (N, N)
        The sparse connectivity matrix representing edges.
    target_shape : tuple of int (D, H, W)
        The structural grid dimensions of the target 3D matrix.

    Returns
    -------
    dense_volume : ndarray of shape (D, H, W)
        A binary 3D array where 1 represents the skeleton path.
    """
    # 1. Initialize empty dense matrix
    dense_volume = np.zeros(target_shape, dtype=np.uint8)

    # 2. Extract edge pairs from the sparse adjacency matrix
    rows, cols = sparse.triu(adjacency_matrix).nonzero()

    # 3. Rasterize edges and nodes into the grid
    for u, v in zip(rows, cols):
        p1 = X[u]
        p2 = X[v]

        # Calculate distance between vertices to determine how many samples to take
        dist = np.linalg.norm(p1 - p2)
        num_samples = max(int(np.ceil(dist * 2)), 2)  # Sample at sub-voxel resolution

        # Linearly interpolate points between vertex u and vertex v
        t = np.linspace(0, 1, num_samples)
        line_points = p1[None, :] * (1 - t[:, None]) + p2[None, :] * t[:, None]

        # Round coordinates to the nearest discrete voxel indices
        voxels = np.round(line_points).astype(int)

        # Clip indices to prevent out-of-bounds array crashing
        voxels[:, 0] = np.clip(voxels[:, 0], 0, target_shape[0] - 1)
        voxels[:, 1] = np.clip(voxels[:, 1], 0, target_shape[1] - 1)
        voxels[:, 2] = np.clip(voxels[:, 2], 0, target_shape[2] - 1)

        # Burn the line into our dense volume
        dense_volume[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1

    return dense_volume


def coords_to_dense_3d(X, target_shape):
    """
    Rasterizes an abstract graph topology into a dense 3D binary volume.

    Parameters
    ----------
    X : ndarray of shape (N, 3)
        The 3D coordinates of the areas with content.
    target_shape : tuple of int (D, H, W)
        The structural grid dimensions of the target 3D matrix.

    Returns
    -------
    dense_volume : ndarray of shape (D, H, W)
        A binary 3D array where 1 represents the skeleton path.
    """
    # 1. Initialize empty dense matrix
    dense_volume = np.zeros(target_shape, dtype=bool)

    coords = np.rint(X).astype(np.int8)

    # 3. Rasterize edges and nodes into the grid
    for i in coords:
        dense_volume[tuple(i)] = True

    return dense_volume


def laplacian_skeletonisation(
    nifti_path,
    out_path=None,
    use_edt=True,
    use_anisotropic=True,
    beta_edt=1.0,
    w_L=0.5,
    w_H_base=0.5,
    decimate_every=2,
    downsample=False,
):
    """
    Load a NIfTI file volume image and perform geometric graph contraction skeletonisation.

    Parameters
    ----------
    nifti_path : str
        File system location path pointing directly toward the source input .nii or .nii.gz file.
    out_path : str, optional
        Output destination storage path base where resulting arrays and generated skeleton
        files will be written. Default is None (autogenerated from input file name).
    use_edt : bool, optional
        Enables boundary tracking potential constraints using Euclidean Distance Transforms.
        Default is True.
    use_anisotropic : bool, optional
        Enables anisotropic geometry handling to penalize internal longitudinal shortening vectors.
        Default is True.
    beta_edt : float, optional
        Scaling modulation weight assigned to boundary energy calculation properties. Default is 1.0.
    w_L : float, optional
        Contraction weight step modifier targeting structural local geometric collapse. Default is 0.5.
    w_H_base : float, optional
        Baseline structural node anchor positional persistence value metric. Default is 0.5.
    decimate_every : int, optional
        Sampling cadence sequence gap length setting how frequently graph e-collapses execute.
        Default is 2.
    downsample : bool, optional
        Flag setting whether point arrays containing high density are uniformly downsampled
        to stay within safe RAM footprints. Default is False.

    Returns
    -------
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).

    Raises
    ------
    ValueError
        If the loaded structural NIfTI mask image is completely empty or lacks foreground elements.
    """
    print(f"Ingesting NIfTI image: {nifti_path}")
    _, volume_data, img = io.load_nifti_get_mask(nifti_path, is_mask=True, ndim=3)

    vessel_voxels = np.argwhere(volume_data).astype(float)
    if len(vessel_voxels) == 0:
        raise ValueError("Provided segmentation volume lacks any foreground structure.")

    # Downsample points cloud initialization limits if necessary to guard RAM bounds
    if downsample and len(vessel_voxels) > 200000:
        print(f"Volume contains {len(vessel_voxels)} points. Downsampling.")
        idx = np.random.choice(len(vessel_voxels), 150000, replace=False)
        X_init = vessel_voxels[idx]
    else:
        X_init = vessel_voxels

    print("Computing proximity network coordinates...")
    dists = cdist(X_init, X_init)
    adj_matrix = (dists > 0) & (dists < 2.5)
    adj_sparse = sparse.csr_matrix(adj_matrix)

    contracted_X, final_adj = laplacian_graph_contraction_edt(
        X_init,
        adj_sparse,
        binary_segmentation=volume_data,
        use_edt=use_edt,
        use_anisotropic=use_anisotropic,
        beta_edt=beta_edt,
        w_L=w_L,
        w_H_base=w_H_base,
        decimate_every=decimate_every,
    )

    out_path = (
        out_path
        if out_path
        else f"{os.path.splitext(os.path.splitext(nifti_path)[0])[0]}_skel"
    )

    print(f"Saving structural centerline data matrices to: {out_path}")
    np.savez_compressed(f"{out_path}_coords.npz", contracted_X=contracted_X)
    sparse.save_npz(f"{out_path}.npz", final_adj)
    nifti_skel = coords_to_dense_3d(contracted_X, volume_data.shape)
    io.export_nifti(nifti_skel, img, f"{out_path}.nii.gz")

    return contracted_X, final_adj


def _main(argv=None):
    args = _get_parser().parse_args(argv)

    laplacian_skeletonisation(**vars(args))


if __name__ == "__main__":
    _main(sys.argv[1:])
