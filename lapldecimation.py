import argparse
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
from scipy.sparse import csr_matrix, diags, eye
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
        type=str,
        required=True,
        help="Path pointing toward the input .nii or .nii.gz file volume.",
    )

    optional = parser.add_argument_group("Other Optional Arguments")
    optional.add_argument(
        "--output",
        "-o",
        type=str,
        default="skeleton_output.npz",
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
        default=1.2,
        help="Contraction weight scalar multiplier variable.",
    )
    optional.add_argument(
        "--w_H",
        type=float,
        default=0.2,
        help="Baseline structural anchor retention weight variable.",
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

    Supports toggling between:
      1. Standard (Isotropic) Laplacian: Purely reciprocal Euclidean distance weights.
      2. Anisotropic Laplacian: Multiplies distance affinity by directional alignment factors
         to penalize longitudinal shrinkage while favoring radial cross-sectional collapse.
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
    W = csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix(
        (degree_values, (range(n_vertices), range(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return D - W


def edge_collapse_decimation(X, adjacency_matrix, min_edge_length):
    """
    Perform structural decimation (E-collapse).

    Do so by merging vertices connected by edges shorter than min_edge_length.
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
    new_adj = csr_matrix(
        (new_data, (new_rows, new_cols)), shape=(len(unique_verts), len(unique_verts))
    )

    return new_X, new_adj


def laplacian_graph_contraction_edt(
    X_init,
    adj_init,
    binary_segmentation=None,
    use_edt=True,
    use_anisotropic=True,
    w_L=1.0,
    w_H_base=0.1,
    beta_edt=1.0,
    delta=0.5,
    max_iter=20,
    tol=1e-3,
    decimate_every=2,
    min_edge_length=0.5,
    alpha_norm=1.5,
    alpha_tang=0.1,
):
    """
    Carry out Laplacian Flow Dynamics.

    It uses optimization using 3D Euclidean Distance Transform (EDT) to dynamically
    scale retention forces and keep centerlines bounded inside the segmentation.

    Parameters
    ----------
    X_init : ndarray of shape (N, 3)
        Initial 3D coordinates of the graph vertices.
    adj_init : csr_matrix of shape (N, N)
        Boolean sparse adjacency matrix representing initial network connectivity.
    binary_segmentation : None, optional
        Description
    use_edt : bool, optional
        Description
    use_anisotropic : bool, optional
        Description
    w_L : float
        Contraction weight coefficient forcing nodes toward neighborhood centers.
    w_H_base : float
        The base baseline retention weight coefficient.
    beta_edt : float, optional
        Description
    delta : float
        A smoothing stabilizer parameter to avoid infinite exponential spikes at boundaries.
    max_iter : int
        Maximum number of flow iterations.
    tol : float
        Convergence tolerance based on average vertex displacement.
    decimate_every : int
        Frequency of contraction steps before triggering structural decimation.
    min_edge_length : float
        The length threshold below which edges will be collapsed.
    alpha_norm : float, optional
        Description
    alpha_tang : float, optional
        Description

    Deleted Parameters
    ------------------
    edt_volume : ndarray of shape (D, H, W)
        Pre-computed 3D Euclidean Distance Transform volume of the vessel mask.
        Voxels represent distance to the nearest background/boundary (higher value = deeper inside).

    No Longer Returned
    ------------------
    X : ndarray (M, 3)
        Contracted centerline coordinates.
    adj : csr_matrix (M, M)
        Decimated skeleton topology graph.
    """
    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    # Conditional 3D EDT Initialization
    edt_volume = None
    if use_edt and binary_segmentation is not None:
        print(
            "Computing 3D Euclidean Distance Transform map for boundary potential well..."
        )
        edt_volume = ndimage.distance_transform_edt(binary_segmentation)
        vol_shape = edt_volume.shape
    elif use_edt and binary_segmentation is None:
        print(
            "Warning: EDT requested but no segmentation mask provided. Falling back to classic approach."
        )
        use_edt = False

    print(
        f"Starting contraction [Anisotropic={use_anisotropic}, EDT={use_edt}] with {X.shape[0]} nodes..."
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
        if use_edt:
            ix = np.clip(np.round(X[:, 0]).astype(int), 0, vol_shape[0] - 1)
            iy = np.clip(np.round(X[:, 1]).astype(int), 0, vol_shape[1] - 1)
            iz = np.clip(np.round(X[:, 2]).astype(int), 0, vol_shape[2] - 1)
            node_distances = edt_volume[ix, iy, iz]
            w_H_per_node = w_H_base * np.exp(beta_edt / (node_distances + delta))
            W_H_sq = diags(w_H_per_node**2, format="csr")
            max_pull = f"- Max EDT w_H Pull: {np.max(w_H_per_node):.4f}"
        else:
            W_H_sq = eye(n_vertices, format="csr") * (w_H_base**2)

        # 3. Solve Implicit Update System equations
        A = (w_L**2) * L_squared + W_H_sq
        B = W_H_sq.dot(X)

        X_next = np.zeros_like(X)
        for dim in range(3):
            X_next[:, dim] = spsolve(A, B[:, dim])

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


def laplacian_skeletonisation(
    nifti_path,
    output_path=None,
    use_edt=True,
    use_anisotropic=True,
    beta_edt=1.0,
    w_L=1.2,
    w_H_base=0.2,
    downsample=False,
):
    """
    Load a nifti file and skeletonise it.

    Parameters
    ----------
    nifti_path : TYPE
        Description
    output_path : None, optional
        Description
    use_edt : bool, optional
        Description
    use_anisotropic : bool, optional
        Description
    beta_edt : float, optional
        Description
    w_L : float, optional
        Description
    w_H_base : float, optional
        Description
    downsample : bool, optional
        Description
    """
    print(f"Ingesting NIfTI image: {nifti_path}")
    img = nib.load(nifti_path)
    volume_data = img.get_fdata().astype(bool)

    vessel_voxels = np.argwhere(volume_data).astype(float)
    if len(vessel_voxels) == 0:
        raise ValueError("Provided segmentation volume lacks any foreground structure.")

    # Downsample points cloud initialization limits if necessary to guard RAM bounds
    if downsample and len(vessel_voxels) > 200000:
        print(
            f"Volume contains {len(vessel_voxels)} points. Structuring node allocations..."
        )
        idx = np.random.choice(len(vessel_voxels), 150000, replace=False)
        X_init = vessel_voxels[idx]
    else:
        X_init = vessel_voxels

    print("Computing proximity network coordinates...")
    dists = cdist(X_init, X_init)
    adj_matrix = (dists > 0) & (dists < 2.5)
    adj_sparse = csr_matrix(adj_matrix)

    contracted_X, final_adj = laplacian_graph_contraction_edt(
        X_init,
        adj_sparse,
        binary_segmentation=volume_data,
        use_edt=use_edt,
        use_anisotropic=use_anisotropic,
        beta_edt=beta_edt,
        w_L=w_L,
        w_H_base=w_H_base,
    )

    if output_path:
        print(f"Saving structural centerline data matrices to: {output_path}")
        np.savez(output_path, coordinates=contracted_X, adjacency=final_adj.toarray())

    return contracted_X, final_adj


def _main(argv=None):
    args = _get_parser().parse_args(argv)

    laplacian_skeletonisation(**vars(args))


if __name__ == "__main__":
    _main(sys.argv[1:])
