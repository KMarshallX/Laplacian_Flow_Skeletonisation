import numpy as np
import scipy.ndimage as ndimage
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist


def compute_laplacian_matrix(X, adjacency_matrix):
    """
    Compute the standard Graph Laplacian Matrix L = D - W.

    Using simple reciprocal Euclidean distance weights for spatial affinity.
    """
    n_vertices = X.shape[0]

    # Get row and col indices from the sparse adjacency matrix
    rows, cols = adjacency_matrix.nonzero()

    # Calculate Euclidean distances for connected pairs
    distances = np.linalg.norm(X[rows] - X[cols], axis=1)

    # Avoid division by zero for identical overlapping vertices
    distances = np.maximum(distances, 1e-6)
    weights = 1.0 / distances

    # Build weight matrix Wn_vertices
    W = csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix(
        (degree_values, (range(n_vertices), range(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    L = D - W
    return L


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
    edt_volume,
    w_L=1.0,
    w_H_base=0.1,
    delta=0.5,
    max_iter=20,
    tol=1e-3,
    decimate_every=2,
    min_edge_length=0.5,
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
    edt_volume : ndarray of shape (D, H, W)
        Pre-computed 3D Euclidean Distance Transform volume of the vessel mask.
        Voxels represent distance to the nearest background/boundary (higher value = deeper inside).
    w_L : float
        Contraction weight coefficient forcing nodes toward neighborhood centers.
    w_H_base : float
        The base baseline retention weight coefficient.
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

    Returns
    -------
    X : ndarray (M, 3)
        Contracted centerline coordinates.
    adj : csr_matrix (M, M)
        Decimated skeleton topology graph.
    """
    X = X_init.copy().astype(float)
    adj = adj_init.copy()

    vol_shape = edt_volume.shape
    print(f"Starting EDT-guided contraction with {X.shape[0]} vertices...")

    for i in range(max_iter):
        # 1. Compute the current graph Laplacian matrix L
        L = compute_laplacian_matrix(X, adj)
        L_squared = L.dot(L)

        # 2. Extract local EDT values for every node via nearest voxel mapping
        # Clip coordinates to guarantee they stay within grid array dimensions
        ix = np.clip(np.round(X[:, 0]).astype(int), 0, vol_shape[0] - 1)
        iy = np.clip(np.round(X[:, 1]).astype(int), 0, vol_shape[1] - 1)
        iz = np.clip(np.round(X[:, 2]).astype(int), 0, vol_shape[2] - 1)

        node_distances = edt_volume[ix, iy, iz]

        # 3. Calculate spatially varying retention forces w_H per node
        # As distance to background boundary -> 0, w_H scales up exponentially
        w_H_per_node = w_H_base * np.exp(1.0 / (node_distances + delta))

        # Construct diagonal matrices for the updated matrix formulation:
        # (w_L^2 * L^2 + W_H^2) X^(t+1) = W_H^2 * X^(t)
        W_H_sq = diags(w_H_per_node**2, format="csr")

        A = (w_L**2) * L_squared + W_H_sq
        B = W_H_sq.dot(X)

        # 4. Solve the Implicit Linear System independently for X, Y, Z axes
        X_next = np.zeros_like(X)
        for dim in range(3):
            X_next[:, dim] = spsolve(A, B[:, dim])

        displacement = np.mean(np.linalg.norm(X_next - X, axis=1))
        X = X_next

        print(
            f"Iter {i + 1}/{max_iter} - Max w_H Pull: {np.max(w_H_per_node):.4f} -"
            f" Nodes: {X.shape[0]}"
        )

        if displacement < tol:
            print("Convergence criteria met.")
            break

        # 5. Structural Decimation Step (E-collapse)
        if (i + 1) % decimate_every == 0:
            X, adj = edge_collapse_decimation(X, adj, min_edge_length)

    return X, adj


# ==========================================
# EXAMPLE VALIDATION: Simulating a Hollow Tube Volume
# ==========================================
if __name__ == "__main__":
    # 1. Create a binary mask volume containing a 3D horizontal tube vessel segment
    volume_size = (30, 30, 30)
    binary_mask = np.zeros(volume_size, dtype=bool)

    # Fill array elements where Y-Z radius <= 6 to carve out a solid tube along the X axis
    grid_x, grid_y, grid_z = np.indices(volume_size)
    radial_dist_from_center = np.sqrt((grid_y - 15) ** 2 + (grid_z - 15) ** 2)
    binary_mask[(radial_dist_from_center <= 6) & (grid_x >= 2) & (grid_x <= 27)] = True

    # Compute the 3D Euclidean Distance Transform map
    # Background voxels are 0; Internal core centerline voxels peak at around 6.0
    edt_volume = ndimage.distance_transform_edt(binary_mask)

    # 2. Extract initial point coordinates lying right on the outer hull of the vessel
    # (Mimics noisy graph initialization configurations bound by outer boundaries)
    vessel_boundary_indices = np.argwhere(
        (radial_dist_from_center >= 5)
        & (radial_dist_from_center <= 6)
        & (grid_x >= 3)
        & (grid_x <= 26)
    )
    X_vessel = vessel_boundary_indices.astype(float)

    # Establish proximity connectivity graph [cite: 20]
    dists = cdist(X_vessel, X_vessel)
    adj_matrix = (dists > 0) & (dists < 2.5)
    adj_sparse = csr_matrix(adj_matrix)

    # 3. Run the EDT-Guided Laplacian flow contraction
    contracted_X, final_adj = laplacian_graph_contraction_edt(
        X_vessel,
        adj_sparse,
        edt_volume=edt_volume,
        w_L=1.5,
        w_H_base=0.1,
        delta=0.4,
        max_iter=12,
        decimate_every=2,
        min_edge_length=1.0,
    )

    # Confirm that all centerline endpoints are strictly contained inside the segmentation domain
    ix = np.clip(np.round(contracted_X[:, 0]).astype(int), 0, volume_size[0] - 1)
    iy = np.clip(np.round(contracted_X[:, 1]).astype(int), 0, volume_size[1] - 1)
    iz = np.clip(np.round(contracted_X[:, 2]).astype(int), 0, volume_size[2] - 1)
    inside_count = np.sum(binary_mask[ix, iy, iz])

    print("\nSkeletonization Complete!")
    print(f"Total Contracted Nodes: {contracted_X.shape[0]}")
    print(
        f"Nodes successfully locked inside the segmentation container: {inside_count} out of {contracted_X.shape[0]}"
    )
