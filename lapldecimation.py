import numpy as np
from scipy.sparse import csr_matrix, eye
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

    # Build weight matrix W
    W = csr_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))

    # Build diagonal degree matrix D
    degree_values = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix(
        (degree_values, (range(n_vertices), range(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    # L = D - W
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


def laplacian_graph_contraction(
    X_init,
    adj_init,
    w_L=1.0,
    w_H=0.1,
    max_iter=20,
    tol=1e-3,
    decimate_every=2,
    min_edge_length=0.5,
):
    """
    Carries out the Laplacian Flow Dynamics optimization loop for graph contraction.

    Parameters
    ----------
    X_init : ndarray of shape (N, 3)
        Initial 3D coordinates of the graph vertices embedded within the vessel segmentations.
    adj_init : csr_matrix of shape (N, N)
        Boolean/binary sparse adjacency matrix representing the initial network connectivity.
    w_L : float
        Contraction weight coefficient forcing nodes toward their local neighborhood center.
    w_H : float
        Retention weight coefficient acting as a longitudinal anchor.
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

    print(f"Starting contraction with {X.shape[0]} vertices...")

    for i in range(max_iter):
        n_vertices = X.shape[0]

        # 1. Compute the current graph Laplacian matrix L
        L = compute_laplacian_matrix(X, adj)

        # 2. Formulate the Implicit Linear System: (w_L^2 * L^2 + w_H^2 * I) X^(t+1) = w_H^2 * X^(t)
        # To avoid computing L^2 explicitly (which ruins sparsity), we can stack systems or solve carefully.
        # Alternatively, a common way to evaluate L^2 implicitly or directly:
        L_squared = L.dot(L)

        A = (w_L**2) * L_squared + (w_H**2) * eye(n_vertices, format="csr")
        B = (w_H**2) * X

        # 3. Solve for the updated positions X_next
        X_next = np.zeros_like(X)
        for dim in range(3):  # Solve independently for X, Y, Z axes
            X_next[:, dim] = spsolve(A, B[:, dim])

        # Calculate displacement to track convergence
        displacement = np.mean(np.linalg.norm(X_next - X, axis=1))
        X = X_next

        print(
            f"Iteration {i + 1}/{max_iter} - Displacement Error: {displacement:.6f} - "
            f"Nodes: {X.shape[0]}"
        )

        # Check convergence condition
        if displacement < tol:
            print("Convergence criteria met.")
            break

        # 4. Structural Decimation Step (E-collapse)
        if (i + 1) % decimate_every == 0:
            X, adj = edge_collapse_decimation(X, adj, min_edge_length)

    return X, adj


# ==========================================
# EXAMPLE USAGE: Simulating a Hollow Cylinder/Vessel Tube
# ==========================================
if __name__ == "__main__":
    # 1. Generate dummy point cloud of a 3D cylindrical tube (vessel segment)
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    theta = np.random.uniform(0, 2 * np.pi, 200)
    radius = 2.0

    # Add coordinates: X creates the length, Y and Z create the vessel tube volume
    x_coords = t
    y_coords = radius * np.cos(theta)
    z_coords = radius * np.sin(theta)
    X_vessel = np.column_stack((x_coords, y_coords, z_coords))

    # 2. Compute a proximity-based initial geometric graph graph adjacency (k-NN proxy)
    # Connect pairs that are physically close within the volume
    dists = cdist(X_vessel, X_vessel)
    adj_matrix = dists < 1.5
    np.fill_diagonal(adj_matrix, 0)  # clear self-connections
    adj_sparse = csr_matrix(adj_matrix)

    # 3. Run the Laplacian flow contraction
    # w_L > w_H to favor rapid inward radial collapse over retention
    contracted_X, final_adj = laplacian_graph_contraction(
        X_vessel,
        adj_sparse,
        w_L=1.2,
        w_H=0.2,
        max_iter=10,
        decimate_every=2,
        min_edge_length=0.8,
    )

    print("\nSkeletonization Complete!")
    print(f"Original Volume Points: {X_vessel.shape[0]}")
    print(f"Contracted Centerline Nodes: {contracted_X.shape[0]}")
    print(
        "Average Y-Z deviation from center (ideal is 0): "
        f"{np.mean(np.abs(contracted_X[:, 1:])):.4f}"
    )
