"""Experimental alternating geometric contraction and topological thinning."""

import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, cKDTree

from .contraction import laplacian_graph_contraction
from .graph import compute_sparse_adjacency_matrix
from .medial import estimate_graph_tangents, transverse_edt_targets
from .refinement import (
    _simplify_paths,
    _thin_foreground,
    _tunnel_graph,
    branch_complex_signature,
    foreground_topology,
    thin_foreground_sweep,
)
from .utils import _edge_voxels, _foreground_voxel


def _adjacency_from_edges(n_vertices, edges):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(edges):
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
    else:
        rows = cols = np.array([], dtype=int)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )


def _skeleton_state(mask, guidance):
    voxels = _thin_foreground(mask, guidance).astype(int)
    edges, _ = _tunnel_graph(voxels)
    return voxels, edges, branch_complex_signature(voxels, edges)


def _rasterize_paths(paths, mask):
    """Rasterize fitted polylines and return their unique foreground voxels."""
    output = np.zeros_like(mask, dtype=bool)
    for path in paths.values():
        for start, end in zip(path[:-1], path[1:]):
            for voxel in _edge_voxels(start, end, mask.shape, mask):
                output[voxel] = True
    return output, np.argwhere(output)


def _rasterize_edges(coordinates, edges, mask):
    paths = {
        (int(u), int(v)): coordinates[[u, v]] for u, v in np.asarray(edges)
    }
    return _rasterize_paths(paths, mask)[0]


def _critical_branch_data(coordinates, edges, edt, spacing):
    """Return critical-node radii and shortest incident branch lengths."""
    neighbours = [set() for _ in coordinates]
    for u, v in edges:
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))
    degrees = np.array([len(items) for items in neighbours])
    critical = np.flatnonzero(degrees != 2)
    radii = {}
    lengths = {}
    physical = coordinates * spacing
    for start in critical:
        branch_radii = []
        branch_lengths = []
        for neighbour in neighbours[start]:
            previous, current = int(start), int(neighbour)
            distance = np.linalg.norm(physical[current] - physical[previous])
            samples = [edt[tuple(coordinates[current].astype(int))]]
            while degrees[current] == 2:
                following = next(iter(neighbours[current] - {previous}))
                distance += np.linalg.norm(physical[following] - physical[current])
                previous, current = current, following
                samples.append(edt[tuple(coordinates[current].astype(int))])
            branch_lengths.append(float(distance))
            branch_radii.append(float(np.max(samples)))
        if branch_lengths:
            lengths[int(start)] = min(branch_lengths)
            if degrees[start] == 1:
                radii[int(start)] = branch_radii[0]
            else:
                radii[int(start)] = float(np.median(branch_radii))
        else:
            lengths[int(start)] = 0.0
            radii[int(start)] = float(edt[tuple(coordinates[start].astype(int))])
    return degrees, critical, radii, lengths


def _match_critical_nodes(reference, reference_edges, current, current_edges, spacing):
    """Match movable critical points to fixed references by degree and distance."""
    ref_degree = np.bincount(
        np.asarray(reference_edges).ravel(), minlength=len(reference)
    )
    ref_critical = np.flatnonzero(ref_degree != 2)
    cur_degree = np.bincount(
        np.asarray(current_edges).ravel(), minlength=len(current)
    )
    mapping = {}
    for degree in sorted(set(ref_degree[ref_critical])):
        ref_group = ref_critical[ref_degree[ref_critical] == degree]
        cur_group = np.flatnonzero((cur_degree == degree) & (cur_degree != 2))
        if len(ref_group) != len(cur_group):
            return None
        costs = np.linalg.norm(
            (current[cur_group, None] - reference[ref_group][None]) * spacing,
            axis=2,
        )
        rows, cols = linear_sum_assignment(costs)
        mapping.update(
            {int(cur_group[row]): int(ref_group[col]) for row, col in zip(rows, cols)}
        )
    return mapping


def _junction_targets(coordinates, adjacency, ridge_targets, tangents, spacing):
    """Fit each junction to the least-squares meeting of incident ridge lines."""
    targets = ridge_targets.copy()
    degree = adjacency.getnnz(axis=1)
    physical_ridges = ridge_targets * spacing
    for vertex in np.flatnonzero(degree >= 3):
        begin, end = adjacency.indptr[vertex : vertex + 2]
        neighbours = adjacency.indices[begin:end]
        system = np.zeros((3, 3), dtype=float)
        target = np.zeros(3, dtype=float)
        for neighbour in neighbours:
            direction = tangents[neighbour]
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm
            normal = np.eye(3) - np.outer(direction, direction)
            system += normal
            target += normal @ physical_ridges[neighbour]
        if np.linalg.matrix_rank(system) >= 2:
            targets[vertex] = np.linalg.lstsq(system, target, rcond=None)[0] / spacing
    return targets


def fit_branch_complex(
    reference_voxels,
    reference_edges,
    skeleton_voxels,
    skeleton_edges,
    original_mask,
    original_edt,
    spacing,
    merge_tolerance,
    max_iterations=20,
):
    """Jointly centre and smooth branches while retaining the discrete complex."""
    coordinates = np.asarray(skeleton_voxels, dtype=float)
    edges = np.asarray(skeleton_edges, dtype=int).reshape(-1, 2)
    adjacency = _adjacency_from_edges(len(coordinates), edges)
    degree = adjacency.getnnz(axis=1)
    reference = np.asarray(reference_voxels, dtype=float)
    reference_edges = np.asarray(reference_edges, dtype=int).reshape(-1, 2)
    mapping = _match_critical_nodes(
        reference, reference_edges, coordinates, edges, spacing
    )
    if mapping is None:
        raise RuntimeError(
            'Final skeleton no longer matches the reference branch complex.'
        )
    ref_degree, _, ref_radii, ref_lengths = _critical_branch_data(
        reference, reference_edges, original_edt, spacing
    )
    expected_topology = foreground_topology(original_mask)

    stable = False
    for _ in range(max_iterations):
        local_radii = ndimage.map_coordinates(
            original_edt, coordinates.T, order=1, mode='nearest'
        )
        window = max(2.0 * float(np.min(spacing)), float(np.median(local_radii)))
        tangents = estimate_graph_tangents(coordinates, adjacency, spacing, window)
        ridge = transverse_edt_targets(
            original_edt, original_mask, coordinates, tangents, spacing
        )
        ridge = _junction_targets(coordinates, adjacency, ridge, tangents, spacing)
        neighbour_sum = adjacency.astype(float).dot(coordinates)
        neighbour_mean = neighbour_sum / np.maximum(degree[:, None], 1)
        proposal = 0.4 * ridge + 0.6 * neighbour_mean

        # Critical points move from their preliminary reference only within the
        # agreed radius- and branch-length-scaled cumulative caps.
        for current_index, reference_index in mapping.items():
            reference_position = reference[reference_index]
            radius = ref_radii[reference_index]
            branch_length = ref_lengths[reference_index]
            coefficient = 0.25 if ref_degree[reference_index] == 1 else 0.5
            cap = min(coefficient * radius, 0.2 * branch_length)
            displacement = (proposal[current_index] - reference_position) * spacing
            norm = np.linalg.norm(displacement)
            if norm > cap > 0:
                proposal[current_index] = (
                    reference_position + displacement * (cap / norm) / spacing
                )
            elif cap == 0:
                proposal[current_index] = reference_position

        # A proposed joint update is accepted only where every incident straight
        # segment remains inside the original foreground voxel complex.
        accepted = proposal.copy()
        for _ in range(len(coordinates) + 1):
            rejected = set()
            for vertex, point in enumerate(accepted):
                try:
                    _foreground_voxel(point, original_mask.shape, original_mask)
                except ValueError:
                    rejected.add(vertex)
            for u, v in edges:
                try:
                    _edge_voxels(
                        accepted[u], accepted[v], original_mask.shape, original_mask
                    )
                except ValueError:
                    rejected.update((int(u), int(v)))
            if not rejected:
                break
            accepted[list(rejected)] = coordinates[list(rejected)]
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = coordinates + fraction * (accepted - coordinates)
            try:
                trial_mask = _rasterize_edges(trial, edges, original_mask)
            except ValueError:
                continue
            if foreground_topology(trial_mask) == expected_topology:
                accepted = trial
                break
        else:
            accepted = coordinates
        movement = np.max(np.linalg.norm((accepted - coordinates) * spacing, axis=1))
        coordinates = accepted
        if movement < 0.05 * float(np.min(spacing)):
            stable = True
            break

    local_radii = ndimage.map_coordinates(
        original_edt, coordinates.T, order=1, mode='nearest'
    )
    direction_window = max(
        2.0 * float(np.min(spacing)), float(np.median(local_radii))
    )
    tangents = estimate_graph_tangents(
        coordinates, adjacency, spacing, direction_window
    )
    ridge = transverse_edt_targets(
        original_edt, original_mask, coordinates, tangents, spacing
    )
    ridge = _junction_targets(coordinates, adjacency, ridge, tangents, spacing)
    ridge_distance = np.linalg.norm((coordinates - ridge) * spacing, axis=1)
    ridge_tolerance = np.maximum(
        0.5 * float(np.min(spacing)), 0.2 * local_radii
    )
    diagnostics = {
        'stable': stable,
        'ridge_satisfied': bool(np.all(ridge_distance <= ridge_tolerance)),
        'mean_ridge_distance': float(np.mean(ridge_distance)),
        'max_ridge_distance': float(np.max(ridge_distance)),
    }
    coordinates, simplified_edges, paths = _simplify_paths(
        coordinates, edges, original_mask, merge_tolerance
    )
    return (
        coordinates,
        _adjacency_from_edges(len(coordinates), simplified_edges),
        paths,
        diagnostics,
    )


def alternating_graph_skeletonisation(
    mask,
    spacing=(1.0, 1.0, 1.0),
    contraction_steps=5,
    max_cycles=100,
    use_anisotropic=True,
    w_L=0.5,
    w_H_base=0.5,
    w_H_medial=1.0,
    tol=0.05,
    local_pca_hops=1,
    solver='CG',
    merge_tolerance=0.25,
):
    """Alternate bounded contraction and one-layer topology-safe thinning.

    The original foreground and physical EDT stay fixed. The preliminary branch
    complex is authoritative: a sweep that changes any terminal/junction attachment
    signature or independent cycle is rolled back.
    """
    foreground = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if foreground.ndim != 3 or not np.any(foreground):
        raise ValueError('Alternating skeletonisation requires a nonempty 3D mask.')
    if contraction_steps < 1 or max_cycles < 1:
        raise ValueError('contraction_steps and max_cycles must be positive.')
    if spacing.shape != (3,) or np.any(spacing <= 0) or np.any(~np.isfinite(spacing)):
        raise ValueError('spacing must contain three positive finite values.')
    if foreground_topology(foreground)[2]:
        raise ValueError(
            'Foreground contains enclosed cavities; a curve graph cannot preserve them.'
        )

    original_edt = ndimage.distance_transform_edt(
        np.pad(foreground, 1), sampling=spacing
    )[1:-1, 1:-1, 1:-1]
    initial = np.argwhere(foreground).astype(float)
    topology_guidance = initial.copy()
    reference_voxels, reference_edges, reference_signature = _skeleton_state(
        foreground, topology_guidance
    )
    working = foreground.copy()
    previous_voxels = None
    previous_guide = None
    last_skeleton_voxels = reference_voxels
    last_skeleton_edges = reference_edges
    outcome = 'budget-limited'
    stagnant_cycles = 0

    for cycle in range(max_cycles):
        voxels = np.argwhere(working).astype(float)
        adjacency = compute_sparse_adjacency_matrix(KDTree(voxels), 26)
        if previous_voxels is None:
            contraction_start = voxels
        else:
            correspondence = cKDTree(previous_voxels).query(voxels)[1]
            contraction_start = voxels + (
                previous_guide[correspondence] - previous_voxels[correspondence]
            )
        contracted, _ = laplacian_graph_contraction(
            contraction_start,
            adjacency,
            binary_segmentation=foreground,
            use_edt=False,
            use_anisotropic=use_anisotropic,
            enforce_containment=True,
            w_L=w_L,
            w_H_base=w_H_base,
            w_H_medial=w_H_medial,
            max_iter=contraction_steps,
            tol=tol,
            decimate_every=contraction_steps + 1,
            min_edge_length=0.0,
            local_pca_hops=local_pca_hops,
            solver=solver,
            provisional=True,
        )
        sampled_radii = original_edt[tuple(voxels.astype(int).T)]
        direction_window = max(
            2.0 * float(np.min(spacing)), float(np.median(sampled_radii))
        )
        tangents = estimate_graph_tangents(
            contracted, adjacency, spacing, direction_window
        )
        ridge_targets = transverse_edt_targets(
            original_edt, foreground, contracted, tangents, spacing
        )
        guide = 0.5 * contracted + 0.5 * ridge_targets
        def preserves_reference(candidate):
            return (
                _skeleton_state(candidate, topology_guidance)[2]
                == reference_signature
            )

        proposed, deleted = thin_foreground_sweep(
            working,
            guide,
            original_edt,
            spacing,
            validator=preserves_reference,
        )
        proposed_voxels, proposed_edges, proposed_signature = _skeleton_state(
            proposed, topology_guidance
        )
        if proposed_signature != reference_signature:
            raise RuntimeError('A validated thinning sweep changed the branch complex.')
        last_skeleton_voxels = proposed_voxels
        last_skeleton_edges = proposed_edges
        if deleted == 0:
            previous_voxels, previous_guide = voxels, guide
            if np.count_nonzero(working) == len(last_skeleton_voxels):
                outcome = 'converged'
                break
            stagnant_cycles += 1
            if stagnant_cycles >= 5:
                outcome = 'feature-limited'
                break
            print(
                f'Alternating cycle {cycle + 1}/{max_cycles}: no accepted '
                f'deletions ({stagnant_cycles}/5 stagnant cycles).'
            )
            continue
        working = proposed
        previous_voxels, previous_guide = voxels, guide
        stagnant_cycles = 0
        print(
            f'Alternating cycle {cycle + 1}/{max_cycles}: deleted {deleted} voxels; '
            f'{np.count_nonzero(working)} remain.'
        )
    else:
        warnings.warn(
            'Alternating skeletonisation reached its cycle budget; returning the '
            'last structure-preserving state.',
            RuntimeWarning,
            stacklevel=2,
        )

    if previous_guide is None:
        previous_guide = np.argwhere(working).astype(float)
    skeleton_voxels = last_skeleton_voxels
    skeleton_edges = last_skeleton_edges
    final_signature = branch_complex_signature(skeleton_voxels, skeleton_edges)
    if final_signature != reference_signature:
        raise RuntimeError(
            'Alternating skeletonisation changed the reference branch complex.'
        )
    coordinates, adjacency, paths, diagnostics = fit_branch_complex(
        reference_voxels,
        reference_edges,
        skeleton_voxels,
        skeleton_edges,
        foreground,
        original_edt,
        spacing,
        merge_tolerance,
    )
    output_mask, output_voxels = _rasterize_paths(paths, foreground)
    if foreground_topology(output_mask) != foreground_topology(foreground):
        warnings.warn(
            'Continuous fitting changed the rasterized foreground topology; using '
            'the last valid digital geometry.',
            RuntimeWarning,
            stacklevel=2,
        )
        discrete = skeleton_voxels.astype(float)
        coordinates, simplified_edges, paths = _simplify_paths(
            discrete, skeleton_edges, foreground, merge_tolerance
        )
        adjacency = _adjacency_from_edges(len(coordinates), simplified_edges)
        output_voxels = skeleton_voxels
        diagnostics['stable'] = False
    if not diagnostics['stable']:
        outcome = 'budget-limited'
    elif not diagnostics['ridge_satisfied']:
        outcome = 'feature-limited'
    print(
        f'Alternating skeletonisation {outcome}: {len(coordinates)} graph nodes, '
        f'{adjacency.nnz // 2} edges; mean/max ridge distance '
        f'{diagnostics["mean_ridge_distance"]:.4g}/'
        f'{diagnostics["max_ridge_distance"]:.4g} physical units.'
    )
    return coordinates, adjacency, output_voxels, paths
