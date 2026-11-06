"""Foreground-guided graph refinement with fixed 26/6 digital topology.

Sequential simple-point deletion preserves the foreground's topology. The
26-neighbour graph of the surviving voxels still has filled triangles; removing
those relations over GF(2) leaves exactly the foreground's independent tunnels,
with their correspondence to the mask intact (not just the same cycle count).

Simple-point criterion: Bertrand's T26(foreground) = T6(background) = 1;
T6 counts components in N18 that touch a face neighbour of the deleted voxel.
See https://www.dgtal.org/doc/2.0/moduleDigitalTopology.html.
"""

import hashlib
import heapq
from itertools import product

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu

from .contraction import laplacian_graph_contraction
from .graph import compute_sparse_adjacency_matrix
from .objects import UnionFind
from .utils import (
    _edge_voxels,
    _foreground_voxel,
    _nonincident_intersection_pairs,
    _nonincident_intersections,
    _valid_node_move,
)


_FOREGROUND = ndimage.generate_binary_structure(3, 3)
_BACKGROUND = ndimage.generate_binary_structure(3, 1)
_N18 = ndimage.generate_binary_structure(3, 2)
_FACES = _BACKGROUND.copy()
_FACES[1, 1, 1] = False
_OFFSETS = np.array([p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0)])


def foreground_topology(mask):
    """Return component, tunnel and cavity counts for closed foreground cubes."""
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(mask)
    components = ndimage.label(mask, _FOREGROUND)[1]
    cavities = ndimage.label(~np.pad(mask, 1), _BACKGROUND)[1] - 1
    if not len(coordinates):
        return 0, 0, 0

    def count_cells(offsets):
        points = np.concatenate([coordinates + offset for offset in offsets])
        return len(np.unique(points, axis=0))

    vertices = count_cells(list(product((0, 1), repeat=3)))
    edges = faces = 0
    for axis in range(3):
        # An edge varies along one axis; a face is perpendicular to one axis.
        edges += count_cells([p for p in product((0, 1), repeat=3) if p[axis] == 0])
        offset = np.zeros(3, dtype=int)
        offset[axis] = 1
        faces += count_cells([np.zeros(3, dtype=int), offset])
    euler = vertices - edges + faces - len(coordinates)
    return components, components + cavities - euler, cavities


def _is_simple(neighbourhood):
    """Test one deletion against the current, rather than a cached, neighbourhood."""
    foreground = neighbourhood.copy()
    foreground[1, 1, 1] = False
    if ndimage.label(foreground, _FOREGROUND)[1] != 1:
        return False
    background = ~neighbourhood & _N18
    background[1, 1, 1] = False
    labels, _ = ndimage.label(background, _BACKGROUND)
    touching = np.unique(labels[_FACES])
    return np.count_nonzero(touching) == 1


def thin_foreground_sweep(
    mask,
    guidance,
    reference_edt,
    spacing=(1.0, 1.0, 1.0),
    validator=None,
    validation_batch=64,
    protected=None,
):
    """Delete one frozen boundary layer using topology-safe sequential checks.

    Low-EDT boundary voxels are considered first. Within one EDT band, voxels
    farther from the current geometric guide are preferred. Newly exposed voxels
    wait for the next call, which makes contraction able to influence every layer.
    """
    foreground = np.asarray(mask, dtype=bool)
    guide = np.asarray(guidance, dtype=float)
    edt = np.asarray(reference_edt, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if foreground.ndim != 3 or edt.shape != foreground.shape:
        raise ValueError('mask and reference_edt must be matching 3D arrays.')
    protected = np.zeros_like(foreground) if protected is None else np.asarray(
        protected, dtype=bool
    )
    if protected.shape != foreground.shape or np.any(protected & ~foreground):
        raise ValueError('protected voxels must be contained in the working mask.')
    if guide.ndim != 2 or guide.shape[1] != 3 or not len(guide):
        raise ValueError('guidance must have shape (N, 3) with at least one point.')
    if spacing.shape != (3,) or np.any(spacing <= 0) or np.any(~np.isfinite(spacing)):
        raise ValueError('spacing must contain three positive finite values.')

    boundary = foreground & ~ndimage.binary_erosion(
        foreground, structure=_FOREGROUND, border_value=0
    )
    candidates = np.argwhere(boundary)
    if not len(candidates):
        return foreground.copy(), 0
    guide_distance = cKDTree(guide * spacing).query(candidates * spacing)[0]
    band_width = float(np.min(spacing))
    edt_bands = np.floor(edt[tuple(candidates.T)] / band_width + 1e-12)
    order = np.lexsort(
        (
            candidates[:, 2],
            candidates[:, 1],
            candidates[:, 0],
            -guide_distance,
            edt_bands,
        )
    )

    if validation_batch < 1:
        raise ValueError('validation_batch must be positive.')
    work = np.pad(foreground, 1)
    protected_work = np.pad(protected, 1)
    deleted = 0

    def try_delete(candidate):
        position = tuple(candidate)
        if not work[position] or protected_work[position]:
            return False
        patch = work[tuple(slice(value - 1, value + 2) for value in candidate)]
        # Retaining the last neighbour of a tip explicitly preserves terminal stubs.
        if np.count_nonzero(patch) <= 2 or not _is_simple(patch):
            return False
        work[position] = False
        return True

    checkpoint = work.copy()
    batch = []
    for candidate in candidates[order] + 1:
        if try_delete(candidate):
            batch.append(candidate)
        if len(batch) < validation_batch:
            continue
        if validator is not None and not validator(work[1:-1, 1:-1, 1:-1]):
            work = checkpoint
        else:
            deleted += len(batch)
        checkpoint = work.copy()
        batch = []
    if batch:
        if validator is not None and not validator(work[1:-1, 1:-1, 1:-1]):
            work = checkpoint
        else:
            deleted += len(batch)
    return work[1:-1, 1:-1, 1:-1], deleted


def _thin_foreground(mask, guidance, protected=None):
    """Peel simple voxels from the boundary inward, retaining curve endpoints."""
    work = np.pad(np.asarray(mask, dtype=bool), 1)
    protected_work = np.pad(
        np.zeros_like(mask, dtype=bool) if protected is None else protected, 1
    )
    if protected_work.shape != work.shape or np.any(protected_work & ~work):
        raise ValueError('protected voxels must be contained in the working mask.')
    coordinates = np.argwhere(work)
    distance = ndimage.distance_transform_edt(work)
    guide_distance = cKDTree(guidance).query(coordinates - 1)[0]
    priorities = {
        tuple(p): (float(distance[tuple(p)]), -float(np.round(d, 12)))
        for p, d in zip(coordinates, guide_distance)
    }
    queue = [(priorities[tuple(p)], tuple(p)) for p in coordinates]
    heapq.heapify(queue)
    queued = set(priorities)
    while queue:
        _, p = heapq.heappop(queue)
        queued.remove(p)
        if not work[p] or protected_work[p]:
            continue
        patch = work[tuple(slice(v - 1, v + 2) for v in p)]
        # A curve tip may be simple but deleting it shortens a branch.
        if np.count_nonzero(patch) <= 2 or not _is_simple(patch):
            continue
        work[p] = False
        for neighbour in np.asarray(p) + _OFFSETS:
            q = tuple(neighbour)
            if work[q] and q not in queued:
                heapq.heappush(queue, (priorities[q], q))
                queued.add(q)
    return np.argwhere(work) - 1


def _insert_binary_row(row, basis):
    """Insert an independent GF(2) row, represented by a Python integer."""
    while row:
        pivot = row.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = row
            return True
        row ^= basis[pivot]
    return False


def _tunnel_graph(coordinates):
    """Keep a forest plus independent cycles modulo filled voxel triangles.

    Pairwise touching axis-aligned voxel cubes have a common intersection.
    Consequently their nerve is the clique complex of the 26-neighbour graph.
    Its triangle boundaries generate all relations needed for first homology.
    Projection onto non-forest edges gives coordinates for the graph cycle space.
    Choosing a basis modulo these boundaries preserves each tunnel class.
    """
    adjacency = compute_sparse_adjacency_matrix(cKDTree(coordinates), 26)
    rows, cols = sparse.triu(adjacency, k=1).nonzero()
    edges = sorted(
        zip(rows.tolist(), cols.tolist()),
        key=lambda e: (np.sum((coordinates[e[0]] - coordinates[e[1]]) ** 2), e),
    )
    forest = UnionFind(len(coordinates))
    kept = []
    chords = {}
    for edge in edges:
        if forest.union(*edge)[0]:
            kept.append(edge)
        else:
            chords[edge] = len(chords)
    neighbours = [
        set(adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]])
        for i in range(len(coordinates))
    ]
    boundaries = {}
    for u, v in edges:
        for w in sorted(neighbours[u] & neighbours[v]):
            if w <= v:
                continue
            row = 0
            for edge in ((u, v), (u, w), (v, w)):
                if edge in chords:
                    row ^= 1 << chords[edge]
            _insert_binary_row(row, boundaries)
    tunnels = len(chords) - len(boundaries)
    for edge, index in chords.items():
        if _insert_binary_row(1 << index, boundaries):
            kept.append(edge)
    return np.asarray(kept, dtype=int).reshape(-1, 2), tunnels


def branch_complex_signature(coordinates, edges):
    """Describe terminal branches, junction attachments, and independent cycles.

    Degree-two sampling is removed from the description. This is deliberately
    stricter than component and Betti-number checks: losing a terminal stub changes
    both the endpoint count and the branch attachment signature.
    """
    n_vertices = len(coordinates)
    neighbours = [set() for _ in range(n_vertices)]
    for u, v in np.asarray(edges, dtype=int).reshape(-1, 2):
        neighbours[int(u)].add(int(v))
        neighbours[int(v)].add(int(u))
    degrees = np.array([len(items) for items in neighbours], dtype=int)
    critical = set(np.flatnonzero(degrees != 2).tolist())
    visited = set()
    attachments = []
    core_edges = []
    closed_cycles = 0

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    for start in sorted(critical):
        for neighbour in sorted(neighbours[start]):
            if edge_key(start, neighbour) in visited:
                continue
            visited.add(edge_key(start, neighbour))
            previous, current = start, neighbour
            while current not in critical:
                onward = sorted(neighbours[current] - {previous})
                if not onward:
                    break
                following = onward[0]
                visited.add(edge_key(current, following))
                previous, current = current, following
            attachments.append(
                tuple(sorted((int(degrees[start]), int(degrees[current]))))
            )
            core_edges.append((int(start), int(current)))

    for start in range(n_vertices):
        for neighbour in neighbours[start]:
            if edge_key(start, neighbour) in visited:
                continue
            closed_cycles += 1
            previous, current = start, neighbour
            visited.add(edge_key(start, neighbour))
            while current != start:
                onward = [item for item in neighbours[current] if item != previous]
                if not onward:
                    break
                following = onward[0]
                visited.add(edge_key(current, following))
                previous, current = current, following

    if n_vertices:
        rows = np.concatenate(
            ([u for u, v in edges], [v for u, v in edges])
        ) if len(edges) else np.array([], dtype=int)
        cols = np.concatenate(
            ([v for u, v in edges], [u for u, v in edges])
        ) if len(edges) else np.array([], dtype=int)
        adjacency = sparse.csr_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n_vertices, n_vertices),
        )
        components = sparse.csgraph.connected_components(adjacency, directed=False)[0]
    else:
        components = 0
    tunnels = len(edges) - n_vertices + components
    colors = {vertex: str(int(degrees[vertex])) for vertex in critical}
    multiplicity = {
        (u, v): sum(
            (start == u and end == v) or (start == v and end == u)
            for start, end in core_edges
        )
        for u in critical
        for v in critical
    }
    for _ in range(len(critical)):
        colors = {
            vertex: hashlib.sha256(
                repr(
                    (
                        colors[vertex],
                        tuple(
                            sorted(
                                (colors[neighbour], multiplicity[vertex, neighbour])
                                for neighbour in critical
                                if multiplicity[vertex, neighbour]
                            )
                        ),
                    )
                ).encode()
            ).hexdigest()
            for vertex in critical
        }
    return (
        int(components),
        int(tunnels),
        tuple(sorted(degrees[degrees != 2].tolist())),
        tuple(sorted(attachments)),
        int(closed_cycles),
        tuple(sorted(colors.values())),
    )


def _filled_box(points, mask):
    """Certify a shortcut's entire swept region lies in solid foreground.

    Testing the bounding box is conservative: it may reject a safe shortcut but
    never accepts one that could cut across a tunnel or leave the foreground.
    """
    lower = np.floor(np.min(points, axis=0) + 0.5).astype(int)
    upper = np.ceil(np.max(points, axis=0) - 0.5).astype(int)
    upper = np.maximum(upper, lower)
    if np.any(lower < 0) or np.any(upper >= mask.shape):
        return False
    return bool(np.all(mask[tuple(slice(a, b + 1) for a, b in zip(lower, upper))]))


def _path_error(points, start, end):
    """Measure deviation of every original path sample from a proposed segment."""
    direction = end - start
    squared_length = np.dot(direction, direction)
    if squared_length == 0:
        return float(np.max(np.linalg.norm(points - start, axis=1)))
    fraction = np.clip((points - start) @ direction / squared_length, 0.0, 1.0)
    return float(
        np.max(np.linalg.norm(points - start - fraction[:, None] * direction, axis=1))
    )


def _fit_frozen_graph(
    coordinates, edges, mask, *, voxel_spacing=(1., 1., 1.),
    smoothing_strength=1., max_sweeps=100,
):
    """Fit existing degree-two nodes with frozen weights and exact fixed nodes.

    Minimize lambda*||L_degree_two X||^2 + ||X-X_reference||^2 in physical
    space. Inverse-length affinities are normalized by the initial median edge
    length. Single-node swept triangles certify every accepted deformation.
    Returns voxel coordinates, unchanged edges, and numerical diagnostics.
    """
    reference_voxels = np.asarray(coordinates, dtype=float).copy()
    edges = np.asarray(edges, dtype=int).reshape(-1, 2).copy()
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(voxel_spacing, dtype=float)
    if (spacing.shape != (3,) or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)):
        raise ValueError('voxel_spacing must contain three positive finite values.')
    if not np.isfinite(smoothing_strength) or smoothing_strength < 0:
        raise ValueError('smoothing_strength must be finite and nonnegative.')
    if max_sweeps < 1:
        raise ValueError('max_sweeps must be positive.')
    if (reference_voxels.ndim != 2 or reference_voxels.shape[1] != 3
            or not np.all(np.isfinite(reference_voxels))):
        raise ValueError('Graph coordinates must be finite with shape (N, 3).')
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError('Fitting requires a nonempty 3D foreground mask.')
    if np.any(edges < 0) or np.any(edges >= len(reference_voxels)):
        raise ValueError('Graph edge index is out of bounds.')
    if len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise ValueError('Graph contains duplicate edges.')
    physical = reference_voxels * spacing
    lengths = np.linalg.norm(physical[edges[:, 0]] - physical[edges[:, 1]], axis=1)
    if np.any(lengths <= 1e-8 * np.min(spacing)):
        raise ValueError('Initial graph contains a degenerate edge.')
    for point in reference_voxels:
        _foreground_voxel(point, mask.shape, mask)
    for u, v in edges:
        _edge_voxels(reference_voxels[u], reference_voxels[v], mask.shape, mask)
    if _nonincident_intersections(reference_voxels, edges, spacing):
        raise ValueError('Initial graph contains an unintended intersection.')
    degree = np.bincount(edges.ravel(), minlength=len(reference_voxels))
    free = np.flatnonzero(degree == 2)
    fixed = np.flatnonzero(degree != 2)
    median_length = float(np.median(lengths)) if len(lengths) else 0.
    diagnostics = {
        'status': 'converged', 'lambda': float(smoothing_strength),
        'median_edge_length': median_length, 'sweeps': 0,
        'rejected_trials': 0, 'relative_residual': 0.,
        'initial_bending_energy': 0., 'bending_energy': 0.,
        'retention_energy': 0., 'max_displacement': 0., 'energy_history': [0.],
    }
    if not len(free):
        return reference_voxels, edges, diagnostics
    # Center and scale physical coordinates for translation and length-unit
    # invariance. This rescales the entire physical objective by one constant.
    origin = physical.mean(axis=0)
    reference = (physical - origin) / median_length
    weights = median_length / lengths
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.csr_matrix((np.tile(weights, 2), (rows, cols)),
                                  shape=(len(reference), len(reference)))
    laplacian = (
        sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    )[free]
    system = (sparse.eye(len(reference), format='csr')
              + smoothing_strength * (laplacian.T @ laplacian)).tocsr()
    right = reference[free] - system[free][:, fixed] @ reference[fixed]
    free_system = system[free][:, free].tocsc()
    scale = max(np.linalg.norm(right), np.linalg.norm(free_system @ reference[free]),
                np.finfo(float).eps)

    def residual(points):
        return float(np.linalg.norm(system[free] @ points - reference[free]) / scale)

    def energy(points):
        return float(smoothing_strength * np.sum((laplacian @ points) ** 2)
                     + np.sum((points - reference) ** 2))

    def voxel_points(points):
        output = (points * median_length + origin) / spacing
        output[fixed] = reference_voxels[fixed]
        return output

    current = reference.copy()
    initial_energy = energy(current)
    history = [initial_energy]
    diagnostics['initial_bending_energy'] = float(
        np.sum((laplacian @ current) ** 2) * median_length ** 2
    )
    if residual(current) > 1e-8:
        try:
            target = reference.copy()
            target[free] = splu(free_system).solve(right)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError('Frozen graph fitting linear solve failed.') from error
        if not np.all(np.isfinite(target)) or residual(target) > 1e-8:
            raise RuntimeError('Frozen graph fitting produced an inaccurate solution.')
        # Certify an explicit single-node route to the unconstrained optimum.
        route = reference_voxels.copy()
        target_voxels = voxel_points(target)
        valid = True
        for vertex in free:
            if np.array_equal(route[vertex], target_voxels[vertex]):
                continue
            if not _valid_node_move(
                route, edges, vertex, target_voxels[vertex], mask, spacing
            ):
                valid = False
                diagnostics['rejected_trials'] += 1
                break
            route[vertex] = target_voxels[vertex]
        if valid and energy(target) < initial_energy:
            current = target
            history.append(energy(current))
        else:
            tolerance = 1e-6 * float(np.min(spacing)) / median_length
            diagonal = system.diagonal()
            current_voxels = reference_voxels.copy()
            for sweep in range(max_sweeps):
                largest_move = 0.
                for vertex in free:
                    row = system.getrow(vertex)
                    gradient = (row @ current)[0] - reference[vertex]
                    displacement = -gradient / diagonal[vertex]
                    if not np.all(np.isfinite(displacement)):
                        raise RuntimeError(
                            'Frozen graph fitting produced a non-finite update.'
                        )
                    for halving in range(21):
                        delta = displacement * (0.5 ** halving)
                        change = (
                            2 * gradient @ delta + diagonal[vertex] * (delta @ delta)
                        )
                        if change >= 0:
                            break
                        proposal = (
                            (current[vertex] + delta) * median_length + origin
                        ) / spacing
                        if _valid_node_move(
                            current_voxels, edges, vertex, proposal, mask, spacing
                        ):
                            current[vertex] += delta
                            current_voxels[vertex] = proposal
                            largest_move = max(
                                largest_move, float(np.linalg.norm(delta))
                            )
                            break
                        diagnostics['rejected_trials'] += 1
                history.append(energy(current))
                diagnostics['sweeps'] = sweep + 1
                if residual(current) <= 1e-8:
                    break
                if largest_move < tolerance:
                    diagnostics['status'] = 'constraint-limited'
                    break
            else:
                diagnostics['status'] = 'iteration-limited'
    diagnostics['relative_residual'] = residual(current)
    # Small motion without constraints can be slow quadratic convergence.
    if (diagnostics['status'] == 'constraint-limited'
            and not diagnostics['rejected_trials']):
        diagnostics['status'] = 'iteration-limited'
    output = voxel_points(current)
    diagnostics['bending_energy'] = float(
        np.sum((laplacian @ current) ** 2) * median_length ** 2
    )
    diagnostics['retention_energy'] = float(
        np.sum((current - reference) ** 2) * median_length ** 2
    )
    diagnostics['max_displacement'] = float(np.max(np.linalg.norm(
        (output - reference_voxels) * spacing, axis=1
    )))
    diagnostics['energy_history'] = [value * median_length ** 2 for value in history]
    if _nonincident_intersections(output, edges, spacing):
        raise RuntimeError('Frozen graph fitting introduced an intersection.')
    return output, edges, diagnostics


def _edges_leaving_foreground(coordinates, edges, mask):
    """Flag edges whose straight segment leaves the closed foreground voxel cubes."""
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    start, end = coordinates[edges[:, 0]], coordinates[edges[:, 1]]
    # Fast certificate: every voxel of the box covering the segment is foreground.
    lower = np.ceil(np.minimum(start, end) - 0.5 - 1e-10).astype(int)
    upper = np.floor(np.maximum(start, end) + 0.5 + 1e-10).astype(int)
    small = np.all(upper - lower <= 2, axis=1)
    certified = small & np.all(lower >= 0, axis=1) & np.all(
        upper < np.asarray(mask.shape), axis=1
    )
    for offset in product(range(3), repeat=3):
        voxel = lower + np.asarray(offset)
        inside_box = np.all(voxel <= upper, axis=1) & certified
        clipped = np.clip(voxel, 0, np.asarray(mask.shape) - 1)
        certified &= ~inside_box | mask[tuple(clipped.T)]
    leaving = np.zeros(len(edges), dtype=bool)
    for index in np.flatnonzero(~certified):
        try:
            _edge_voxels(start[index], end[index], mask.shape, mask)
        except ValueError:
            leaving[index] = True
    return leaving


_PIN_WEIGHT = 1e3
# Retention weights of the second fit, tuned on synthetic trees F01-F12 (odd cases for
# tuning, even cases held out) against the generating centreline; see
# tests/test_output/flow_fitting_20261106/run_20261106_161705/RESULTS.md. Contraction
# keeps its own w_H and medial boost; every other flow setting is shared with it.
_FIT_W_H_BASE = 0.16
_FIT_W_H_MEDIAL = 3.0


def _repair_step(previous, proposed, edges, mask):
    """Return nodes to their previous positions until no edge leaves or crosses.

    Both ends of every offending edge go back, repeatedly. A valid previous state
    guarantees termination, since reverting everything restores it.
    """
    accepted = np.array(proposed, dtype=float)
    reverted = np.zeros(len(accepted), dtype=bool)
    check = np.arange(len(edges))
    while True:
        bad = check[_edges_leaving_foreground(accepted, edges[check], mask)]
        if not len(bad):
            crossing = {index for pair in _nonincident_intersection_pairs(
                accepted, edges, np.ones(3)) for index in pair}
            if not crossing:
                break
            bad = np.fromiter(crossing, dtype=int)
        nodes = np.unique(edges[bad])
        fresh = nodes[~reverted[nodes]]
        if not len(fresh):
            raise RuntimeError('Flow fitting could not restore a valid step.')
        accepted[fresh] = previous[fresh]
        reverted[fresh] = True
        # Only edges touching a reverted node changed since the last in-mask check.
        check = np.flatnonzero(np.any(np.isin(edges, fresh), axis=1))
    return accepted, int(reverted.sum())


def _flow_fit_graph(coordinates, edges, mask, flow_settings=None, pin_weight=_PIN_WEIGHT):
    """Smooth an extracted graph with the contraction flow, keeping its edges.

    Runs the Laplacian contraction on the graph with containment on and merging
    off. Endpoints and junctions keep their positions through a very large
    retention weight; the direct LU solver is always used because CG does not
    converge with weights this large. After each step, both ends of any edge leaving the
    foreground, or crossing another edge, return to their previous accepted positions,
    repeatedly, so every accepted step keeps all nodes and edges inside the
    foreground cubes and free of crossings.
    Returns voxel coordinates, unchanged edges, and diagnostics.
    """
    reference = np.asarray(coordinates, dtype=float).copy()
    edges = np.asarray(edges, dtype=int).reshape(-1, 2).copy()
    mask = np.asarray(mask, dtype=bool)
    settings = dict(flow_settings or {})
    forced = {'enforce_containment', 'min_edge_length', 'provisional',
              'anchor_scale', 'step_repair', 'binary_segmentation', 'solver'}
    if forced & set(settings):
        raise ValueError(f'flow_settings cannot set {sorted(forced & set(settings))}.')
    if not np.isfinite(pin_weight) or pin_weight < 1:
        raise ValueError('pin_weight must be finite and >= 1.')
    diagnostics = {'steps': 0, 'repaired_nodes': 0, 'max_pinned_drift': 0.,
                   'pin_weight': float(pin_weight)}
    degree = np.bincount(edges.ravel(), minlength=len(reference))
    if not len(edges) or not np.any(degree == 2):
        return reference, edges, diagnostics
    if _nonincident_intersections(reference, edges, np.ones(3)):
        raise ValueError('Initial graph contains an unintended intersection.')
    if np.any(_edges_leaving_foreground(reference, edges, mask)):
        raise ValueError('Initial graph leaves the foreground mask.')
    n_vertices = len(reference)
    adjacency = sparse.csr_matrix(
        (np.ones(2 * len(edges), dtype=bool),
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n_vertices, n_vertices),
    )
    anchor_scale = np.where(degree == 2, 1., float(pin_weight))

    def repair(previous, proposed):
        accepted, count = _repair_step(previous, proposed, edges, mask)
        diagnostics['steps'] += 1
        diagnostics['repaired_nodes'] += count
        return accepted, count

    output, output_adjacency = laplacian_graph_contraction(
        reference, adjacency, binary_segmentation=mask, enforce_containment=True,
        min_edge_length=0., provisional=True, anchor_scale=anchor_scale,
        step_repair=repair, solver='LU', **settings,
    )
    if len(output) != n_vertices or (output_adjacency != adjacency).nnz:
        raise RuntimeError('Flow fitting changed the graph structure.')
    if np.any(_edges_leaving_foreground(output, edges, mask)):
        raise RuntimeError('Flow fitting left the foreground mask.')
    if _nonincident_intersections(output, edges, np.ones(3)):
        raise RuntimeError('Flow fitting introduced an intersection.')
    pinned = degree != 2
    if np.any(pinned):
        diagnostics['max_pinned_drift'] = float(
            np.max(np.linalg.norm(output[pinned] - reference[pinned], axis=1))
        )
    return output, edges, diagnostics


def _simplify_paths(coordinates, edges, mask, tolerance):
    """Remove degree-two vertices; mask=None skips foreground shortcut checks."""
    neighbours = [set() for _ in coordinates]
    paths = {}
    for u, v in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
        key = tuple(sorted((u, v)))
        paths[key] = coordinates[list(key)].copy()
    active = np.ones(len(coordinates), dtype=bool)
    queue = list(range(len(coordinates))) if tolerance > 0 else []
    heapq.heapify(queue)
    queued = set(queue)
    while queue:
        vertex = heapq.heappop(queue)
        queued.remove(vertex)
        if not active[vertex] or len(neighbours[vertex]) != 2:
            continue
        u, v = sorted(neighbours[vertex])
        # Adding an already existing edge would destroy a real cycle.
        if v in neighbours[u]:
            continue
        first, second = tuple(sorted((u, vertex))), tuple(sorted((v, vertex)))
        incoming = paths[first] if u < vertex else paths[first][::-1]
        outgoing = paths[second] if vertex < v else paths[second][::-1]
        samples = np.vstack((incoming, outgoing[1:]))
        if _path_error(
            samples, coordinates[u], coordinates[v]
        ) > tolerance or (mask is not None and not _filled_box(samples, mask)):
            continue
        neighbours[u].remove(vertex)
        neighbours[v].remove(vertex)
        neighbours[u].add(v)
        neighbours[v].add(u)
        neighbours[vertex].clear()
        active[vertex] = False
        del paths[first], paths[second]
        paths[(u, v)] = samples
        for neighbour in (u, v):
            if neighbour not in queued:
                heapq.heappush(queue, neighbour)
                queued.add(neighbour)
    remap = np.cumsum(active) - 1
    edges = np.asarray(
        [(remap[u], remap[v]) for u, v in sorted(paths)], dtype=int
    ).reshape(-1, 2)
    paths = {(int(remap[u]), int(remap[v])): path for (u, v), path in paths.items()}
    return coordinates[active], edges, paths


def refine_graph(
    mask,
    contracted_coordinates,
    merge_tolerance=0.25,
    w_H_medial=1.0,
    return_paths=False,
    *,
    voxel_spacing=(1., 1., 1.),
    second_fitting=True,
    flow_settings=None,
):
    """Extract a graph with every foreground tunnel and optionally fit its geometry.

    Parameters
    ----------
    mask : (D, H, W) array
        Original foreground, interpreted with fixed 26-connectivity.
    contracted_coordinates : (N, 3) array
        Laplacian contraction result in the mask's voxel coordinate system.
        Guides which medial voxels survive; never determines tunnel removal.
    merge_tolerance : float, optional
        Retained for call compatibility and still validated; refinement no
        longer merges degree-two nodes, so this value has no effect here.
    w_H_medial : float, optional
        Retained for call compatibility; still validated. The fit takes its
        medial boost from flow_settings instead. Default is 1.
    return_paths : bool, optional
        Also return reference voxels and the retained edge polylines for export.
    voxel_spacing : tuple of float, optional
        Retained for call compatibility; the flow fit works in voxel units, as
        contraction does. Returned coordinates are in voxel units.
    second_fitting : bool, optional
        Smooth degree-two node positions with the contraction flow; endpoints
        and junctions are held by a very large retention weight. Every extracted
        skeleton node is kept either way, joined by straight edges; without
        fitting, nodes stay at their voxel centres. Default is True.
    flow_settings : dict, optional
        Contraction keyword arguments for the fit, e.g. w_L, w_H_base,
        w_H_medial, beta_edt, use_edt, use_anisotropic, tol, tol_mean_drift,
        max_iter and local_pca_hops. Containment is always on, merging always
        off and the solver always LU. Unset values use the contraction defaults.

    Returns
    -------
    coordinates : (M, 3) array
        Refined graph coordinates in voxel units.
    adjacency : sparse.csr_matrix
        Symmetric adjacency with no redundant independent cycles.
    reference_voxels : (K, 3) array, optional
        Topology-preserving digital skeleton, returned when return_paths is True.
    edge_paths : dict, optional
        Straight edge paths keyed by sorted node-index pairs, returned when
        return_paths is True; each path holds only its two end nodes.
    """
    if not np.isfinite(merge_tolerance) or merge_tolerance < 0:
        raise ValueError('merge_tolerance must be finite and >= 0.')
    if not np.isfinite(w_H_medial) or w_H_medial < 1:
        raise ValueError('w_H_medial must be finite and >= 1.')
    mask = np.asarray(mask, dtype=bool)
    guidance = np.asarray(contracted_coordinates, dtype=float)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError('Refinement requires a nonempty 3D foreground mask.')
    if (
        guidance.ndim != 2
        or guidance.shape[1] != 3
        or not len(guidance)
        or not np.all(np.isfinite(guidance))
    ):
        raise ValueError(
            'Contracted coordinates must be a finite nonempty (N, 3) array.'
        )
    # A curve graph cannot represent an enclosed three-dimensional cavity.
    expected_components, expected_tunnels, cavities = foreground_topology(mask)
    if cavities:
        raise ValueError(
            'Foreground contains enclosed cavities; a curve graph cannot preserve their topology.'
        )
    coordinates = _thin_foreground(mask, guidance).astype(float)
    reference_voxels = coordinates.astype(int)
    edges, tunnels = _tunnel_graph(coordinates)
    if second_fitting:
        # Fitting moves node positions only; it never changes the edge set.
        coordinates, edges, diagnostics = _flow_fit_graph(
            coordinates, edges, mask, flow_settings
        )
        print(
            f"Flow fit: {diagnostics['steps']} steps; "
            f"{diagnostics['repaired_nodes']} node moves repaired; "
            f"pin weight {diagnostics['pin_weight']:g}, max endpoint/junction "
            f"drift {diagnostics['max_pinned_drift']:.3g} voxels."
        )
    else:
        print('Second fitting disabled; keeping voxel-centre node positions.')
    # Keep every extracted skeleton node; each edge is its own straight path.
    edges = np.asarray(sorted(map(tuple, edges.tolist())), dtype=int).reshape(-1, 2)
    paths = {(int(u), int(v)): coordinates[[u, v]] for u, v in edges}
    for (u, v), path in paths.items():
        if (not np.array_equal(path[0], coordinates[u])
                or not np.array_equal(path[-1], coordinates[v])):
            raise RuntimeError('Refinement edge path endpoints are inconsistent.')
        for start, end in zip(path[:-1], path[1:]):
            _edge_voxels(start, end, mask.shape, mask)
    if len(edges):
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))
    else:
        rows = cols = np.array([], dtype=int)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(coordinates), len(coordinates)),
    )
    components = sparse.csgraph.connected_components(adjacency, directed=False)[0]
    if (
        components != expected_components
        or tunnels != expected_tunnels
        or len(edges) - len(coordinates) + components != expected_tunnels
    ):
        raise RuntimeError('Refinement failed its foreground topology checks.')
    print(
        f'Refined graph: {len(coordinates)} nodes, {len(edges)} edges, '
        f'{components} components, {tunnels} independent tunnels preserved.'
    )
    if return_paths:
        return coordinates, adjacency, reference_voxels, paths
    return coordinates, adjacency
