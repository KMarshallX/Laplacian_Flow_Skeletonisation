"""Flux-guided triangle decimation for the default contraction workflow."""

import numpy as np
from scipy import ndimage, sparse


_POSITION_TOLERANCE = 1e-6
_FLUX_TOLERANCE = 1e-6
_FLUX_BATCH_SIZE = 256


def _sphere_directions():
    """Return 64 deterministic, approximately uniform, antipodal directions."""
    count = 32
    indices = np.arange(count, dtype=float)
    z = (indices + 0.5) / count
    angle = indices * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    hemisphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return np.vstack((hemisphere, -hemisphere))


SPHERE_DIRECTIONS = _sphere_directions()


class DistanceFluxSampler:
    """Cache a signed physical distance-gradient field for flux queries."""

    def __init__(self, foreground, spacing):
        foreground = np.asarray(foreground, dtype=bool)
        spacing = np.asarray(spacing, dtype=float)
        if foreground.ndim != 3 or not np.any(foreground):
            raise ValueError('triangle decimation requires a nonempty 3D segmentation.')
        if (
            spacing.shape != (3,)
            or np.any(spacing <= 0)
            or np.any(~np.isfinite(spacing))
        ):
            raise ValueError('voxel_spacing must contain three positive finite values.')
        self.foreground = foreground
        self.spacing = spacing
        self.radius = float(np.min(spacing))
        base = np.ceil(self.radius / spacing).astype(int) + 2
        self._before = base.copy()
        self._after = base.copy()
        self._gradient = None
        self._rebuild()

    def _rebuild(self):
        padding = tuple(
            (int(before), int(after))
            for before, after in zip(self._before, self._after)
        )
        padded = np.pad(
            self.foreground, padding, mode='constant', constant_values=False
        )
        inside = ndimage.distance_transform_edt(padded, sampling=self.spacing)
        outside = ndimage.distance_transform_edt(~padded, sampling=self.spacing)
        signed_distance = inside - outside
        self._gradient = np.asarray(
            np.gradient(signed_distance, *self.spacing, edge_order=2)
        )

    def _ensure_bounds(self, sample_voxels):
        minima = np.min(sample_voxels, axis=0)
        maxima = np.max(sample_voxels, axis=0)
        required_before = np.maximum(0, np.ceil(1.0 - minima).astype(int))
        required_after = np.maximum(
            0,
            np.ceil(maxima - np.asarray(self.foreground.shape) + 2.0).astype(int),
        )
        before = np.maximum(self._before, required_before)
        after = np.maximum(self._after, required_after)
        if np.array_equal(before, self._before) and np.array_equal(after, self._after):
            return
        self._before = before
        self._after = after
        self._rebuild()

    def score(self, physical_positions):
        """Estimate average outward flux at physical-coordinate positions."""
        positions = np.asarray(physical_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('physical_positions must have shape (N, 3).')
        if len(positions) == 0:
            return np.empty(0, dtype=float)

        scores = np.empty(len(positions), dtype=float)
        offsets = self.radius * SPHERE_DIRECTIONS
        for start in range(0, len(positions), _FLUX_BATCH_SIZE):
            stop = min(start + _FLUX_BATCH_SIZE, len(positions))
            samples = positions[start:stop, None, :] + offsets[None, :, :]
            sample_voxels = samples.reshape(-1, 3) / self.spacing
            self._ensure_bounds(sample_voxels)
            padded_voxels = sample_voxels + self._before
            values = np.column_stack(
                [
                    ndimage.map_coordinates(
                        component,
                        padded_voxels.T,
                        order=1,
                        mode='constant',
                        cval=np.nan,
                        prefilter=False,
                    )
                    for component in self._gradient
                ]
            ).reshape(stop - start, len(SPHERE_DIRECTIONS), 3)
            if not np.all(np.isfinite(values)):
                raise RuntimeError('Flux sampling exceeded the padded distance field.')
            scores[start:stop] = np.mean(
                np.sum(values * SPHERE_DIRECTIONS[None, :, :], axis=2), axis=1
            )
        return scores


def enumerate_triangles(adjacency_matrix):
    """Return each undirected graph triangle once as an ordered integer triple."""
    adjacency = adjacency_matrix.astype(bool).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    triangles = []
    for u in range(adjacency.shape[0]):
        u_neighbours = adjacency.indices[
            adjacency.indptr[u] : adjacency.indptr[u + 1]
        ]
        for v in u_neighbours[u_neighbours > u]:
            v_neighbours = adjacency.indices[
                adjacency.indptr[v] : adjacency.indptr[v + 1]
            ]
            common = np.intersect1d(u_neighbours, v_neighbours, assume_unique=True)
            triangles.extend((u, int(v), int(w)) for w in common[common > v])
    return triangles


def _circumcentre(points):
    """Return the circumcentre of three non-collinear physical points."""
    a, b, c = points
    normal = np.cross(b - a, c - a)
    system = np.vstack((2.0 * (b - a), 2.0 * (c - a), normal))
    target = np.array(
        [
            np.dot(b, b) - np.dot(a, a),
            np.dot(c, c) - np.dot(a, a),
            np.dot(normal, points.mean(axis=0)),
        ]
    )
    return np.linalg.solve(system, target)


def _compact_target(points, flux_sampler):
    centroid = points.mean(axis=0)
    candidates = np.vstack((points, centroid, _circumcentre(points)))
    scores = flux_sampler.score(candidates)
    tied = np.flatnonzero(scores <= np.min(scores) + _FLUX_TOLERANCE)
    distances = np.linalg.norm(candidates[tied] - centroid, axis=1)
    nearest = tied[distances <= np.min(distances) + _POSITION_TOLERANCE]
    if len(nearest) == 1:
        return candidates[nearest[0]]
    order = np.lexsort(
        (candidates[nearest, 2], candidates[nearest, 1], candidates[nearest, 0])
    )
    return candidates[nearest[order[0]]]


def _cluster_projected(points, tolerance):
    """Cluster projected triangle points without exceeding a complete-group diameter."""
    groups = [[index] for index in range(len(points))]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    pairs = sorted(
        ((distances[u, v], u, v) for u in range(3) for v in range(u + 1, 3)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for _, u, v in pairs:
        group_u = next(group for group in groups if u in group)
        group_v = next(group for group in groups if v in group)
        if group_u is group_v:
            continue
        combined = group_u + group_v
        combined_points = points[combined]
        diameter = np.max(
            np.linalg.norm(
                combined_points[:, None, :] - combined_points[None, :, :], axis=2
            )
        )
        if diameter <= tolerance:
            group_u.extend(group_v)
            groups.remove(group_v)
    return groups


def triangle_collapse_decimation(
    X,
    adjacency_matrix,
    binary_segmentation,
    voxel_spacing,
    elongation_ratio=25.0,
    medialness=None,
    flux_sampler=None,
):
    """Collapse compact graph triangles and flatten elongated ones into chains."""
    coordinates = np.asarray(X, dtype=float)
    spacing = np.asarray(voxel_spacing, dtype=float)
    if not np.isfinite(elongation_ratio) or elongation_ratio <= 1.0:
        raise ValueError('dev_elongation_ratio must be finite and greater than 1.')
    if (
        spacing.shape != (3,)
        or np.any(spacing <= 0)
        or np.any(~np.isfinite(spacing))
    ):
        raise ValueError('voxel_spacing must contain three positive finite values.')
    if flux_sampler is None:
        flux_sampler = DistanceFluxSampler(binary_segmentation, spacing)

    adjacency = adjacency_matrix.astype(bool).tocsr()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    triangles = enumerate_triangles(adjacency)
    physical = coordinates * spacing
    face_areas = (
        spacing[0] * spacing[1],
        spacing[0] * spacing[2],
        spacing[1] * spacing[2],
    )
    area_limit = float(np.min(face_areas))
    candidates = []
    for triangle in triangles:
        points = physical[np.asarray(triangle)]
        area = 0.5 * np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
        if area < area_limit:
            candidates.append((area, triangle))
    candidates.sort(key=lambda item: (item[0], item[1]))

    tolerance = _POSITION_TOLERANCE * float(np.min(spacing))
    touched = set()
    operations = []
    compact_count = 0
    elongated_count = 0
    for _, triangle in candidates:
        if touched.intersection(triangle):
            continue
        indices = np.asarray(triangle)
        points = physical[indices]
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        centroid = points.mean(axis=0)

        if np.max(pairwise) <= tolerance:
            operations.append(('compact', triangle, [[0, 1, 2]], [centroid], []))
            compact_count += 1
            touched.update(triangle)
            continue

        centred = points - centroid
        covariance = centred.T.dot(centred) / 3.0
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, -1]
        second = eigenvectors[:, -2]
        axial = centred.dot(tangent)
        residual = centred - axial[:, None] * tangent
        collinear = np.max(np.linalg.norm(residual, axis=1)) <= tolerance
        ratio = np.inf if eigenvalues[-2] <= 0 else eigenvalues[-1] / eigenvalues[-2]

        if not collinear and ratio <= elongation_ratio:
            target = _compact_target(points, flux_sampler)
            operations.append(('compact', triangle, [[0, 1, 2]], [target], []))
            compact_count += 1
        else:
            projected = (
                points
                if collinear
                else points - centred.dot(second)[:, None] * second
            )
            groups = _cluster_projected(projected, tolerance)
            group_points = [projected[group].mean(axis=0) for group in groups]
            longitudinal = [np.mean(axial[group]) for group in groups]
            chain_order = np.argsort(longitudinal, kind='stable')
            chain = [int(index) for index in chain_order]
            operations.append(('elongated', triangle, groups, group_points, chain))
            elongated_count += 1
        touched.update(triangle)

    if not operations:
        print(
            f'Triangle pass: detected={len(triangles)}, eligible={len(candidates)}, '
            f'compact=0, elongated=0, nodes={len(coordinates)}, '
            f'edges={adjacency.nnz // 2}'
        )
        copied_medialness = None if medialness is None else medialness.copy()
        return coordinates.copy(), adjacency, copied_medialness, False

    labels = np.arange(len(coordinates))
    positions = {}
    removed_internal = set()
    added_edges = []
    for _, triangle, groups, group_points, chain in operations:
        group_labels = []
        for local_group, point in zip(groups, group_points):
            members = [triangle[index] for index in local_group]
            label = min(members)
            labels[members] = label
            positions[label] = np.asarray(point) / spacing
            group_labels.append(label)
        for u_index in range(3):
            for v_index in range(u_index + 1, 3):
                removed_internal.add(
                    tuple(sorted((triangle[u_index], triangle[v_index])))
                )
        for first, second_index in zip(chain[:-1], chain[1:]):
            added_edges.append((group_labels[first], group_labels[second_index]))

    rows, cols = sparse.triu(adjacency, k=1).nonzero()
    rebuilt_edges = set()
    for u, v in zip(rows, cols):
        if (int(u), int(v)) in removed_internal:
            continue
        new_u, new_v = int(labels[u]), int(labels[v])
        if new_u != new_v:
            rebuilt_edges.add(tuple(sorted((new_u, new_v))))
    for u, v in added_edges:
        if u != v:
            rebuilt_edges.add(tuple(sorted((int(u), int(v)))))

    unique_labels = np.unique(labels)
    label_to_index = {int(label): index for index, label in enumerate(unique_labels)}
    new_X = np.vstack(
        [positions.get(int(label), coordinates[label]) for label in unique_labels]
    )
    if rebuilt_edges:
        new_rows = []
        new_cols = []
        for u, v in sorted(rebuilt_edges):
            mapped_u, mapped_v = label_to_index[u], label_to_index[v]
            new_rows.extend((mapped_u, mapped_v))
            new_cols.extend((mapped_v, mapped_u))
        new_adj = sparse.csr_matrix(
            (np.ones(len(new_rows), dtype=bool), (new_rows, new_cols)),
            shape=(len(unique_labels), len(unique_labels)),
        )
    else:
        new_adj = sparse.csr_matrix(
            (len(unique_labels), len(unique_labels)), dtype=bool
        )

    new_medialness = None
    if medialness is not None:
        new_medialness = np.zeros(len(unique_labels), dtype=float)
        remapped = np.asarray([label_to_index[int(label)] for label in labels])
        np.maximum.at(new_medialness, remapped, medialness)

    print(
        f'Triangle pass: detected={len(triangles)}, eligible={len(candidates)}, '
        f'compact={compact_count}, elongated={elongated_count}, '
        f'nodes={len(new_X)}, edges={new_adj.nnz // 2}'
    )
    return new_X, new_adj, new_medialness, True
