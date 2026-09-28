"""Frozen contraction objectives and physical distance-flux derivatives."""

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import LinearOperator, cg, splu

from .graph import _compute_n_hop_adjacency

_FLUX_BATCH_SIZE = 256
_MAX_INNER_UPDATES = 20
_MAX_BACKTRACKS = 20
_SUFFICIENT_DECREASE = 1e-4
_CORRECTION_TOL = 1e-3
_ENERGY_TOL = 1e-6


def validate_segmentation(segmentation):
    """Require a finite, nonempty 3D Boolean or numeric foreground mask."""
    mask = np.asarray(segmentation)
    if mask.ndim != 3 or mask.dtype.kind not in 'buifc':
        raise ValueError('binary_segmentation must be a nonempty 3D numeric mask.')
    if not np.all(np.isfinite(mask)):
        raise ValueError('binary_segmentation must contain only finite values.')
    foreground = mask != 0
    if not np.any(foreground):
        raise ValueError('binary_segmentation must be a nonempty 3D numeric mask.')
    return foreground


def validate_spacing(spacing):
    """Require three positive finite physical voxel spacings in millimetres."""
    spacing = np.asarray(spacing, dtype=float)
    if (
        spacing.shape != (3,)
        or np.any(spacing <= 0)
        or not np.all(np.isfinite(spacing))
    ):
        raise ValueError('voxel_spacing must contain three positive finite values.')
    return spacing


def _sphere_directions():
    """Return the reference sampler's 64 deterministic antipodal directions."""
    indices = np.arange(32, dtype=float)
    z = (indices + 0.5) / 32
    angle = indices * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    hemisphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return np.vstack((hemisphere, -hemisphere))


SPHERE_DIRECTIONS = _sphere_directions()


class DistanceFluxSampler:
    """Sample signed-distance flux and differentiate its trilinear interpolant."""

    def __init__(self, foreground, spacing):
        """Cache the padded distance-gradient field in physical coordinates."""
        self.foreground = validate_segmentation(foreground)
        self.spacing = validate_spacing(spacing)
        self.radius = float(np.min(self.spacing))
        base = np.ceil(self.radius / self.spacing).astype(int) + 2
        self._before = base.copy()
        self._after = base.copy()
        self._rebuild()

    def _rebuild(self):
        padding = tuple(
            (int(before), int(after))
            for before, after in zip(self._before, self._after)
        )
        padded = np.pad(self.foreground, padding, constant_values=False)
        inside = ndimage.distance_transform_edt(padded, sampling=self.spacing)
        outside = ndimage.distance_transform_edt(~padded, sampling=self.spacing)
        self._gradient = np.asarray(
            np.gradient(inside - outside, *self.spacing, edge_order=2)
        )

    def _ensure_bounds(self, sample_voxels):
        minima = np.min(sample_voxels, axis=0)
        maxima = np.max(sample_voxels, axis=0)
        before = np.maximum(
            self._before, np.maximum(0, np.ceil(1 - minima)).astype(int)
        )
        after = np.maximum(
            self._after,
            np.maximum(
                0, np.ceil(maxima - np.asarray(self.foreground.shape) + 2)
            ).astype(int),
        )
        if np.array_equal(before, self._before) and np.array_equal(after, self._after):
            return
        self._before, self._after = before, after
        self._rebuild()

    def _interpolate(self, physical_samples, normals, derivative):
        """Interpolate flux contributions and optionally their physical slopes."""
        voxels = physical_samples / self.spacing
        self._ensure_bounds(voxels)
        padded = voxels + self._before
        # floor selects the positive-side cell at an exact grid boundary.
        lower = np.floor(padded).astype(np.intp)
        fraction = padded - lower
        values = np.zeros(len(voxels))
        slopes = np.zeros((len(voxels), 3)) if derivative else None
        for corner in product((0, 1), repeat=3):
            index = lower + corner
            field = self._gradient[:, index[:, 0], index[:, 1], index[:, 2]].T
            contribution = np.sum(field * normals, axis=1)
            factors = np.where(np.asarray(corner), fraction, 1 - fraction)
            values += np.prod(factors, axis=1) * contribution
            if derivative:
                for axis in range(3):
                    other_axes = [other for other in range(3) if other != axis]
                    weight = np.prod(factors[:, other_axes], axis=1)
                    sign = 1.0 if corner[axis] else -1.0
                    slopes[:, axis] += sign * weight * contribution / self.spacing[axis]
        return values, slopes

    def _evaluate(self, physical_positions, derivative):
        positions = np.asarray(physical_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('physical_positions must have shape (N, 3).')
        if not np.all(np.isfinite(positions)):
            raise ValueError('Flux sampling requires finite positions.')
        scores = np.empty(len(positions))
        gradients = np.empty_like(positions) if derivative else None
        directions = SPHERE_DIRECTIONS
        count = len(directions)
        for start in range(0, len(positions), _FLUX_BATCH_SIZE):
            stop = min(start + _FLUX_BATCH_SIZE, len(positions))
            samples = positions[start:stop, None, :] + self.radius * directions
            normals = np.tile(directions, (stop - start, 1))
            values, slopes = self._interpolate(
                samples.reshape(-1, 3), normals, derivative
            )
            scores[start:stop] = values.reshape(-1, count).mean(axis=1)
            if derivative:
                gradients[start:stop] = slopes.reshape(-1, count, 3).mean(axis=1)
        return scores, gradients

    def score(self, physical_positions):
        """Return average outward flux around physical-coordinate positions."""
        return self._evaluate(physical_positions, False)[0]

    def score_and_gradient(self, physical_positions):
        """Return flux and its analytic gradient with respect to sphere centres."""
        return self._evaluate(physical_positions, True)


def physical_tangents(coordinates, adjacency, spacing, local_pca_hops=1):
    """Estimate physical PCA tangents and confidence with terminal-edge overrides."""
    physical = (coordinates - coordinates.mean(axis=0)) * spacing
    neighbours = _compute_n_hop_adjacency(adjacency, local_pca_hops)
    rows, cols = neighbours.nonzero()
    count = len(coordinates)
    degrees = np.bincount(rows, minlength=count)
    means = neighbours.dot(physical) / np.maximum(degrees[:, None], 1)
    deviations = physical[cols] - means[rows]
    covariance = np.zeros((count, 3, 3))
    for first in range(3):
        for second in range(first, 3):
            values = np.bincount(
                rows,
                weights=deviations[:, first] * deviations[:, second],
                minlength=count,
            )
            covariance[:, first, second] = values
            covariance[:, second, first] = values
    covariance /= np.maximum(degrees - 1, 1)[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    principal = eigenvalues[:, -1]
    confidence = np.zeros(count)
    valid = principal > 0
    confidence[valid] = (principal[valid] - eigenvalues[valid, -2]) / principal[valid]
    confidence = np.clip(confidence, 0, 1)
    tangents = eigenvectors[:, :, -1].copy()
    tangents[~valid] = 0

    immediate = adjacency.astype(bool).tocsr().copy()
    immediate.setdiag(False)
    immediate.eliminate_zeros()
    edge_rows, edge_cols = immediate.nonzero()
    immediate_degrees = np.bincount(edge_rows, minlength=count)
    terminal_edges = immediate_degrees[edge_rows] == 1
    terminals = edge_rows[terminal_edges]
    edges = physical[edge_cols[terminal_edges]] - physical[terminals]
    lengths = np.linalg.norm(edges, axis=1)
    tangents[terminals] = 0
    confidence[terminals] = 0
    nonzero = lengths > 0
    tangents[terminals[nonzero]] = edges[nonzero] / lengths[nonzero, None]
    confidence[terminals[nonzero]] = 1
    tangents[immediate_degrees == 0] = 0
    confidence[immediate_degrees == 0] = 0
    return tangents, confidence


class FrozenObjective:
    """Evaluate a frozen objective using displacements from its voxel anchor."""

    def __init__(
        self,
        anchor,
        laplacian,
        retention,
        w_L,
        spacing,
        tangents,
        confidence,
        lambda_f,
        lambda_parallel,
        flux_sampler=None,
    ):
        """Freeze graph operators, physical directions and positional retention."""
        self.anchor = anchor.copy()
        self.laplacian = laplacian.tocsr()
        self.retention_sq = np.asarray(retention, dtype=float) ** 2
        self.w_L_sq = w_L**2
        self.spacing = spacing
        self.axial = tangents * spacing
        self.axial_weight = lambda_parallel * confidence
        self.lambda_f = lambda_f
        self.flux_sampler = flux_sampler
        centred = anchor - anchor.mean(axis=0)
        self.anchor_laplacian = self.laplacian.dot(centred)
        size = anchor.size
        self.hessian = LinearOperator((size, size), matvec=self._hessian_product)

    def _hessian_product(self, vector):
        displacement = np.asarray(vector).reshape(-1, 3)
        result = self.w_L_sq * self.laplacian.T.dot(self.laplacian.dot(displacement))
        result += self.retention_sq[:, None] * displacement
        longitudinal = np.sum(displacement * self.axial, axis=1)
        result += (self.axial_weight * longitudinal)[:, None] * self.axial
        return (2 * result).ravel()

    def quadratic(self, displacement):
        """Return the quadratic energy and gradient in voxel displacement units."""
        contracted = self.anchor_laplacian + self.laplacian.dot(displacement)
        longitudinal = np.sum(displacement * self.axial, axis=1)
        value = self.w_L_sq * np.sum(contracted**2)
        value += np.sum(self.retention_sq[:, None] * displacement**2)
        value += np.sum(self.axial_weight * longitudinal**2)
        gradient = self.w_L_sq * self.laplacian.T.dot(contracted)
        gradient += self.retention_sq[:, None] * displacement
        gradient += (self.axial_weight * longitudinal)[:, None] * self.axial
        return float(value), 2 * gradient

    def evaluate(self, displacement, derivative=True):
        """Return energy and optionally its voxel-coordinate gradient."""
        value, gradient = self.quadratic(displacement)
        if self.lambda_f:
            physical = (self.anchor + displacement) * self.spacing
            if derivative:
                flux, slope = self.flux_sampler.score_and_gradient(physical)
                gradient += self.lambda_f * slope * self.spacing
            else:
                flux = self.flux_sampler.score(physical)
            value += self.lambda_f * np.sum(flux)
        return float(value), gradient if derivative else None

    def diagonal_blocks(self):
        """Return the Hessian's nodewise blocks for preconditioning."""
        diagonal = (
            self.w_L_sq
            * np.asarray(self.laplacian.multiply(self.laplacian).sum(axis=0)).ravel()
            + self.retention_sq
        )
        blocks = diagonal[:, None, None] * np.eye(3)
        blocks += self.axial_weight[:, None, None] * (
            self.axial[:, :, None] * self.axial[:, None, :]
        )
        return 2 * blocks

    def sparse_hessian(self):
        """Assemble the coupled Hessian sparsely for LU or AMG setup."""
        base = self.w_L_sq * (self.laplacian.T @ self.laplacian)
        base += sparse.diags(self.retention_sq, format='csr')
        blocks = self.axial_weight[:, None, None] * (
            self.axial[:, :, None] * self.axial[:, None, :]
        )
        indices = np.arange(len(blocks))
        axial = sparse.bsr_matrix((blocks, indices, np.arange(len(blocks) + 1)))
        return (2 * (sparse.kron(base, sparse.eye(3), format='csr') + axial)).tocsr()


class _QuadraticSolver:
    """Reuse a frozen Hessian's factorisation or preconditioner across inner steps."""

    def __init__(self, objective, solver):
        self.operator = objective.hessian
        self.factorisation = None
        self.error = None
        inverse_blocks = np.linalg.inv(objective.diagonal_blocks())
        self.preconditioner = LinearOperator(
            self.operator.shape,
            matvec=lambda vector: np.einsum(
                'nij,nj->ni', inverse_blocks, vector.reshape(-1, 3)
            ).ravel(),
        )
        if solver == 'CG':
            return
        matrix = objective.sparse_hessian()
        if solver == 'LU':
            try:
                self.factorisation = splu(matrix.tocsc())
            except (RuntimeError, ValueError) as error:
                self.error = str(error)
            return
        try:
            import pyamg
        except ImportError:
            warnings.warn(
                'PyAMG is unavailable; switching to CG.', RuntimeWarning, stacklevel=2
            )
            return
        if matrix.indptr.dtype == np.int64 or matrix.indices.dtype == np.int64:
            if max(matrix.shape[0], matrix.nnz) > np.iinfo(np.int32).max:
                warnings.warn(
                    'AMGCG does not support this matrix index range; switching to CG.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            matrix.indptr = matrix.indptr.astype(np.int32)
            matrix.indices = matrix.indices.astype(np.int32)
        try:
            hierarchy = pyamg.ruge_stuben_solver(matrix)
            self.preconditioner = hierarchy.aspreconditioner(cycle='V')
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            warnings.warn(
                f'AMGCG setup failed ({error}); switching to CG.',
                RuntimeWarning,
                stacklevel=2,
            )

    def solve(self, gradient):
        if self.error is not None:
            return None
        try:
            if self.factorisation is not None:
                solution = self.factorisation.solve(-gradient.ravel())
                info = 0
            else:
                solution, info = cg(
                    self.operator,
                    -gradient.ravel(),
                    x0=np.zeros(gradient.size),
                    M=self.preconditioner,
                    rtol=1e-6,
                    maxiter=500,
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return None
        if info != 0 or not np.all(np.isfinite(solution)):
            return None
        return solution.reshape(gradient.shape)


@dataclass
class ObjectiveResult:
    """Internal objective outcome without changing public contraction returns."""

    coordinates: np.ndarray
    status: str
    updates: int
    energies: tuple
    backtracks: int = 0
    projected: int = 0


def solve_objective(objective, solver='CG', project=None):
    """Minimise a frozen objective with quadratic directions and actual trial checks."""
    displacement = np.zeros_like(objective.anchor)
    value, gradient = objective.evaluate(displacement)
    energies = [value]
    backtracks = 0
    projected_count = 0
    stable_updates = 0
    updates = 0

    def outcome(status):
        return ObjectiveResult(
            objective.anchor + displacement,
            status,
            updates,
            tuple(energies),
            backtracks,
            projected_count,
        )

    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return outcome('linear_failure')
    if not np.any(gradient):
        return outcome('converged')
    try:
        linear_solver = _QuadraticSolver(objective, solver)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return outcome('linear_failure')
    limit = _MAX_INNER_UPDATES if objective.lambda_f else 1
    radius = float(np.min(objective.spacing))
    for _ in range(limit):
        direction = linear_solver.solve(gradient)
        if direction is None or np.sum(gradient * direction) >= 0:
            return outcome('linear_failure')
        correction = np.max(np.linalg.norm(direction * objective.spacing, axis=1))
        multiplier = 1.0
        accepted = False
        for reduction in range(_MAX_BACKTRACKS + 1):
            trial = objective.anchor + displacement + multiplier * direction
            projected = 0
            if project is not None:
                trial, projected = project(trial)
            trial_displacement = trial - objective.anchor
            step = trial_displacement - displacement
            prediction = -(
                np.sum(gradient * step)
                + 0.5 * np.dot(step.ravel(), objective.hessian @ step.ravel())
            )
            if prediction > 0 and np.all(np.isfinite(trial)):
                trial_value, _ = objective.evaluate(
                    trial_displacement, derivative=False
                )
                if (
                    np.isfinite(trial_value)
                    and trial_value <= value - _SUFFICIENT_DECREASE * prediction
                ):
                    accepted = True
                    break
            if reduction < _MAX_BACKTRACKS:
                multiplier *= 0.5
                backtracks += 1
        if not accepted:
            return outcome('stalled')
        relative_change = abs(trial_value - value) / max(
            1, abs(value), abs(trial_value)
        )
        displacement = trial_displacement
        projected_count += projected
        updates += 1
        energies.append(trial_value)
        value, gradient = objective.evaluate(displacement)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return outcome('linear_failure')
        if not np.any(gradient):
            return outcome('converged')
        small_update = (
            correction < _CORRECTION_TOL * radius and relative_change < _ENERGY_TOL
        )
        stable_updates = stable_updates + 1 if small_update else 0
        if stable_updates >= 2:
            return outcome('converged')
        if not objective.lambda_f:
            # A full quadratic step solves the frozen model. Projection or a
            # shortened step must not masquerade as a fully solved objective.
            residual = np.linalg.norm(gradient)
            initial_gradient = objective.evaluate(np.zeros_like(displacement))[1]
            if residual <= 1e-6 * np.linalg.norm(initial_gradient):
                return outcome('converged')
    return outcome('iteration_limited')
