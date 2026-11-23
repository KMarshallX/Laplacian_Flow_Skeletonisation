#!/usr/bin/env python3

import os

import numpy as np
from nigsp import io
from scipy import sparse

from .parallelisation import process_components
from .refinement import DEFAULT_FIT_W_H, DEFAULT_FIT_W_H_MEDIAL
from .utils import coords_to_dense_3d, label_and_sort_by_size, write_graphml

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


def _nifti_output(path):
    """Return the skeleton NIfTI output path; it must end in .nii or .nii.gz."""
    path = '' if path is None else os.fspath(path)
    if not path.lower().endswith(NIFTI_EXTENSIONS):
        raise ValueError(
            f'The output NIfTI path must end in .nii or .nii.gz, got {path!r}.'
        )
    return path


def _graph_files(path, option):
    """Return the files one graph output writes; its ending selects the format.

    A .graphml path is written as GraphML. A .npz path holds the sparse adjacency,
    with coordinates (and edge paths, if any) in <stem>_coords.npz beside it.
    """
    path = os.fspath(path)
    stem, extension = os.path.splitext(path)
    if extension.lower() == '.graphml' and stem:
        return [path]
    if extension.lower() == '.npz' and stem:
        return [path, f'{stem}_coords.npz']
    raise ValueError(f'{option} must end in .graphml or .npz, got {path!r}.')


def _write_graph(path, coordinates, adjacency, **graphml_options):
    """Write one graph to a .graphml or .npz path (see _graph_files)."""
    path = os.fspath(path)
    stem, extension = os.path.splitext(path)
    if extension.lower() == '.graphml':
        write_graphml(coordinates, adjacency, output_path=path, **graphml_options)
        return
    arrays = dict(contracted_X=coordinates)
    edge_paths = graphml_options.get('edge_paths')
    if edge_paths is not None:
        ordered_edges = sorted(edge_paths)
        paths = [edge_paths[edge] for edge in ordered_edges]
        arrays.update(
            edge_path_nodes=np.asarray(ordered_edges, dtype=int).reshape(-1, 2),
            edge_path_offsets=np.cumsum([0] + [len(path) for path in paths]),
            edge_path_points=np.vstack(paths) if paths else np.empty((0, 3)),
        )
    np.savez_compressed(f'{stem}_coords.npz', **arrays)
    sparse.save_npz(path, adjacency)


def laplacian_skeletonisation(
    nifti_path,
    out_path,
    beta_edt=1.0,
    w_L=0.5,
    w_H_base=0.5,
    w_H_medial=20.0,
    tol_max_drift=0.05,
    tol_mean_drift=0.0025,
    max_iter=500,
    decimate_ratio=0.93,
    min_edge_length=0.01,
    separate_streams=False,
    label_connectivity=26,
    init_graph_adj=26,
    local_pca_hops=1,
    solver='CG',
    n_jobs=None,
    graph_output=None,
    graph_original=None,
    second_fitting=False,
    fit_w_H=DEFAULT_FIT_W_H,
    fit_w_H_medial=DEFAULT_FIT_W_H_MEDIAL,
):
    """
    Load a NIfTI file volume image and perform geometric graph contraction skeletonisation.

    Contraction always uses EDT-scaled retention, the anisotropic Laplacian and
    hard foreground containment.

    Parameters
    ----------
    nifti_path : str
        File system location path pointing directly toward the source input .nii or .nii.gz file.
    out_path : str or os.PathLike
        Output skeleton NIfTI path, absolute or relative, ending in .nii.gz or .nii.
        Missing folders are created. Required.
    beta_edt : float, optional
        Scaling modulation weight assigned to boundary energy calculation properties.
        Shared by contraction and the second fit. Default is 1.0.
    w_L : float, optional
        Contraction weight step modifier targeting structural local geometric collapse.
        This should be alpha in Damseh 2021. Shared by contraction and the second
        fit. Default is 0.5.
    w_H_base : float, optional
        Baseline contraction retention weight. This should be equivalent to beta in
        Damseh 2021. The second fit uses fit_w_H instead. Default is 0.5.
    w_H_medial : float, optional
        Multiplicative contraction retention boost applied to nodes at or around
        inscribed-sphere centres, raised to the power of each node's medialness
        score. Holding medial nodes in place tightens the centreline onto the medial
        axis. A value of 1.0 disables the boost. The second fit uses fit_w_H_medial
        instead. Default is 20.0.
    tol_max_drift : float, optional
        Maximum displacement in voxel units. Contraction (and the second fit) stops
        after three consecutive steps below it without structural changes; it is
        also the small-step decimation threshold. Default is 0.05.
    tol_mean_drift : float, optional
        Mean displacement in voxel units. Contraction (and the second fit) also
        stops after three consecutive steps below it without structural changes.
        Zero disables this test. Default is 0.0025.
    max_iter : int, optional
        Maximum iterations of contraction and of the second fit. Default is 500.
    decimate_ratio : float, optional
        Adaptive decimation threshold in (0, 1]. Edge-collapse decimation runs once
        three consecutive mean-drift ratios (current / previous) lie in
        [decimate_ratio, 1]. Default is 0.93.
    min_edge_length : float, optional
        The Euclidean spatial threshold criteria below which two connected nodes undergo
        structural merging, expressed as a fraction of the isotropic voxel length.
        Default is 0.01.
    separate_streams : bool, optional
        Process each "independent" vessel by itself (i.e. non-connected segment)
    label_connectivity : 6, 18, 26, optional
        Must be 26 when separating components, to preserve foreground topology.
    init_graph_adj : {6, 18, 26}, int, optional
        Voxel-neighborhood connectivity used to construct the initial graph.
        Default is 26.
    local_pca_hops : int, optional
        Number of graph hops included in each node's neighborhood when estimating
        local tangent directions. Default is 1.
    solver : ['LU', 'CG', 'AMGCG'], string, optional
        The solver to use to solve the linear system Ax = b. LU uses SuperLU, a direct
        solver, CG uses Conjugate Gradient (iterative solver), better for memory on big
        data, AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before
        running CG, which makes it far faster, but may require a tad more memory.
        Default is CG.
    n_jobs : None, optional
        Number of parallel jobs. If not set or <=0, defaults to ~30%% of available CPU
        cores.
    graph_output : str or os.PathLike, optional
        Path for the skeleton graph, absolute or relative. A .graphml path is
        written as GraphML; a .npz path holds the sparse adjacency, with
        coordinates and edge paths in <stem>_coords.npz beside it. Missing folders
        are created. Default is None: no graph file, only the NIfTI skeleton.
    graph_original : str or os.PathLike, optional
        Path for the contracted graph before thinning, with the same format rules
        as graph_output (its .npz companion holds coordinates only). Default is
        None: not written.
    second_fitting : bool, optional
        Smooth the extracted skeleton graph's node positions with the contraction
        flow, holding endpoints and junctions in place. It shares the contraction
        settings except its own retention weights (fit_w_H, fit_w_H_medial). EDT
        retention, the anisotropic Laplacian and containment are always on,
        merging always off and the solver always LU. The graph keeps every digital
        skeleton voxel as a node either way; if False, nodes stay at voxel
        centres. Default is False.
    fit_w_H : float, optional
        Baseline retention weight of the second fit. Only used with
        second_fitting. Default is 0.16.
    fit_w_H_medial : float, optional
        Medial retention boost of the second fit; 1.0 disables it. Only used with
        second_fitting. Default is 3.0.

    Returns
    -------
    contracted_X : numpy.ndarray
        An (M, 3) matrix mapping the continuous 3D spatial points along the skeleton path.
        Without second_fitting, these are the digital skeleton voxel centres.
    final_adj : scipy.sparse.csr_matrix
        The resulting graph sparse adjacency connectivity representation of shape (M, M).
    nifti_skel : numpy.ndarray
        A dense nifti matrix with the skeleton inside.

    Raises
    ------
    ValueError
        If the loaded structural NIfTI mask image is completely empty or lacks foreground elements.
    """
    if not np.isfinite(decimate_ratio) or not 0 < decimate_ratio <= 1:
        raise ValueError('decimate_ratio must be finite and in (0, 1].')
    if not np.isfinite(tol_mean_drift) or tol_mean_drift < 0:
        raise ValueError('tol_mean_drift must be finite and >= 0.')
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError('max_iter must be a positive integer.')
    if not np.isfinite(fit_w_H) or fit_w_H <= 0:
        raise ValueError('fit_w_H must be finite and > 0.')
    if not np.isfinite(fit_w_H_medial) or fit_w_H_medial < 1:
        raise ValueError('fit_w_H_medial must be finite and >= 1.')
    if separate_streams and label_connectivity != 26:
        raise ValueError(
            'Tunnel preservation requires --label_connectivity 26 '
            'when separating streams.'
        )
    nifti_out = _nifti_output(out_path)
    output_files = [nifti_out]
    for option, path in (('graph_output', graph_output),
                         ('graph_original', graph_original)):
        if path is not None:
            output_files += _graph_files(path, option)
    resolved = [os.path.abspath(path) for path in output_files]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f'Output paths must be distinct files, got {output_files}.')
    # Create output folders first, so a bad path fails before processing.
    for folder in {os.path.dirname(path) for path in resolved}:
        os.makedirs(folder, exist_ok=True)
    print(f'Ingesting NIfTI image: {nifti_path}')
    _, volume_data, img = io.load_nifti_get_mask(nifti_path, is_mask=True, ndim=3)

    if not np.any(volume_data):
        raise ValueError('Provided segmentation volume lacks any foreground structure.')

    # Process each component independently if separate_streams is True
    if separate_streams:
        labeled_volume, num_features = label_and_sort_by_size(
            volume_data, label_connectivity
        )
    else:
        labeled_volume, num_features = volume_data * 1, 1

    results = process_components(
        labeled_volume,
        num_features,
        beta_edt,
        w_L,
        w_H_base,
        w_H_medial,
        tol_max_drift,
        init_graph_adj,
        local_pca_hops,
        decimate_ratio,
        min_edge_length,
        n_jobs,
        solver,
        img.header.get_zooms()[:3],
        graph_original is not None,
        second_fitting,
        tol_mean_drift,
        max_iter,
        fit_w_H,
        fit_w_H_medial,
    )

    print('Reuniting results from parallel jobs.')

    if not results:
        raise ValueError('No valid components found for contraction.')

    # Merge coordinates and sparse block-diagonal adjacency matrices across all labels
    contracted_X = np.vstack([res[1] for res in results])
    final_adj = sparse.block_diag([res[2] for res in results], format='csr')
    # Keep the digital reference and edge paths: rounding simplified straight
    # segments can create extra voxel loops even when graph topology is correct.
    digital_voxels = np.vstack([res[3] for res in results])
    nifti_skel = coords_to_dense_3d(digital_voxels, volume_data.shape)
    edge_paths = {}
    node_offset = 0
    for result in results:
        edge_paths.update({
            (u + node_offset, v + node_offset): path
            for (u, v), path in result[4].items()
        })
        node_offset += len(result[1])

    if graph_output is not None:
        print(f'\nSaving skeleton graph to {graph_output}')
        labels = np.concatenate(
            [np.full(len(res[1]), res[0], dtype=int) for res in results]
        )
        _write_graph(
            graph_output, contracted_X, final_adj, component_labels=labels,
            affine=img.affine, volume_shape=volume_data.shape,
            binary_segmentation=volume_data, edge_paths=edge_paths,
        )
    if graph_original is not None:
        print(f'Saving contracted graph before thinning to {graph_original}')
        _write_graph(
            graph_original,
            np.vstack([res[5] for res in results]),
            sparse.block_diag([res[6] for res in results], format='csr'),
            component_labels=np.concatenate(
                [np.full(len(res[5]), res[0], dtype=int) for res in results]
            ),
            affine=img.affine,
            volume_shape=volume_data.shape,
        )
    print(f'Saving skeleton NIfTI to {nifti_out}')
    io.export_nifti(nifti_skel, img, nifti_out)

    return contracted_X, final_adj, nifti_skel
