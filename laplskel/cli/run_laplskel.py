"""Command-line interface for Laplacian skeletonisation."""

import argparse
import sys

from laplskel.refinement import DEFAULT_FIT_W_H, DEFAULT_FIT_W_H_MEDIAL
from laplskel.workflows import _graph_files, _nifti_output, laplacian_skeletonisation

VALID_CONNECTIVITY = (6, 18, 26)
VALID_SOLVER = ('LU', 'CG', 'AMGCG')


def _nifti_path(value):
    """Parse the skeleton NIfTI output path (.nii or .nii.gz)."""
    try:
        return _nifti_output(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _graph_path(option):
    """Return a parser for a graph output path (.graphml or .npz)."""
    def parse(value):
        try:
            _graph_files(value, option)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        return value
    return parse


def _positive_integer(value):
    """Parse a strictly positive integer argument."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser(
        description='Configurable EDT-Guided Laplacian Graph Contraction Pipeline.',
        add_help=False,
    )

    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--input',
        '-i',
        dest='nifti_path',
        type=str,
        required=True,
        help='Path pointing toward the input .nii or .nii.gz file volume.',
    )
    required.add_argument(
        '--output',
        '-o',
        dest='out_path',
        type=_nifti_path,
        required=True,
        help=(
            'Output skeleton NIfTI path, absolute or relative, ending in .nii.gz or '
            '.nii (e.g. ./results/case01/skeleton.nii.gz). Missing folders are created.'
        ),
    )

    optional = parser.add_argument_group('Other Optional Arguments')
    optional.add_argument(
        '--graph_output',
        type=_graph_path('--graph_output'),
        default=None,
        help=(
            'Also write the skeleton graph to this path, absolute or relative. The '
            'ending selects the format: .graphml for GraphML, or .npz for the sparse '
            'adjacency with coordinates and edge paths in <stem>_coords.npz beside it. '
            'Missing folders are created. Without it only the NIfTI is written.'
        ),
    )
    optional.add_argument(
        '--graph_original',
        type=_graph_path('--graph_original'),
        default=None,
        help=(
            'Also write the contracted graph before thinning to this path, with the '
            'same format rules as --graph_output.'
        ),
    )
    optional.add_argument(
        '--second_fitting',
        action='store_true',
        help=(
            'Smooth the positions of the extracted skeleton graph nodes with the '
            'contraction flow, holding endpoints and junctions in place. It shares the '
            'contraction settings except its own retention weights (--fit_w_H, '
            '--fit_w_H_medial). The graph keeps every digital skeleton voxel as a node '
            'either way; by default nodes stay at voxel centres.'
        ),
    )
    optional.add_argument(
        '--fit_w_H',
        dest='fit_w_H',
        type=float,
        default=argparse.SUPPRESS,
        help=(
            'Baseline retention weight of the second fit. Requires --second_fitting '
            f'[Default={DEFAULT_FIT_W_H}].'
        ),
    )
    optional.add_argument(
        '--fit_w_H_medial',
        dest='fit_w_H_medial',
        type=float,
        default=argparse.SUPPRESS,
        help=(
            'Medial retention boost of the second fit; 1.0 disables it. Requires '
            f'--second_fitting [Default={DEFAULT_FIT_W_H_MEDIAL:g}].'
        ),
    )
    optional.add_argument(
        '--beta_edt',
        type=float,
        default=1.0,
        help=(
            'Custom scaling weight assigned to modulate the EDT boundary energy '
            'constraints.'
        ),
    )
    optional.add_argument(
        '--w_L',
        type=float,
        default=0.5,
        help='Contraction weight scalar multiplier variable.',
    )
    optional.add_argument(
        '--w_H',
        dest='w_H_base',
        type=float,
        default=0.5,
        help='Baseline structural anchor retention weight variable.',
    )
    optional.add_argument(
        '--w_H_medial',
        dest='w_H_medial',
        type=float,
        default=20.0,
        help=(
            'Contraction retention weight boost applied to nodes at or around '
            'inscribed-sphere centres, tightening the centreline onto the medial axis. '
            'Raised to the power of each node medialness score, so boundary nodes keep '
            'their baseline weight. 1.0 disables the boost. The second fit uses '
            '--fit_w_H_medial instead [Default=20].'
        ),
    )
    optional.add_argument(
        '--tol_max_drift',
        type=float,
        default=0.05,
        help=(
            'Stop contraction (and the second fit) after three consecutive steps whose '
            'largest node movement is below this value in voxels; also the small-step '
            'decimation threshold [Default=0.05].'
        ),
    )
    optional.add_argument(
        '--tol_mean_drift',
        type=float,
        default=0.0025,
        help=(
            'Also stop contraction (and the second fit) after three consecutive steps '
            'whose mean node movement is below this value in voxels; 0 disables this '
            'test [Default=0.0025].'
        ),
    )
    optional.add_argument(
        '--max_iter',
        type=_positive_integer,
        default=500,
        help='Maximum contraction (and second-fit) iterations [Default=500].',
    )
    optional.add_argument(
        '--decimate_ratio',
        dest='decimate_ratio',
        type=float,
        default=0.93,
        help=(
            'Decimate once three consecutive mean-drift ratios (current/previous) '
            'lie in [ratio, 1]; must be in (0, 1] [Default=0.93].'
        ),
    )
    optional.add_argument(
        '--dec_grid_size',
        dest='min_edge_length',
        type=float,
        default=0.01,
        help=(
            'Maximum intermediate merge-cluster diameter in voxel units, used by '
            'contraction-stage decimation [Default=0.01].'
        ),
    )
    optional.add_argument(
        '--init_graph_adj',
        type=int,
        choices=VALID_CONNECTIVITY,
        default=26,
        help=(
            'Voxel-neighborhood connectivity for the initial graph (6, 18, or 26) '
            '[Default=26].'
        ),
    )
    optional.add_argument(
        '--local_pca_hops',
        type=_positive_integer,
        default=1,
        help=(
            'Number of graph hops used for each local tangent PCA neighborhood '
            '[Default=1].'
        ),
    )
    optional.add_argument(
        '--separate_streams',
        action='store_true',
        help=(
            'Separate segmentation components into individual labels using '
            'scipy.ndimage.label and run contraction on each independently.'
        ),
    )
    optional.add_argument(
        '--label_connectivity',
        type=int,
        choices=VALID_CONNECTIVITY,
        default=26,
        help='Component labeling connectivity; use 26 for tunnel preservation [Default=26].',
    )
    optional.add_argument(
        '--solver',
        type=str,
        choices=VALID_SOLVER,
        default='CG',
        help=(
            'The solver to use to solve the linear system in computing the new '
            'coordinates system. LU uses SuperLU, a direct solver, CG uses Conjugate '
            'Gradient (iterative solver), better for memory performance on big data, '
            'AMGCG constructs an Algebraic Multigrid (AMG) preconditioner before '
            'running CG, far faster (but may be a bit more memory demanding than pure '
            'CG) [Default=CG].'
        ),
    )
    optional.add_argument(
        '--n_jobs',
        type=int,
        default=None,
        help=(
            'Number of parallel jobs. If not set or <=0, defaults to '
            '~30%% of available CPU cores.'
        ),
    )
    optional.add_argument(
        '-h', '--help', action='help', help='Show this help message and exit'
    )
    return parser


def _main(argv=None):
    parser = _get_parser()
    args = parser.parse_args(argv)
    fit_options = [name for name in ('fit_w_H', 'fit_w_H_medial') if name in vars(args)]
    if fit_options and not args.second_fitting:
        verb = 'require' if len(fit_options) > 1 else 'requires'
        parser.error(
            ' and '.join(f'--{name}' for name in fit_options) + f' {verb} --second_fitting.'
        )

    laplacian_skeletonisation(**vars(args))


if __name__ == '__main__':
    _main(sys.argv[1:])
