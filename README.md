<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->

[![All Contributors](https://img.shields.io/badge/all_contributors-0-orange.svg?style=flat-square)](#contributors-)

<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Code Structure

Start with `laplskel/cli/run_laplskel.py` for CLI options and `laplskel/workflows.py`
for input loading, component processing, and output assembly. Algorithm modules
live in `laplskel/`:

| Work area               | Main files                                  | Responsibility                                                                                                                                     |
| ----------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thinning and refinement | `refinement.py`                           | Topology-preserving voxel deletion, graph extraction, and the optional second fit.                                                                  |
| Contraction             | `contraction.py`                          | Laplacian contraction, linear solvers, convergence, and edge collapse.                                                                             |
| Runtime efficiency      | `parallelisation.py`                      | Component cropping, worker scheduling, and per-component algorithm dispatch. Solver and thinning optimizations also touch their algorithm modules. |
| Shared foundations      | `graph.py`, `medial.py`, `objects.py` | Sparse adjacency/Laplacians, medialness scoring, and union-find.                                                                                    |
| Data utilities          | `utils.py`                                | Component labeling, voxel reconstruction, and GraphML export.                                                                                      |

Thinning and contraction can be developed separately within their modules.
Coordinate changes to shared helpers and parameter forwarding through the CLI,
workflow, and component dispatcher. Runtime work spans scheduling
and algorithm internals, so agree on ownership before editing those internals.
Preserve coordinate conventions, graph topology, return values, and numerical
defaults across these boundaries.

## Standard Workflow Output

The workflow runs contraction, including contraction-stage decimation controlled by
`--decimate_ratio` and `--dec_grid_size`, then topology-preserving thinning and graph
extraction. Contraction always uses EDT-scaled retention, the anisotropic Laplacian and
hard foreground containment; its medial retention boost `--w_H_medial` defaults to 20.
The exported graph keeps every digital skeleton voxel as a node, with straight edges
between neighbouring voxel centres. Add `--second_fitting` to also smooth the node
positions of the extracted graph with the contraction flow, holding endpoints and junctions in place. The fit
shares the contraction settings (`--w_L`, `--beta_edt`, `--tol_max_drift`,
`--tol_mean_drift`, `--max_iter`, `--local_pca_hops`) except its own retention weights,
`--fit_w_H` (default 0.16) and `--fit_w_H_medial` (default 3), tuned on the synthetic
trees; these two flags require `--second_fitting`. The fit always uses containment, no
merging and the LU solver. Every skeleton voxel still stays a node. The NIfTI skeleton
is the digital skeleton before fitting in both cases.

### Output files

`-o/--output` is required and names the skeleton NIfTI, as an absolute or relative path
ending in `.nii.gz` or `.nii` (for example `./results/case01/skeleton.nii.gz`). By
default only this NIfTI is written. Graph files are written only when asked for, each
at exactly the path given; the ending selects the format. Missing folders are created,
and two outputs may not share a file.

| Option | Writes |
| --- | --- |
| `-o out.nii.gz` / `-o out.nii` | always: the digital skeleton |
| `--graph_output g.graphml` | the skeleton graph as GraphML |
| `--graph_output g.npz` | the skeleton graph's sparse adjacency in `g.npz`, coordinates and edge paths in `g_coords.npz` |
| `--graph_original o.graphml` / `o.npz` | the contracted graph before thinning, same format rules (the `.npz` companion holds coordinates only) |
