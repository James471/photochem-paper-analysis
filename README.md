# Figures for the Quokka photochemistry paper

Scripts that run the simulations, reduce their output and make every figure of the paper.
All simulations use [Quokka](https://github.com/quokka-astro/quokka) at commit
[`295c78f90`](https://github.com/quokka-astro/quokka/tree/295c78f90ef2b5978a369b548fa5924595fabce5)
of the branch `james471/photochem-paper` (`paper` in `config.example.yaml`).

## Setup

```bash
./setup_env.sh                       # Python venv in .venv (numpy, pandas, matplotlib, scipy, yt, ...)
cp config.example.yaml config.yaml   # then set your paths: Quokka clone, run directories, cluster settings
```

`quokka` in `config.yaml` must be a clone of Quokka that has the pinned commit. The scripts build from a detached worktree
`quokka-<commit>` next to it, so the clone itself is never changed.

## One script per figure

| Script | Paper figure(s) | Where it runs |
|---|---|---|
| `figures/fig_flowchart.py` | `fig_flowchart.pdf` | no simulation (TikZ) |
| `figures/fig_onezone_photoionization.py` | `onezone_species_photons.pdf`, `onezone_errors.pdf` | local, seconds (needs Julia) |
| `figures/fig_tolerance_scaled.py` | `tolerance_scaled.pdf` | GPU (one GCD per run) |
| `figures/fig_integrator_choice.py` | `integrator_choice.pdf` | GPU (one GCD per run) |
| `figures/fig_stromgren_front_photons.py` | `stromgren_front_photons.pdf` | GPU |
| `figures/fig_stromgren_rsla.py` | `stromgren_snapshot.pdf`, `stromgren_ode.pdf`, `stromgren_rsla_chat.pdf` | GPU |
| `figures/fig_dtype_front.py` | `dtype_front_snapshot.pdf`, `dtype_front_radius.pdf` | GPU |
| `figures/fig_dtype_front_he.py` | `dtype_front_he_snapshot.pdf` | GPU |
| `figures/fig_dtype_front_3d.py` | `dtype_front_3d_snapshot.pdf`, `dtype_front_3d_radius.pdf` | GPU |

`fig_integrator_study.py` and `fig_tolerance_choice.py` hold helpers for the tolerance and integrator figures.

Each script declares its runs (`RUNS`) and has the stages

```bash
python figures/fig_x.py configure|build [RUN ...]   # cmake / build at the pinned commit
python figures/fig_x.py run [RUN ...] [--launch]    # prepare run directories (inputs, provenance run.json, job script); submit
python figures/fig_x.py reduce [RUN ...]            # raw output -> results/<figure>/<run>/ (small files)
python figures/fig_x.py plot                        # results -> <figures>/<name>.pdf
python figures/fig_x.py status
```

Runs, builds and job scripts for Setonix (Slurm) and Gadi (PBS) are handled by `photochem/`; the job templates are in `slurm/`.
`sync.py` runs a command in a cluster's copy of this repository (`sync.py remote -- python figures/fig_x.py reduce`) and copies
the reduced results back (`sync.py pull <figure>`).
