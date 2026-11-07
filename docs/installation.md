# Installation and data

[Back to README](../README.md)

## Installation

### Hardware used for development

Measured with `scripts/check_environment.py` (the report is saved to `results/metrics/environment.json`):

| Item | Value |
|---|---|
| Host | Lenovo Legion, Windows 11 + WSL2 Ubuntu 22.04 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| CPU | AMD Ryzen 7 4800H, 8 cores / 16 threads |
| RAM (visible to WSL) | 7.4 GiB + 2 GiB swap |
| GPU | NVIDIA GeForce GTX 1650 Ti, 4 GiB VRAM, compute capability 7.5, driver 591.74 (CUDA 13.1) |
| Python | 3.12.13 (3.11 is not installed on this machine; the package requires >= 3.11) |

Measured implications:
- **Precision.** The GTX 1650 Ti (TU117, no Tensor Cores) is *slower* in FP16 than in FP32: 0.36 vs 1.62 TFLOPS on a 4096² matmul, and 943 vs 237 ms/image for ViT-H/14. It also has no native BF16. Inference therefore runs in FP32, selected automatically by a micro-benchmark (see [Phase 2](phase2_ijepa.md#gpu-optimisation-and-robustness)).
- **VRAM.** 4 GiB holds the 2.5 GB FP32 ViT-H/14 weights plus batch-8 activations (peak 2.68 GB reserved). Models are loaded one at a time.
- **RAM.** 7.4 GiB of RAM means memory-mapped checkpoint loading and disk-cached embeddings.

### Environment setup

The repository and its virtual environment `.env/` live together on the Windows data drive
(`/home/ubuntu_22/harish_ws/hdd_ws/jepa_slam` → `/mnt/e/project_ws/jepa_slam`).

```bash
cd /home/ubuntu_22/harish_ws/hdd_ws/jepa_slam   # repository root
unset PYTHONPATH                                # see Troubleshooting: ROS 2 leaks into PYTHONPATH
python3.12 -m venv .env                         # any Python >= 3.11; .env/ is git-ignored
source .env/bin/activate
pip install --upgrade pip
pip install -e ".[dev,ml]"                      # ml = PyTorch (Phase 2+), faiss-cpu (Phase 3+)
```

On this machine `pip install` of PyTorch 2.15.0 (CUDA 13.2 wheels, about 3 GB unpacked) took
about 25 minutes on `/mnt/e`. File I/O over the WSL–Windows bridge (DrvFs/9P) is slow, so a
venv on the Linux filesystem (e.g. `~/.venvs/latent-loop`) is several times faster if speed
matters more than keeping everything on one drive. `requirements.txt` lists the same pinned
dependencies for non-editable installs.

In VS Code, choose **Python: Select Interpreter** → `.env/bin/python` so that diagnostics see
the installed packages.

### Repository layout (Phases 1–6)

```text
configs/            default.yaml (synthetic), kitti.yaml, ijepa.yaml, dinov2.yaml, benchmark.yaml
src/latent_loop/
  datasets/         base.py, kitti.py, kitti_download.py, synthetic.py, ground_truth.py
  encoders/         base.py (VisualEncoder), vit.py, ijepa.py, dinov2.py, classical.py
  features/         extraction.py, pooling.py, cache.py
  retrieval/        index.py (FAISS), temporal_filter.py, detector.py
  verification/     orb.py, geometry.py, pipeline.py, parallel.py
  evaluation/       metrics.py, profiling.py, benchmark.py, runner.py
  utils/            config.py, logging.py, device.py
  visualization/    plots.py, trajectory.py, animations.py, embeddings.py, phase2.py,
                    retrieval.py, phase3.py, matches.py, phase4.py, baselines.py, phase5.py,
                    evaluation.py, phase6.py
scripts/            check_environment.py, prepare_dataset.py, download_kitti.py,
                    download_checkpoints.py, extract_embeddings.py, verify_ijepa.py,
                    verify_dinov2.py, fit_vocabulary.py, profile_encoders.py,
                    run_retrieval.py, run_verification.py, compare_ransac.py, run_benchmark.py,
                    generate_visualizations.py
tests/              conftest.py + test_{datasets,ground_truth,config,visualization,kitti_download,
                    encoders,features,metrics,embedding_plots,retrieval,retrieval_plots,
                    verification,verification_plots,baselines,baseline_plots,benchmark}.py
assets/phase_0{1..6}_*/    committed figures and GIFs
results/            run outputs: retrieval/, verification/, benchmark/ (small, trackable);
                    metrics/, plots/, examples/ are git-ignored
data/ checkpoints/ .env/           git-ignored (datasets, weights, caches, venv)
```

## Dataset preparation

### KITTI Odometry (selective download, done)

The official colour archive (`data_odometry_color.zip`, 69.0 GB) contains all 22 sequences
and both colour cameras. Loop-closure experiments need only the sequences that revisit
places, from one camera, so `scripts/download_kitti.py` fetches **individual files** from the
official archives using HTTP range requests:
1. Read the zip index (central directory).
2. Stream each selected byte span once.
3. Verify every file against its CRC-32.
4. Write files atomically, resuming if interrupted.

```bash
python scripts/download_kitti.py --list-only   # sizes only, nothing written
python scripts/download_kitti.py               # default: sequences 00 02 05 06 08 09, camera 2
```

Downloaded on this machine (24 min 21 s at about 10.5 MB/s; every file CRC-verified):

| Archive | Selected | Size |
|---|---|---|
| `data_odometry_color.zip` (69.0 GB) | `image_2` of sequences 00, 02, 05, 06, 08, 09: 18,726 PNGs | 15.35 GB |
| `data_odometry_calib.zip` | `calib.txt` + `times.txt` of all 22 sequences | 0.6 MB |
| `data_odometry_poses.zip` | ground-truth poses of sequences 00–10 | 1.3 MB |

Layout (what `configs/kitti.yaml` expects):

```text
data/raw/kitti/dataset/
├── sequences/00/
│   ├── image_2/000000.png ...   # left colour camera
│   ├── calib.txt                # P0..P3 (3x4 rectified projections), Tr (velodyne -> cam0)
│   └── times.txt                # one timestamp [s] per frame
└── poses/00.txt                 # ground truth, sequences 00-10 only
```

Validated with the Phase 1 loader (`python scripts/prepare_dataset.py kitti --sequence <SS>`):

| Seq | Frames | Image size | Trajectory | fx [px] | Revisit frames | same dir. / opposite / crossing |
|---|---|---|---|---|---|---|
| 00 | 4541 | 1241 × 376 | 3.72 km | 718.86 | 804 (17.7 %) | 744 / 2 / 58 |
| 02 | 4661 | 1241 × 376 | 5.07 km | 718.86 | 315 (6.8 %) | 246 / 44 / 25 |
| 05 | 2761 | 1226 × 370 | 2.21 km | 707.09 | 448 (16.2 %) | 390 / 0 / 58 |
| 06 | 1101 | 1226 × 370 | 1.23 km | 707.09 | 270 (24.5 %) | 268 / 0 / 2 |
| 08 | 4071 | 1226 × 370 | 3.22 km | 707.09 | 345 (8.5 %) | 0 / 287 / 58 |
| 09 | 1591 | 1226 × 370 | 1.71 km | 707.09 | 19 (1.2 %) | 19 / 0 / 0 |

*Revisit frames* are measured from the ground-truth poses (CPU): frames within 5 m of a frame at
least 100 frames older, classified by the heading difference to the nearest such frame as same
direction (< 30°), opposite (> 150°) or crossing (30–150°). KITTI 08 has no same-direction
revisits at all, and KITTI 09 has only 19 revisit frames.

KITTI is licensed CC BY-NC-SA 3.0 (non-commercial use; see
<https://www.cvlibs.net/datasets/kitti/>). Other sequences or the grayscale archive can be
fetched with `--sequences` and `--gray --cameras 0`.

### Synthetic dataset (no download)

```bash
python scripts/prepare_dataset.py synthetic --out data/processed/synthetic_kitti
```

This writes the synthetic sequence in KITTI layout: 599 PNGs, `calib.txt`, `times.txt`,
`poses/00.txt` and `revisit_pairs.txt`, 46 MB in total. The in-memory synthetic dataset does
not need this export. The export exists so the KITTI loader can be exercised end to end.

---
