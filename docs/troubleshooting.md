# Troubleshooting

[Back to README](../README.md)

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Warnings about `launch-ros`/`ament` or odd imports from `ros2_humble` | `~/.bashrc` sources ROS 2 Humble, which adds Python 3.10 packages to `PYTHONPATH`. Run `unset PYTHONPATH` in the shell (or prefix commands with `env -u PYTHONPATH`). |
| VS Code reports `opencv-python-headless`/`imageio` "not installed" | The editor uses a different interpreter. Select `.env/bin/python`. |
| `pip install` or imports are very slow | The venv (`.env/`) is on `/mnt/<drive>`, where file I/O is slow. This is expected; a venv on the Linux filesystem is faster. |
| `ERROR: Image directory not found ... image_2` | KITTI is not unpacked at `data/raw/kitti/dataset`. See [Dataset preparation](installation.md#kitti-odometry-selective-download-done), or pass `--root`. |
| `Pose/frame misalignment` | `poses/XX.txt` and `times.txt` come from different sequences or an incomplete download. |
| GIF larger than 10 MB | Lower `--max-frames` or `--gif-width`. Noisy images compress poorly in GIF. |
| `unable to mmap ... Cannot allocate memory (12)` when loading a checkpoint | Private memory maps are charged against RAM + swap. The project loads with a shared map (`load_mmap`); call that instead of `torch.load(..., mmap=True)`. |
| I-JEPA inference is slower than expected in FP16 | GPUs without Tensor Cores (e.g. GTX 16xx) can be slower in FP16. Keep `ijepa.precision: auto`, which benchmarks and picks FP32 here. |
| `ModuleNotFoundError: faiss` | Install the ML extra: `pip install -e ".[dev,ml]"` (or `pip install faiss-cpu==1.15.1`). |
| `descriptors must be L2-normalised` from the retrieval index | The descriptors did not come from a `VisualEncoder` (which normalises them). Normalise with `latent_loop.features.pooling.l2_normalize`. |
| A stopped vehicle "revisits" itself | The 100-frame exclusion assumes motion. Add `--set retrieval.temporal_exclusion_seconds=...`; a distance-travelled rule is future work. |
| Verification of KITTI is slow (~0.2 s per pair) | RANSAC runs up to 2000 iterations on pairs with few inliers, i.e. on most wrong candidates. Lower `--top-k`, or use `--estimator numpy` (vectorised, adaptive stopping; median RANSAC 40 vs 206 ms per pair on KITTI 00). |
| `No retrieval result in ...` from `run_verification.py` | Phase 4 reads saved Phase 3 results. Run `scripts/run_retrieval.py` with the same `--config`, `--stride` and `--set` first. |
| `Re-verification of (q, c) gave ...` from `--phase 4` | The saved verification result was produced with different code or settings. Re-run `scripts/run_verification.py`. |
| Many correct revisits flagged `low_parallax` | Expected: revisits along the same lane give nearly identical views. The place match is correct, but its relative pose is poorly conditioned. Only `degenerate_policy: reject` turns the flag into a rejection. |
| `ORB visual vocabulary not found` | Fit it first: `python scripts/fit_vocabulary.py` with the same `--config`/`--set` as the evaluation run. |
| `Vocabulary fit sequences [...] are not development-only` | `orb_bovw.fit_kitti_sequences` lists a held-out (`splits.test`) or unlisted sequence. Only `splits.dev` sequences may be used for fitting. |
| `DINOv2 weights not found` | `python scripts/download_checkpoints.py dinov2 --name vitb14` (346 MB). |
| `No cached descriptors / retrieval result for <method>` from `--phase 5` | Run `scripts/run_retrieval.py --encoder <method>` with the same `--stride`/`--set` (step 4 of Phase 5). |
| Phase 5 GIF larger than 10 MB | Use fewer frames, a narrower width or fewer palette colours, e.g. `--max-frames 64 --gif-width 800 --gif-colors 128`. |
| `Run the development stage first` from `run_benchmark.py --stage test` | The test stage only applies `selection.json`/`thresholds.json` from the development stage. Run `--stage dev` first. |
| Benchmark verification is slow | Increase `--workers` (CPU cores), or lower `benchmark.verify_top_k`. Verified pairs are cached in `results/benchmark/<name>/pair_cache/`, so re-runs are fast. |
| Extraction much slower than the profile | The profile excludes disk I/O; on `/mnt/e` image reads can dominate (KITTI 08: 665 s of GPU encoding within a 46 min extraction). |
| Wall-clock gaps in long runs | The machine slept. Timings use a monotonic clock that pauses during suspend, so measurements are unaffected. |
| `torch.cuda.OutOfMemoryError` | Handled automatically: the batch is halved, then the CPU is used. Lower `extraction.batch_size` to avoid the retries. |
| `I-JEPA target-encoder weights not found` | Run `python scripts/download_checkpoints.py ijepa --name vith14_in1k` (step 2 of [Phase 2](phase2_ijepa.md#steps-to-run-phase-2)). |
| `import torch` fails with `cannot read file data` | A corrupted PyTorch install (seen after two concurrent `pip install` runs). Run `pip install --force-reinstall --no-deps torch==2.15.0`. |
| Downloads or extraction look stuck in `D` state (`p9_client_rpc`) | `/mnt/<drive>` I/O is slow under concurrent load. Wait, or run one heavy job at a time. |
| `ModuleNotFoundError: streamlit` / `plotly` | `pip install -e ".[dashboard]"` |
| Dashboard: "No benchmark results in ..." | Run `scripts/run_benchmark.py --stage dev` then `--stage test`, or point the app at another run: `streamlit run app/streamlit_app.py -- --set benchmark.output_dir=DIR` |
| Dashboard: "KITTI images not found" | Only the image-based views need KITTI; overview and benchmark work without it |
| Dashboard screenshots show grey placeholders | Raise `--delay` in `scripts/capture_dashboard.py` (45 s was enough on `/mnt/e`) |
| `pkill -f streamlit` also kills your shell | The pattern matches the shell's own command line; stop the server by PID |
| `--phase 8` raises `FileNotFoundError` | It needs the Phase 6 outputs, the I-JEPA embedding cache of KITTI 06 and the KITTI 06 images |
| `DINOv3 weights not found` | The weights are licence-gated and not in this repository. Request them at <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>, then run `python scripts/download_checkpoints.py dinov3 --name vitb16 --url "<your URL>"` |
| `Official DINOv3 code not found in external/dinov3` | `git clone https://github.com/facebookresearch/dinov3 external/dinov3` (git-ignored) |
| `SHA-256 ... does not match the official ...` | The download is incomplete or is a different file. Delete it and download again. Signed URLs expire after a few days |
| Hugging Face DINOv3 repos return 401 | They are gated with manual approval. Accept the licence on the model page and log in, or use Meta's e-mailed `.pth` links |
