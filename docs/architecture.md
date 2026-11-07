# Architecture

[Back to README](../README.md)

Module map, data flow per phase, and the design rules that keep the evaluation honest.
The full algorithm descriptions are in the phase documents linked from the README.

## Module map

```mermaid
flowchart LR
    subgraph utils
        config[config.py<br/>YAML + extends + overrides]
        logging[logging.py<br/>console + JSON lines]
        device[device.py<br/>hardware probe,<br/>device + precision resolution]
    end
    subgraph datasets
        base[base.py<br/>VisualSequenceDataset<br/>SamplingConfig, Frame,<br/>CameraCalibration]
        kitti[kitti.py<br/>KittiOdometryDataset]
        synthetic[synthetic.py<br/>SyntheticLoopDataset<br/>export_kitti_layout]
        gt[ground_truth.py<br/>pose I/O + SE3 helpers,<br/>pair_labels]
        dl[kitti_download.py<br/>selective zip-range download]
    end
    subgraph encoders
        ebase[base.py<br/>VisualEncoder]
        vit[vit.py<br/>ViT-H/14, sincos pos-embed]
        ijepa[ijepa.py<br/>IJEPAEncoder, checkpoint extraction]
        dino[dinov2.py<br/>DINOv2Encoder, DinoViT]
        classical[classical.py<br/>HSV histogram, ORB BoVW,<br/>vocabulary + dev-only guard]
    end
    subgraph features
        extraction[extraction.py<br/>batching, prefetch,<br/>OOM recovery, CPU fallback]
        pooling[pooling.py<br/>mean pool + L2]
        cache[cache.py<br/>EmbeddingCache by frame ID]
    end
    subgraph retrieval
        index[index.py<br/>FAISS IndexFlatIP<br/>+ IDSelectorRange]
        temporal[temporal_filter.py<br/>eligible prefix by frame ID / time]
        detector[detector.py<br/>offline + causal retrieve,<br/>candidate filtering]
    end
    subgraph verification
        orb[orb.py<br/>ORB, Hamming kNN,<br/>ratio + mutual, twin fraction]
        geometry[geometry.py<br/>essential-matrix RANSAC<br/>OpenCV 5-pt / own 8-pt,<br/>degeneracy, decisions]
        vpipe[pipeline.py<br/>FeatureStore, verify_candidates,<br/>VerificationRun, label_pairs]
        par[parallel.py<br/>process pool,<br/>pair cache by config hash]
    end
    subgraph evaluation
        metrics[metrics.py<br/>roc_auc, recall_at_k]
        profiling[profiling.py<br/>Profiler]
        bench[benchmark.py<br/>AP, detection metrics,<br/>PR curve, threshold]
        runner[runner.py<br/>Benchmark: run_dev, run_test]
    end
    subgraph dashboard
        dstore[data.py<br/>ResultStore: read-only<br/>cached results]
        app[app/streamlit_app.py<br/>5 views]
    end
    subgraph visualization
        plots[plots.py]
        traj[trajectory.py]
        anim[animations.py]
        emb[embeddings.py + phase2.py]
        ret[retrieval.py + phase3.py]
        mat[matches.py + phase4.py]
        base5[baselines.py + phase5.py]
        eval6[evaluation.py + phase6.py]
        p8[phase8.py<br/>end-to-end GIF]
    end
    ijepa --> vit
    ijepa --> ebase
    ijepa --> pooling
    extraction --> ebase
    extraction --> cache
    extraction --> profiling
    extraction --> base
    emb --> metrics
    emb --> gt
    detector --> index
    detector --> temporal
    detector --> cache
    geometry --> orb
    dino --> vit
    dino --> ebase
    classical --> ebase
    classical --> orb
    base5 --> detector
    base5 --> cache
    vpipe --> geometry
    vpipe --> detector
    vpipe --> gt
    mat --> vpipe
    par --> vpipe
    runner --> detector
    runner --> par
    runner --> bench
    runner --> extraction
    eval6 --> runner
    app --> dstore
    app --> mat
    dstore --> detector
    p8 --> dstore
    p8 --> mat
    kitti --> base
    synthetic --> base
    kitti --> gt
    synthetic --> gt
    traj --> plots
    anim --> plots
    anim --> base
    scripts[scripts/*.py] --> config
    scripts --> datasets
    scripts --> visualization
```

## Data flow (Phase 1)

```mermaid
sequenceDiagram
    participant CLI as generate_visualizations.py
    participant Cfg as load_config
    participant DS as build_dataset
    participant Viz as visualization
    CLI->>Cfg: configs/*.yaml + --set overrides
    Cfg-->>CLI: merged dict
    CLI->>DS: config["dataset"]
    DS-->>CLI: VisualSequenceDataset (sampled view)
    CLI->>DS: poses(), timestamps(), get_frame(i)
    CLI->>Viz: positions, frame IDs, images, revisit pairs
    Viz-->>CLI: PNG figures, GIF frames
```

## Design rules

1. **Original frame IDs are the identity of a frame.** View indices are only positions in a
   sampled list. Temporal exclusion and causal filtering (Phase 3) use frame IDs or timestamps.
2. **Ground truth is for labels only.** `get_pose` exists for evaluation and plots; retrieval
   code will receive only images and frame IDs and timestamps.
3. **Lazy I/O.** Images are decoded on access. Datasets hold poses and timestamps only.
4. **Determinism.** The synthetic world, trajectory and pixel noise are fully determined by
   `dataset.synthetic.seed`.

## Data flow (Phase 2)

```mermaid
sequenceDiagram
    participant CLI as extract_embeddings.py / generate_visualizations.py --phase 2
    participant DS as VisualSequenceDataset
    participant Enc as IJEPAEncoder
    participant Ext as extract_embeddings
    participant C as EmbeddingCache
    CLI->>Enc: build_encoder("ijepa", config)
    Note over Enc: device auto -> cuda/cpu; precision auto -> bf16 / benchmarked fp16 vs fp32
    CLI->>Ext: dataset, encoder, cache, batch_size
    Ext->>C: missing(frame_ids)
    loop batches of missing frame IDs
        Ext->>DS: get_image (prefetch thread)
        Ext->>Enc: encode_batch(images)
        Note over Enc: resize + ImageNet norm -> ViT-H/14 (inference_mode) -> mean pool -> L2
        Enc-->>Ext: (B, 1280) float32, unit norm
        Ext->>C: add(frame_ids, embeddings) every flush_every frames (atomic)
    end
    Ext->>C: load().select(dataset.frame_ids)
    C-->>CLI: embeddings in dataset order + profile
```

## Design rules (continued)

5. **Checkpoint identity is part of the cache key.** The encoder fingerprint hashes the checkpoint
   SHA-256, architecture, input size, resize mode, normalisation and pooling, so changing any of
   them can never silently reuse stale descriptors. Device and precision are recorded per run but
   not hashed.

## Data flow (Phase 3)

```mermaid
sequenceDiagram
    participant CLI as run_retrieval.py
    participant C as EmbeddingCache
    participant T as TemporalExclusion
    participant I as FlatIPIndex (FAISS)
    participant GT as revisit_ground_truth
    CLI->>C: descriptors for dataset.frame_ids (Phase 2 cache)
    CLI->>T: eligible_counts(frame_ids, timestamps)
    alt offline
        CLI->>I: add(all descriptors)
        CLI->>I: search(all queries, K, prefix limits)
    else causal replay
        loop t = 0 .. N-1
            CLI->>I: search(query t, K, min(limit_t, frames so far))
            CLI->>I: add(frame t)
        end
    end
    I-->>CLI: candidate frame IDs + cosine scores
    CLI->>CLI: filter (threshold, temporal NMS), assert_causal
    Note over CLI,GT: ground truth is computed only after retrieval, for labels and metrics
    CLI->>GT: poses -> positives per query (distance-only, same-direction)
    CLI-->>CLI: save retrieval.npz / retrieval.json / top1.csv
```

6. **Eligibility is a prefix.** Frames are indexed chronologically, so every query's eligible set
   is `[0, n_q)`, enforced inside FAISS with `IDSelectorRange`. Offline evaluation and causal
   replay are therefore exactly equivalent, which is tested.

## Data flow (Phase 4)

```mermaid
sequenceDiagram
    participant CLI as run_verification.py
    participant R as RetrievalResult (Phase 3)
    participant F as FeatureStore (ORB per frame ID)
    participant V as verify_features
    participant GT as label_pairs
    CLI->>R: load results/retrieval/.../<mode>/
    loop each query q, rank 1..verify_top_k
        CLI->>F: ORB(q), ORB(candidate) (computed once per frame)
        CLI->>V: features + CameraCalibration
        V->>V: twin fraction, ratio + mutual matching
        V->>V: essential-matrix RANSAC (Sampson, 1 px), recoverPose
        V->>V: homography diagnostic, parallax, degeneracy flag
        V-->>CLI: accepted / first failing reason + diagnostics
    end
    Note over CLI,GT: poses are used only after all decisions, to label pairs
    CLI->>GT: distance -> true revisit (<= 5 m) / not (> 25 m) / unlabelled
    CLI-->>CLI: save verification.csv / decisions.csv / verification.json (top<K>_<estimator>/)
```

7. **Decisions never see poses.** `verify_features`, `verify_pair` and `verify_candidates` take
   images (or ORB features) and the calibration only; a test asserts that their signatures have
   no pose or ground-truth argument. Labels are attached afterwards by `label_pairs`.
8. **Figures re-derive, then check.** Phase 4 figures re-run verification with
   `keep_details=True` for the pairs they draw and fail unless the decision and inlier count
   match the saved run exactly (RANSAC is seeded, so this is deterministic).

## Phase 5: one interface, four descriptors

```mermaid
flowchart LR
    F[frames of one sequence] --> I[I-JEPA ViT-H/14<br/>patch mean]
    F --> D[DINOv2 ViT-B/14<br/>CLS token]
    F --> H[HSV histogram<br/>sqrt-normalised]
    F --> O[ORB bag of words<br/>tf-idf]
    V[(vocabulary fitted on<br/>splits.dev only)] --> O
    I & D & H & O --> C[EmbeddingCache<br/>key = dataset + encoder fingerprint]
    C --> R[retrieve: same exclusion,<br/>top-K, offline / causal]
    R --> S[retrieval.npz per method]
    S --> E[Recall@K, AUC, figures]
```

9. **Development and test data are separated in configuration.** `splits.dev` (00, 05) may be
   used for fitting and tuning; `splits.test` (02, 06, 08, 09) may not. `vocabulary_plan`
   refuses to fit on anything but development sequences, and the vocabulary digest and fit
   sources are part of the ORB-BoVW fingerprint.
10. **Baselines share every stage but the descriptor.** All encoders implement `VisualEncoder`
    (L2-normalised float32 output), so caching, FAISS retrieval, saved-result format and
    metrics are the same code for every method.

## Phase 6: development selection, then one held-out evaluation

```mermaid
flowchart TD
    subgraph DEV[Development stage: KITTI 00, 05]
        D1[descriptors + causal retrieval<br/>every method, every variant] --> D2[variant per encoder<br/>max pooled retrieval mAP]
        D2 --> D3[top-5 geometric verification<br/>parallel, pair cache]
        D3 --> D4[one hypothesis per query<br/>score = cosine or inliers]
        D4 --> D5[pooled PR curve<br/>tau = argmax F1, tau at P = 1]
    end
    D2 --> SEL[(selection.json)]
    D5 --> THR[(thresholds.json)]
    subgraph TEST[Test stage: KITTI 02, 06, 08, 09]
        T1[same pipeline with the frozen variant] --> T2[metrics at the development tau]
    end
    SEL --> T1
    THR --> T2
    T2 --> OUT[test_metrics.csv / json, curves, predictions,<br/>test_timing.json, efficiency.csv]
```

11. **The test stage cannot tune.** `Benchmark.run_test` refuses to run without the development
    stage's `selection.json` and `thresholds.json` and rejects overlapping splits; it only
    applies them. Verification results are cached per (sequence, verification-config hash),
    so re-running never changes a decision.

## Phase 7: dashboard over cached results

```mermaid
flowchart LR
    R[(results/benchmark/kitti_stride5<br/>metrics, thresholds, curves,<br/>predictions, retrieval, pair cache)] --> S[ResultStore<br/>read-only, returns None if missing]
    K[(KITTI images + poses<br/>optional)] --> S
    S --> C[st.cache_data / st.cache_resource]
    C --> V1[Overview]
    C --> V2[Retrieval explorer]
    C --> V3[Trajectory]
    C --> V4[Geometric verification<br/>ORB + RANSAC recomputed on CPU,<br/>compared with the pair cache]
    C --> V5[Benchmark comparison]
```

12. **The dashboard never runs a neural network.** It reads Phase 6 outputs only. The single
    computation it does, ORB + RANSAC for one pair, is checked against the cached decision
    and the result of that check is shown.
13. **Missing inputs degrade, they do not crash.** Every `ResultStore` loader returns `None`
    when its file is absent, and each view states what is missing and which command makes it.

## Phase 8: integration

14. **Release figures come from saved outputs.** `visualization/phase8.py` reads the same
    `ResultStore` as the dashboard, the embedding cache and the pair cache; the five pairs it
    draws are re-verified and compared with the cache (`phase8_summary.json`, `matches_cache`).
15. **Reproducibility is checked by re-running, not assumed.** Re-running the test stage from
    caches reproduces `test_metrics.csv`, `test_metrics.json`, every prediction file and every
    retrieval array byte for byte; only wall-clock timing fields change.

