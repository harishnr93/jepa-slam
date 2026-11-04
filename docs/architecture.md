# Architecture

Status: Phase 3. This document grows with each phase.

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
    subgraph evaluation
        metrics[metrics.py<br/>roc_auc, recall_at_k]
        profiling[profiling.py<br/>Profiler]
    end
    subgraph visualization
        plots[plots.py]
        traj[trajectory.py]
        anim[animations.py]
        emb[embeddings.py + phase2.py]
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
