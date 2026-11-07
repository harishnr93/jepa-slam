"""Latent-Loop dashboard: explore cached Phase 3-6 results (no neural network is loaded).

Run from the repository root:

    streamlit run app/streamlit_app.py

Views are selected in the sidebar or by URL query parameters, e.g.
``?view=retrieval&seq=06&method=ijepa_gv&q=600`` (used for the README screenshots).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from latent_loop.dashboard.data import METHOD_ENCODER, ResultStore, overview_numbers, query_table
from latent_loop.visualization.evaluation import COLORS, LABELS

VIEWS = {
    "overview": "Overview",
    "retrieval": "Retrieval explorer",
    "trajectory": "Trajectory",
    "verification": "Geometric verification",
    "benchmark": "Benchmark comparison",
}
GREEN, RED, ORANGE, GREY = "#2f855a", "#c53030", "#dd6b20", "#a0aec0"

st.set_page_config(page_title="Latent-Loop dashboard", layout="wide")


def cli_args() -> argparse.Namespace:
    """``streamlit run app/streamlit_app.py -- --config X --set key=value ...``"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_known_args(sys.argv[1:])[0]


@st.cache_resource
def store() -> ResultStore:
    a = cli_args()
    return ResultStore(a.config, a.set)


@st.cache_resource
def dataset(seq: str):
    return store().dataset(seq)


@st.cache_data
def image(seq: str, frame_id: int) -> np.ndarray | None:
    ds = dataset(seq)
    return None if ds is None else ds.image_for_frame_id(int(frame_id))


@st.cache_data
def positions(seq: str) -> tuple[np.ndarray, np.ndarray] | None:
    ds = dataset(seq)
    if ds is None or not ds.has_poses:
        return None
    return ds.frame_ids, ds.poses()[:, :3, 3]


def qp(name: str, default: str) -> str:
    return str(st.query_params.get(name, default))


def label(m: str) -> str:
    return LABELS.get(m, m)


def show_image(seq: str, fid: int, caption: str) -> None:
    img = image(seq, fid) if fid >= 0 else None
    if img is None:
        st.info(f"{caption}\n\n(image unavailable: KITTI images not found)")
    else:
        st.image(img, caption=caption, width="stretch")


# ----------------------------------------------------------------------------- sidebar
S = store()
GT = S.config["ground_truth"]
R_POS = float(GT["spatial_threshold_m"])
R_NEG = float(GT.get("negative_radius_m", 25.0))  # pair_labels default (display only)
st.sidebar.title("Latent-Loop")
st.sidebar.caption("Cached results only. No neural model is loaded.")
if not S.available():
    msg = S.missing_message()
    st.error(f"{msg.what}.\n\nCreate them with:\n\n`{msg.how}`")
    st.stop()

seqs = S.sequences()
all_seqs = [s for split in ("test", "dev") for s in seqs.get(split, [])]
view_keys = list(VIEWS)
view = st.sidebar.radio(
    "View", view_keys, index=view_keys.index(qp("view", "overview")), format_func=VIEWS.get
)
seq = st.sidebar.selectbox(
    "Sequence",
    all_seqs,
    index=all_seqs.index(qp("seq", all_seqs[0])),
    format_func=lambda s: f"KITTI {s} ({S.split_of(s)})",
)
method = st.sidebar.selectbox(
    "Method", S.methods, index=S.methods.index(qp("method", "ijepa_gv")), format_func=label
)
thresholds = S.thresholds() or {}
thr = thresholds.get(method, {}).get("f1", np.inf)
st.sidebar.markdown(
    f"**Detection threshold** (chosen on development data): `{thr:.3f}` "
    f"({'RANSAC inliers' if method.endswith('_gv') else 'cosine similarity'})"
)
pred = S.predictions(method, seq)
result = S.retrieval(method, seq)

# ----------------------------------------------------------------------------- views
st.title(VIEWS[view])

if view == "overview":
    if pred is None:
        st.warning("No predictions for this sequence and method.")
    else:
        n = overview_numbers(pred, thr, S.row(method, seq), S.efficiency(), method)
        st.markdown(
            f"**KITTI {seq}** ({S.split_of(seq)} split), **{label(method)}**, encoder "
            f"`{METHOD_ENCODER[method]}` (selected variant: `{S.row(method, seq).variant}`)"
        )
        c = st.columns(5)
        c[0].metric("Processed frames", n["processed_frames"], help="every 5th frame")
        c[1].metric("Candidate loops", n["candidate_loops"], help="queries with a hypothesis")
        c[2].metric("Accepted loops", n["accepted_loops"], help="score >= development threshold")
        c[3].metric("Correct accepted", n["correct_accepted"])
        c[4].metric(
            "Revisit queries",
            n["revisit_queries"],
            help="ground truth: <= 5 m, >= 100 frames earlier",
        )
        c = st.columns(4)
        c[0].metric("Precision", f"{n.get('precision', float('nan')):.3f}")
        c[1].metric("Recall", f"{n.get('recall', float('nan')):.3f}")
        c[2].metric("F1", f"{n.get('f1', float('nan')):.3f}")
        c[3].metric(
            "End-to-end latency",
            f"{n.get('latency_ms', float('nan')):.0f} ms",
            help=f"encoder on {n.get('device', '?')}; retrieval and verification on CPU",
        )
        st.caption("Precision, recall and F1: query-level detection at the development threshold.")
        if result is not None:
            tbl = query_table(result, pred, thr)
            st.subheader("Queries")
            st.dataframe(tbl[tbl.eligible_refs > 0].round(3), width="stretch", height=320)

elif view == "retrieval":
    if result is None:
        st.warning("No saved retrieval result for this method and sequence.")
    else:
        ids = result.query_frame_ids
        valid = ids[result.eligible_counts > 0]
        q = int(qp("q", str(int(valid[len(valid) // 2]))))
        q = int(
            st.select_slider(
                "Query frame ID", options=list(valid), value=q if q in set(valid) else valid[0]
            )
        )
        i = int(np.searchsorted(ids, q))
        pos = positions(seq)
        index = {int(f): k for k, f in enumerate(ids)}
        pairs = S.pairs(seq)
        k_show = st.slider("Top-K shown", 1, result.candidate_ids.shape[1], 5)
        cols = st.columns([1.2, 3])
        with cols[0]:
            show_image(seq, q, f"Query #{q}")
            if pred is not None:
                p = pred.iloc[i]
                acc = np.isfinite(p.scores) and p.scores >= thr
                st.markdown(
                    f"**Detector ({label(method)})**: "
                    + (
                        f"loop to **#{int(p.candidate_ids)}** (score {p.scores:.3g}), "
                        + ("**correct**" if p.correct else "**wrong**")
                        if acc
                        else "no loop accepted"
                    )
                    + f"  \nQuery has a true revisit: **{bool(p.has_positive)}**"
                )
        with cols[1]:
            grid = st.columns(min(k_show, 3))
            for r in range(k_show):
                cid = int(result.candidate_ids[i, r])
                if cid < 0:
                    continue
                d = (
                    float(np.linalg.norm(pos[1][i, [0, 2]] - pos[1][index[cid], [0, 2]]))
                    if pos
                    else float("nan")
                )
                gt = (
                    "TRUE revisit"
                    if d <= R_POS
                    else (f"near ({R_POS:g}-{R_NEG:g} m)" if d <= R_NEG else "not a revisit")
                )
                ver = ""
                if pairs is not None and (q, cid) in pairs.index and r < 5:
                    v = pairs.loc[(q, cid)]
                    status = (
                        "checks passed"
                        if v.reason == "accepted"
                        else "rejected: " + str(v.reason).replace("_", " ")
                    )
                    ver = f"geometry: {status} ({int(v.n_inliers)} inliers)"
                elif r >= 5:
                    ver = "geometry: not run (rank > 5)"
                with grid[r % len(grid)]:
                    show_image(
                        seq,
                        cid,
                        f"rank {r + 1}: #{cid} | cos {result.scores[i, r]:.3f} | {d:.1f} m, {gt}",
                    )
                    st.caption(ver)

elif view == "trajectory":
    pos = positions(seq)
    if pos is None or pred is None:
        st.warning("Trajectory needs KITTI poses and saved predictions for this sequence.")
    else:
        ids, P = pos
        index = {int(f): k for k, f in enumerate(ids)}
        on = (pred.scores.fillna(-np.inf) >= thr).to_numpy()
        show_rejected = st.checkbox("Show rejected candidates (below threshold)", value=False)
        q = int(qp("q", "-1"))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=P[:, 0],
                y=P[:, 2],
                mode="lines",
                line=dict(color="#cbd5e0", width=2),
                name="ground-truth trajectory",
            )
        )
        for kind, mask, colr in (
            ("accepted loop, correct", on & pred.correct.to_numpy(), GREEN),
            ("accepted loop, false positive", on & ~pred.correct.to_numpy(), RED),
            ("rejected candidate", ~on & pred.scores.notna().to_numpy(), GREY),
        ):
            if kind == "rejected candidate" and not show_rejected:
                continue
            xs, ys = [], []
            for k in np.flatnonzero(mask):
                c = index[int(pred.candidate_ids.iloc[k])]
                xs += [P[k, 0], P[c, 0], None]
                ys += [P[k, 2], P[c, 2], None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=colr, width=2 if colr != GREY else 0.6),
                    name=f"{kind} ({int(mask.sum())})",
                )
            )
        miss = np.flatnonzero(pred.has_positive.to_numpy() & ~(on & pred.correct.to_numpy()))
        fig.add_trace(
            go.Scatter(
                x=P[miss, 0],
                y=P[miss, 2],
                mode="markers",
                marker=dict(color=ORANGE, symbol="x", size=7),
                name=f"revisit without a correct loop ({miss.size})",
            )
        )
        if q in index:
            k = index[q]
            fig.add_trace(
                go.Scatter(
                    x=[P[k, 0]],
                    y=[P[k, 2]],
                    mode="markers",
                    marker=dict(color="black", size=12),
                    name=f"query #{q}",
                )
            )
            cands = (
                [int(c) for c in result.candidate_ids[k, :5] if c >= 0]
                if result is not None
                else []
            )
            fig.add_trace(
                go.Scatter(
                    x=[P[index[c], 0] for c in cands],
                    y=[P[index[c], 2] for c in cands],
                    mode="markers",
                    marker=dict(color="#2b6cb0", size=9, symbol="square"),
                    name="its top-5 candidates",
                )
            )
        fig.update_layout(
            height=650,
            xaxis_title="x [m]",
            yaxis_title="z [m]",
            yaxis_scaleanchor="x",
            legend=dict(orientation="h", y=-0.12),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Ground-truth poses are used only to draw and label; they never select candidates."
        )

elif view == "verification":
    vc = S.verification_config()
    pairs = S.pairs(seq)
    if result is None or vc is None or pairs is None:
        st.warning("Needs the saved retrieval result and the benchmark pair cache.")
    else:
        from latent_loop.verification import (
            VerificationConfig,
            raw_nearest_matches,
            verify_features,
        )  # noqa: PLC0415
        from latent_loop.verification.orb import extract_orb  # noqa: PLC0415
        from latent_loop.visualization.matches import PairExample, inlier_canvas, tentative_canvas  # noqa: PLC0415

        ids = result.query_frame_ids
        verified = sorted(
            {q for q, _ in pairs.index} & set(ids[result.eligible_counts > 0].tolist())
        )
        q = int(qp("q", str(verified[len(verified) // 2])))
        q = int(
            st.select_slider(
                "Query frame ID", options=verified, value=q if q in set(verified) else verified[0]
            )
        )
        i = int(np.searchsorted(ids, q))
        cands = [int(c) for c in result.candidate_ids[i, :5] if c >= 0]
        c = int(
            st.radio(
                "Candidate (retrieval rank)",
                cands,
                horizontal=True,
                format_func=lambda x: f"#{x} (rank {cands.index(x) + 1})",
            )
        )
        ds = dataset(seq)
        cfg = VerificationConfig.from_dict(vc["config"])
        if ds is None:
            st.warning("KITTI images not found; showing the cached decision only.")
            st.json(pairs.loc[(q, c)].to_dict())
        else:
            img_q, img_r = ds.image_for_frame_id(q), ds.image_for_frame_id(c)
            fq, fr = extract_orb(img_q, cfg.orb), extract_orb(img_r, cfg.orb)
            res = verify_features(fq, fr, ds.get_calibration(), cfg, keep_details=True)
            cached = pairs.loc[(q, c)]
            same = res.reason == cached.reason and res.n_inliers == int(cached.n_inliers)
            accepted = res.accepted and res.n_inliers >= thresholds.get(
                f"{METHOD_ENCODER[method]}_gv", {}
            ).get("f1", np.inf)
            m = st.columns(6)
            m[0].metric("Matches (ratio + mutual)", res.n_matches)
            m[1].metric("RANSAC inliers", res.n_inliers)
            m[2].metric("Inlier ratio", f"{res.inlier_ratio:.2f}")
            m[3].metric("Coverage", f"{res.coverage:.2f}")
            m[4].metric(
                "Geometric checks",
                "passed" if res.reason == "accepted" else res.reason.replace("_", " "),
                help="matches, inliers, ratio and coverage checks; the loop decision also "
                "needs inliers >= the threshold",
            )
            m[5].metric(
                "Loop accepted",
                "YES" if accepted else "no",
                help="checks passed and inliers >= the GV method's development threshold",
            )
            st.caption(
                f"Recomputed on the CPU (ORB + essential-matrix RANSAC); equals cached decision: "
                f"**{same}**. Degeneracy: `{res.degenerate or 'none'}`; homography/essential ratio "
                f"{res.h_ratio:.2f}."
            )
            ex = PairExample(
                q,
                c,
                img_q,
                img_r,
                fq,
                fr,
                res,
                raw_nearest_matches(fq, fr),
                float(result.scores[i, cands.index(c)]),
                cands.index(c) + 1,
            )
            t1, t2 = st.tabs(
                ["Inliers (green) and rejected matches (red)", "All tentative matches (yellow)"]
            )
            t1.image(inlier_canvas(ex), caption="query (top) / candidate (bottom)", width="stretch")
            t2.image(
                tentative_canvas(ex), caption="ratio-tested mutual ORB matches", width="stretch"
            )

elif view == "benchmark":
    split = st.radio(
        "Split", ["test", "dev"], horizontal=True, index=["test", "dev"].index(qp("split", "test"))
    )
    df = S.metrics(split)
    eff = S.efficiency()
    if df is None:
        st.warning(f"No {split} metrics.")
    else:
        pooled = df[df.sequence == "pooled"].set_index("method").reindex(S.methods)
        metric_names = {
            "det_f1": "F1",
            "det_precision": "Precision",
            "det_recall": "Recall",
            "recall_at_1": "Recall@1",
            "map": "mAP",
            "detection_ap": "Detection AP",
        }
        fig = go.Figure()
        for m in S.methods:
            fig.add_trace(
                go.Bar(
                    name=label(m),
                    x=list(metric_names.values()),
                    y=[pooled.loc[m, k] for k in metric_names],
                    marker_color=COLORS.get(m),
                )
            )
        fig.update_layout(
            barmode="group",
            height=420,
            yaxis_range=[0, 1],
            yaxis_title="value (axis from 0)",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"Pooled {split} metrics at the development thresholds",
        )
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(2)
        curves = S.curves(split)
        pr = go.Figure()
        for m, c in curves.items():
            pr.add_trace(
                go.Scatter(
                    x=np.r_[0, c["recall"]],
                    y=np.r_[1, c["precision"]],
                    mode="lines",
                    line=dict(
                        color=COLORS.get(m),
                        dash="dash" if m.endswith("_gv") else "solid",
                        shape="hv",
                    ),
                    name=f"{label(m)} (AP {float(c['ap']):.3f})",
                )
            )
        pr.update_layout(
            height=420,
            xaxis_title="recall",
            yaxis_title="precision",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1.02],
            title="Detection precision-recall",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        cols[0].plotly_chart(pr, width="stretch")
        if eff is not None:
            e = eff.set_index("method").reindex(S.methods)
            lat = go.Figure()
            for part, colr in (
                ("encoder_ms_per_frame", "#4a5568"),
                ("retrieval_ms_per_query", "#a0aec0"),
                ("verification_ms_per_query", ORANGE),
            ):
                lat.add_trace(
                    go.Bar(
                        name=part.replace("_", " "),
                        y=[label(m) for m in S.methods],
                        x=e[part],
                        orientation="h",
                        marker_color=colr,
                    )
                )
            lat.update_layout(
                barmode="stack",
                height=420,
                xaxis_title="ms per query",
                title="End-to-end latency (test)",
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.2),
            )
            cols[1].plotly_chart(lat, width="stretch")
        show = [
            "method",
            "sequence",
            "positive_queries",
            "recall_at_1",
            "map",
            "detection_ap",
            "threshold",
            "det_precision",
            "det_recall",
            "det_f1",
            "det_tp",
            "det_fp",
            "det_fn",
            "det_fp_per_1000",
        ]
        st.subheader("All rows")
        st.dataframe(df[[c for c in show if c in df]].round(3), width="stretch", height=360)
        st.caption(
            "Thresholds and preprocessing were selected on the development split (00, 05) only."
        )

st.sidebar.markdown("---")
st.sidebar.caption(f"Results: `{S.root}`")
