#!/usr/bin/env python
"""Capture screenshots of the running dashboard and assemble the walkthrough GIF.

Usage (dashboard already running: streamlit run app/streamlit_app.py):
    python scripts/capture_dashboard.py \
        --chrome "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
    python scripts/capture_dashboard.py --chrome /usr/bin/chromium --delay 20

Headless Chrome takes its screenshot at the page's load event, which comes before
Streamlit has rendered over its websocket. This script therefore serves a small wrapper
page that embeds the dashboard in an iframe together with one image whose response is
delayed by ``--delay`` seconds, so the load event (and the screenshot) waits for the real
rendering. Works with a Linux Chrome/Chromium or, from WSL, the Windows Chrome binary
(paths are converted with ``wslpath``). Writes assets/phase_07_dashboard/*.png and
dashboard_walkthrough.gif.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

from latent_loop.utils.config import resolve_path
from latent_loop.visualization.animations import save_gif

PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
# (file stem, query string, page height in px, seconds the walkthrough GIF shows it)
SHOTS = [
    ("overview", "view=overview&seq=06&method=ijepa_gv", 1150, 2.5),
    ("retrieval_explorer", "view=retrieval&seq=06&method=ijepa_gv&q=600", 1350, 2.5),
    ("retrieval_explorer_aliasing", "view=retrieval&seq=02&method=ijepa&q=4570", 1350, 2.0),
    ("trajectory", "view=trajectory&seq=02&method=dinov2_gv&q=4570", 1150, 2.5),
    ("trajectory_08", "view=trajectory&seq=08&method=ijepa_gv", 1150, 2.0),
    ("verification", "view=verification&seq=06&method=ijepa_gv&q=600", 1400, 2.5),
    ("benchmark", "view=benchmark&split=test", 1700, 3.0),
]


def wrapper_server(port: int) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            if url.path == "/delay":
                time.sleep(float(q.get("s", ["15"])[0]))
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(PIXEL)
                return
            if not all(k in q for k in ("u", "w", "h", "s")):  # e.g. /favicon.ico
                self.send_response(404)
                self.end_headers()
                return
            target, w, h, s = (q[k][0] for k in ("u", "w", "h", "s"))
            html = (
                '<html><body style="margin:0;overflow:hidden">'
                f'<iframe src="{target}" width="{w}" height="{h}" style="border:0"></iframe>'
                f'<img src="/delay?s={s}" style="position:absolute;left:-10px"></body></html>'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def native_path(path: Path, chrome: str) -> str:
    if chrome.lower().endswith(".exe"):  # Windows Chrome driven from WSL
        return subprocess.run(
            ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True
        ).stdout.strip()
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chrome", required=True, help="Chrome/Chromium binary")
    parser.add_argument("--base-url", default="http://localhost:8501/")
    parser.add_argument("--out", default="assets/phase_07_dashboard")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--delay", type=float, default=25.0, help="seconds to let each view render")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--gif-width", type=int, default=900)
    parser.add_argument("--gif-fps", type=float, default=8.0)
    args = parser.parse_args()
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    server = wrapper_server(args.port)
    report = []
    try:
        for stem, query, height, _ in SHOTS:
            target = urllib.parse.quote(f"{args.base_url}?{query}", safe="")
            page = (
                f"http://localhost:{args.port}/?u={target}&w={args.width}&h={height}&s={args.delay}"
            )
            png = out / f"{stem}.png"
            png.unlink(missing_ok=True)  # a stale file must never pass as a new capture
            t0 = time.perf_counter()
            for _attempt in range(2):  # Chrome occasionally exits at once on first launch
                proc = subprocess.run(
                    [
                        args.chrome,
                        "--headless=new",
                        "--disable-gpu",
                        "--hide-scrollbars",
                        f"--window-size={args.width},{height}",
                        f"--screenshot={native_path(png, args.chrome)}",
                        page,
                    ],
                    capture_output=True,
                    timeout=args.delay + 120,
                )
                if png.is_file() and time.perf_counter() - t0 >= args.delay:
                    break
                png.unlink(missing_ok=True)
                t0 = time.perf_counter()
            if not png.is_file():
                raise RuntimeError(
                    f"No screenshot for {stem} (Chrome exit {proc.returncode}): "
                    f"{proc.stderr.decode(errors='replace')[-500:]}"
                )
            report.append(
                {
                    "shot": stem,
                    "query": query,
                    "bytes": png.stat().st_size,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            print(report[-1])
    finally:
        server.shutdown()

    frames = []
    for stem, *_ in SHOTS:
        img = Image.open(out / f"{stem}.png").convert("RGB")
        img = img.resize(
            (args.gif_width, round(img.height * args.gif_width / img.width)), Image.LANCZOS
        )
        frames.append(np.asarray(img))
    height = max(f.shape[0] for f in frames)
    padded = [
        np.pad(f, ((0, height - f.shape[0]), (0, 0), (0, 0)), constant_values=255) for f in frames
    ]
    seq = [
        f
        for f, (_, _, _, hold) in zip(padded, SHOTS, strict=True)
        for _ in range(int(hold * args.gif_fps))
    ]
    gif = save_gif(seq, out / "dashboard_walkthrough.gif", fps=args.gif_fps)
    summary = {
        "shots": report,
        "gif": gif.name,
        "gif_frames": len(seq),
        "gif_mb": round(gif.stat().st_size / 1e6, 2),
    }
    (out / "capture_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
