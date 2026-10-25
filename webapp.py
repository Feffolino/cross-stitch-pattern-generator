#!/usr/bin/env python3
"""Local web page for gridify: drop an image, move the sliders, see the chart live.

    python webapp.py            # opens http://127.0.0.1:8765 in the browser
    python webapp.py 9000 --no-browser

Runs only on this computer (127.0.0.1), standard library only.
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
import tempfile
import threading
import uuid
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import gridify

HERE = Path(__file__).resolve().parent
UPLOADS = Path(tempfile.mkdtemp(prefix="gridify-web-"))
MAX_UPLOAD = 40 * 1024 * 1024
PORT = 8765
LOCK = threading.Lock()  # numpy/rembg work one request at a time

# Background removal takes seconds; moving a slider must not redo it.
gridify.remove_background = lru_cache(maxsize=8)(gridify.remove_background)


def build_args(q: dict[str, str]):
    """gridify command-line options from the page's controls."""
    argv = ["x", "--page", q.get("page", "a4"), "--cell-mm", q.get("mm", "3")]
    # The page sends a width only for photos and drawings (charts keep their own grid).
    width = q.get("width", "")
    if q.get("kind") == "photo" or width:
        argv += ["--photo", width or "0"]
    if q.get("style") in ("linee", "colori"):
        argv += ["--style", q["style"]]
    if q.get("detail") in ("1", "2", "3"):
        argv += ["--detail", q["detail"]]
    if q.get("colors"):
        argv += ["--colors", q["colors"]]
    if q.get("overview") == "1":
        argv.append("--overview")
    if q.get("bg") == "remove":
        argv.append("--remove-bg")
    elif q.get("bg") == "keep":
        argv.append("--keep-bg")
    return gridify.build_parser().parse_args(argv)


@lru_cache(maxsize=16)
def layout(path: str, photo, style: str, remove_bg: bool, keep_bg: bool, page: str):
    """Cached analysis: background removal and grid detection are the slow part."""
    args = gridify.build_parser().parse_args(["x"])
    args.photo, args.style, args.remove_bg, args.keep_bg, args.page = photo, style, remove_bg, keep_bg, page
    notes: list[str] = []
    lay = gridify.analyze(Path(path), args, say=notes.append)
    return lay, notes


class Stale(Exception):
    """A newer preview of the same image arrived while this one was waiting."""


LATEST: dict[str, int] = {}  # image id -> newest preview request number seen


def make(q: dict[str, str]):
    path = UPLOADS / q["id"]
    if not path.is_file():
        raise ValueError("immagine non trovata: ricaricala")
    args = build_args(q)
    seq = int(q.get("seq", 0))
    if seq:
        LATEST[q["id"]] = max(LATEST.get(q["id"], 0), seq)
    with LOCK:
        # The page drops superseded requests, but the server would still work through
        # each of them in turn and the current one would wait behind them all.
        if seq and seq < LATEST[q["id"]]:
            raise Stale
        lay, notes = layout(str(path), args.photo, args.style, args.remove_bg, args.keep_bg, args.page)
        notes = list(notes)
        cells = gridify.chart_cells(lay, args, say=notes.append)
    return args, lay, cells, notes


def preview(q: dict[str, str]) -> dict:
    args, lay, cells, notes = make(q)
    rows, cols = cells.shape[:2]
    # Same look as the *_clean.png the command line writes (24 px cells); the browser
    # scales it down smoothly. Very wide charts get smaller cells to stay quick.
    args.px = max(6, min(24, 9000 // max(cols, 1)))
    args.numbers = True
    args.grid_lines = q.get("grid") if q.get("grid") in ("none", "major", "full") else "full"
    png = io.BytesIO()
    gridify.render_chart(cells, args).save(png, "PNG", compress_level=1)
    (w, h), fx, fy = gridify.plan_sheets(cols, rows, args)
    sheets = math.ceil(cols / fx) * math.ceil(rows / fy)
    return {
        "png": base64.b64encode(png.getvalue()).decode(),
        "cols": cols, "rows": rows,
        "colors": int(len({tuple(c) for c in cells.reshape(-1, 3)})),
        "kind": lay.kind,
        "sheets": sheets,
        "orientation": "verticale" if w < h else "orizzontale",
        "size_cm": [round(cols * args.cell_mm / 10, 1), round(rows * args.cell_mm / 10, 1)],
        "notes": notes,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # keep the console quiet
        pass

    def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data: dict, code: int = 200):
        self.send(code, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/preview":
                try:
                    self.send_json(preview(q))
                except Stale:
                    self.send_json({"stale": True})
            elif url.path == "/api/pdf":
                args, _, cells, _ = make(q)
                out = UPLOADS / f"{q['id']}.pdf"
                name = Path(q.get("name", "schema")).stem or "schema"
                gridify.write_pdf(out, cells, name, args)
                self.send(200, out.read_bytes(), "application/pdf",
                          {"Content-Disposition": f'attachment; filename="{name}_schema.pdf"'})
            else:
                self.send(404, b"not found", "text/plain")
        except Exception as e:  # report to the page instead of dropping the connection
            self.send_json({"error": str(e)}, 400)

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            return self.send(404, b"not found", "text/plain")
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= MAX_UPLOAD:
            return self.send_json({"error": "immagine vuota o troppo grande (max 40 MB)"}, 400)
        data = self.rfile.read(size)
        ext = Path(self.headers.get("X-Filename", "img.png")).suffix.lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            return self.send_json({"error": f"formato non supportato: {ext}"}, 400)
        name = uuid.uuid4().hex + ext
        (UPLOADS / name).write_bytes(data)
        self.send_json({"id": name})


def main() -> int:
    opts = [a for a in sys.argv[1:] if a.startswith("--")]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    port = int(nums[0]) if nums else PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Schemi punto croce: {url}  (Ctrl+C per chiudere)", flush=True)
    if "--no-browser" not in opts:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
