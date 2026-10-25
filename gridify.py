#!/usr/bin/env python3
"""Rileva la griglia di un'immagine pixel art e disegna le linee sui bordi dei quadrati.

Pensato per immagini prese online (ingrandite, sfocate, JPEG) da trasformare in
schemi per il punto croce / ricamo.

Esempi:
    python gridify.py immagine.png
    python gridify.py *.jpg --out risultati
    python gridify.py immagine.png --mode overlay
    python gridify.py immagine.png --cell 12.5 --colors 16 --svg
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pdfchart import Document, Page


@dataclass
class Axis:
    """Cells along one axis: cell k spans [start + k*pitch, start + (k+1)*pitch)."""
    start: float
    pitch: float
    count: int
    score: float = 1.0

    def line(self, k: int) -> float:
        return self.start + k * self.pitch


# ---------------------------------------------------------------- detection

def load_rgb(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path))  # phone photos are stored sideways
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def edge_profile(img: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Average color jump across each vertical (x) or horizontal (y) pixel boundary.

    Returns weights and boundary coordinates (boundary between pixel i and i+1 is i+1).
    """
    a = img.astype(np.float32)
    if axis == "x":
        d = np.abs(np.diff(a, axis=1)).sum(axis=2).mean(axis=0)
    else:
        d = np.abs(np.diff(a, axis=0)).sum(axis=2).mean(axis=1)
    # Remove the noise floor (JPEG artifacts, gradients): most boundaries are not cell edges.
    d = np.clip(d - np.median(d), 0, None)
    pos = np.arange(1, len(d) + 1, dtype=np.float64)
    keep = d > 0
    return d[keep].astype(np.float64), pos[keep]


def lattice_fit(w: np.ndarray, pos: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Normalized phasor sum for each candidate period.

    Magnitude is 1 when every edge falls on one lattice of that period, ~0 when edges
    are unrelated to it. The angle gives the lattice offset.
    """
    out = np.empty(len(periods), np.complex128)
    for s in range(0, len(periods), 256):
        f = 1.0 / periods[s:s + 256]
        out[s:s + 256] = np.exp(-2j * np.pi * np.outer(f, pos)) @ w
    return out / w.sum()


def active_span(w: np.ndarray, pos: np.ndarray) -> float:
    c = np.cumsum(w) / w.sum()
    return pos[np.searchsorted(c, 0.95)] - pos[np.searchsorted(c, 0.05)]


def score(profiles, periods):
    return np.mean([np.abs(lattice_fit(w, p, periods)) for w, p in profiles], axis=0)


def refine(profiles, p0: float, length: int) -> tuple[float, float]:
    f0, df = 1.0 / p0, 1.0 / (8 * length)
    periods = 1.0 / np.linspace(f0 - 2 * df, f0 + 2 * df, 201)
    s = score(profiles, periods)
    i = int(np.argmax(s))
    return float(periods[i]), float(s[i])


def detect_period(profiles, length: int, pmin: float, pmax: float,
                  harmonic_tol: float) -> tuple[float, float]:
    """Find the cell size shared by the given edge profiles.

    Halves of the true size (p/2, p/3...) fit the edges just as well, so among strong
    candidates the largest one wins. A candidate must also repeat at least 4 times
    inside the region that has edges, or a lone sprite's outline would look like one
    giant cell.
    """
    span = min(active_span(w, p) for w, p in profiles)
    pmax = min(pmax, span / 4)
    if pmax <= pmin:
        raise ValueError("troppi pochi bordi per rilevare la griglia")
    df = 1.0 / (8 * length)
    freqs = np.arange(1.0 / pmax, 1.0 / pmin + df, df)
    periods = 1.0 / freqs
    s = score(profiles, periods)
    peaks = np.where((s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    if len(peaks) == 0:
        raise ValueError("nessuna griglia trovata")
    best = s[peaks].max()
    i0 = peaks[s[peaks] >= harmonic_tol * best].min()  # lowest frequency = largest cell
    return refine(profiles, float(periods[i0]), length)


def boundary_offset(profile, pitch: float) -> float:
    w, pos = profile
    c = lattice_fit(w, pos, np.array([pitch]))[0]
    return (-np.angle(c) / (2 * np.pi) * pitch) % pitch


def make_axis(offset: float, pitch: float, length: int, score_: float = 1.0) -> Axis:
    # First cell starts within half a cell of the image edge; a cell counts if at
    # least half of it is inside the image.
    start = ((offset + pitch / 2) % pitch) - pitch / 2
    count = int(math.floor((length - start) / pitch + 0.5))
    return Axis(start, pitch, count, score_)


def detect_grid(img: np.ndarray, args) -> tuple[Axis, Axis]:
    H, W = img.shape[:2]
    if args.grid:
        cols, rows = args.grid
        return Axis(0.0, W / cols, cols), Axis(0.0, H / rows, rows)

    prof_x, prof_y = edge_profile(img, "x"), edge_profile(img, "y")
    for name, (w, _) in (("orizzontale", prof_x), ("verticale", prof_y)):
        if w.size == 0 or w.sum() == 0:
            raise ValueError(f"nessun bordo in direzione {name}")
    L = max(W, H)

    if args.cell:
        px = py = args.cell
        sx = sy = 1.0
    elif args.nonsquare:
        px, sx = detect_period([prof_x], W, args.min_cell, args.max_cell, args.harmonic_tol)
        py, sy = detect_period([prof_y], H, args.min_cell, args.max_cell, args.harmonic_tol)
    else:
        p, _ = detect_period([prof_x, prof_y], L, args.min_cell, args.max_cell, args.harmonic_tol)
        # Each axis may drift slightly (non-uniform resize); keep it if close.
        px, sx = refine([prof_x], p, W)
        py, sy = refine([prof_y], p, H)
        if abs(px - py) / p > 0.015:
            print(f"  attenzione: lato x={px:.3f} e y={py:.3f} diversi; "
                  f"se i quadrati non sono quadrati usa --nonsquare", file=sys.stderr)
            px = py = p

    if args.offset:
        ox, oy = args.offset
    else:
        ox, oy = boundary_offset(prof_x, px), boundary_offset(prof_y, py)
    return make_axis(ox, px, W, sx), make_axis(oy, py, H, sy)


def drawn_grid_extent(img: np.ndarray, gx: Axis, gy: Axis,
                      edge: float = 20.0) -> tuple[range, range] | None:
    """Cell rows/columns covered by grid lines already drawn in the picture.

    Charts found online often have their grid drawn in and sit inside screenshot
    padding. Inside the chart nearly every lattice line shows an edge, in the
    padding none does. Returns None when the picture has no drawn grid.
    """
    a = img.astype(np.float32)
    dx = np.abs(np.diff(a, axis=1)).sum(axis=2)  # edge before column x+1
    dy = np.abs(np.diff(a, axis=0)).sum(axis=2)

    def coverage(d, along: Axis, across: Axis, transpose: bool, cells: range):
        # For each cell band of `across`, fraction of `along` lines bordering `cells`
        # that show an edge.
        length = d.shape[0 if transpose else 1]
        idx = [int(round(along.line(k))) - 1 for k in range(cells.start + 1, cells.stop)]
        idx = [i for i in idx if 1 <= i < length - 2]
        if not idx:
            return np.zeros(across.count)
        out = []
        for b0, b1 in cell_ranges(across, d.shape[1 if transpose else 0], 0.6):
            band = d[:, b0:b1].T if transpose else d[b0:b1]
            prof = band.mean(axis=0)
            hits = [max(prof[i - 1], prof[i], prof[i + 1]) > edge for i in idx]
            out.append(np.mean(hits))
        return np.array(out)

    def span(cov, threshold):
        on = np.nonzero(cov > threshold)[0]
        return range(on[0], on[-1] + 1) if len(on) else range(0)

    # Padding on one axis dilutes the other axis' coverage, so measure each axis
    # only inside the other's rough extent.
    rows = coverage(dx, gx, gy, False, range(gx.count))
    cols = coverage(dy, gy, gx, True, range(gy.count))
    r, c = span(rows, 0.5 * rows.max()), span(cols, 0.5 * cols.max())
    if not r or not c:
        return None
    rows = coverage(dx, gx, gy, False, c)
    cols = coverage(dy, gy, gx, True, r)
    if np.median(rows[r.start:r.stop]) < 0.8 or np.median(cols[c.start:c.stop]) < 0.8:
        return None
    r, c = span(rows, 0.5), span(cols, 0.5)
    return (r, c) if r and c else None


# ------------------------------------------------------------------- colors

def cell_ranges(ax: Axis, length: int, inner: float) -> list[tuple[int, int]]:
    half = max(0.5, inner * ax.pitch / 2)
    out = []
    for k in range(ax.count):
        c = ax.start + (k + 0.5) * ax.pitch
        a = min(max(int(round(c - half)), 0), length - 1)
        b = min(max(int(round(c + half)), a + 1), length)
        out.append((a, b))
    return out


def cell_texture(img: np.ndarray, gx: Axis, gy: Axis) -> float:
    """Median color spread inside cell centres (sampling every 3rd cell).

    Pixel art, even blurred or JPEG'd, stays below ~20; photos of beads or
    diamond-painting drills (shaded spheres with dark gaps) score 60+.
    """
    H, W = img.shape[:2]
    a = img.astype(np.float32)
    xs, ys = cell_ranges(gx, W, 0.5)[::3], cell_ranges(gy, H, 0.5)[::3]
    return float(np.median([a[y0:y1, x0:x1].reshape(-1, 3).std(0).sum()
                            for y0, y1 in ys for x0, x1 in xs]))


def sample_cells(img: np.ndarray, gx: Axis, gy: Axis, inner: float) -> np.ndarray:
    """One color per cell: per-channel median of the cell's central area (skips blur)."""
    H, W = img.shape[:2]
    xs, ys = cell_ranges(gx, W, inner), cell_ranges(gy, H, inner)
    out = np.zeros((gy.count, gx.count, 3), np.uint8)
    for j, (y0, y1) in enumerate(ys):
        for i, (x0, x1) in enumerate(xs):
            out[j, i] = np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return out


def _nearest(u: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = ((u[:, None, :] - centers[None]) ** 2).sum(-1)
    return d.argmin(1), np.sqrt(d.min(1))


def _kmeans(u, w, centers, iters=20):
    for _ in range(iters):
        lab, _ = _nearest(u, centers)
        new = centers.copy()
        for j in range(len(centers)):
            m = lab == j
            if m.any():
                new[j] = (u[m] * w[m, None]).sum(0) / w[m].sum()
        done = np.abs(new - centers).max() < 0.1
        centers = new
        if done:
            break
    lab, dist = _nearest(u, centers)
    return centers, lab, dist


def build_palette(cells: np.ndarray, tol: float, n_colors: int | None = None,
                  max_colors: int = 64) -> np.ndarray:
    """Group cell colors with k-means so JPEG/blur noise collapses onto clean colors.

    With n_colors the palette has exactly that many colors. Otherwise colors are
    added (starting from the most frequent, then the cell farthest from every
    color so far) until nearly every cell is within `tol` of its color.
    """
    flat = cells.reshape(-1, 3).astype(np.float64)
    u, inv, w = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    target = min(n_colors, len(u)) if n_colors else max_colors
    centers = u[[np.argmax(w)]]
    while True:
        centers, lab, dist = _kmeans(u, w, centers)
        if len(centers) >= min(target, len(u)):
            break
        if not n_colors and dist.max() <= 2 * tol and w[dist > tol].sum() <= 0.005 * w.sum():
            break
        centers = np.vstack([centers, u[np.argmax(dist)]])
    if not n_colors:
        # Outlier seeds can split one real color in near-twins; fold them back together.
        while len(centers) > 1:
            d = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= tol:
                break
            centers = np.delete(centers, j, axis=0)
            centers, lab, dist = _kmeans(u, w, centers)
    pal = np.round(centers).clip(0, 255).astype(np.uint8)
    return pal[lab][inv].reshape(cells.shape)


def fold_blends(cells: np.ndarray, rare: float = 0.03, tol: float = 12.0) -> np.ndarray:
    """Merge rare in-between colors into the main colors they are a mix of.

    Cells half covered by a stitch, or blurred edges, come out as a mix of their two
    neighbours (grey between black and white). A rare color lying on the Lab line
    between two common colors goes to the nearer one; a real accent color (red
    text, an eye highlight) is off that line and stays.
    """
    flat = cells.reshape(-1, 3)
    colors, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    share = counts / counts.sum()
    main = np.nonzero(share >= rare)[0]
    if len(main) < 2:
        return cells
    lab = rgb_to_lab(colors.astype(np.float64))
    target = np.arange(len(colors))
    for i in np.nonzero(share < rare)[0]:
        best = None
        for a in main:
            for b in main:
                if a >= b:
                    continue
                d = lab[b] - lab[a]
                t = np.clip((lab[i] - lab[a]) @ d / (d @ d), 0, 1)
                off = np.linalg.norm(lab[a] + t * d - lab[i])
                if off <= tol and (best is None or off < best[0]):
                    best = (off, a if t < 0.5 else b)
        if best:
            target[i] = best[1]
    return colors[target][inv].reshape(cells.shape)


def drop_edge_text(cells: np.ndarray, rare: float = 0.03, band: float = 0.15) -> tuple[np.ndarray, bool]:
    """Remove a watermark or site name written over the chart near one edge.

    Such text comes out as a handful of rare colors (anti-aliased letters) that all
    sit in a strip along one edge. Those cells take the chart's most common color.
    Rare colors spread over the chart (an eye, a red mouth) are left alone.
    """
    flat = cells.reshape(-1, 3)
    colors, inv, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    share = counts / counts.sum()
    if (share >= rare).sum() < 2 or (share < rare).sum() == 0:
        return cells, False
    odd = (share < rare)[inv.reshape(cells.shape[:2])]
    ys, xs = np.nonzero(odd)
    H, W = odd.shape
    in_strip = (ys.min() >= H * (1 - band) or ys.max() < H * band or
                xs.min() >= W * (1 - band) or xs.max() < W * band)
    if not in_strip:
        return cells, False
    out = cells.copy()
    out[odd] = colors[np.argmax(counts)]
    return out, True


def trim_background(cells: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Drop outer rows/columns made only of the most common border color."""
    border = np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    bg = colors[np.argmax(counts)]
    is_bg = np.all(cells == bg, axis=2)
    rows = np.where(~is_bg.all(axis=1))[0]
    cols = np.where(~is_bg.all(axis=0))[0]
    if len(rows) == 0:
        return cells, 0, 0
    return cells[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], int(cols[0]), int(rows[0])


# ---------------------------------------------------------------- rendering

def parse_color(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError(f"colore non valido: {s} (usa es. #404040)")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def draw_grid(d: ImageDraw.ImageDraw, xs: list[int], ys: list[int], args) -> None:
    """Draw lines at pixel positions xs / ys. Every `major`-th line (and the border) is thick."""
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]

    def is_major(k, n):
        return k == 0 or k == n - 1 or (args.major and k % args.major == 0)

    for thick in (False, True):  # thick lines on top
        w = args.major_width if thick else args.line_width
        col = args.major_color if thick else args.line_color
        for k, x in enumerate(xs):
            if is_major(k, len(xs)) == thick:
                d.rectangle([x - w // 2, y0 - args.major_width // 2,
                             x - w // 2 + w - 1, y1 + args.major_width // 2], fill=col)
        for k, y in enumerate(ys):
            if is_major(k, len(ys)) == thick:
                d.rectangle([x0 - args.major_width // 2, y - w // 2,
                             x1 + args.major_width // 2, y - w // 2 + w - 1], fill=col)


def draw_numbers(d: ImageDraw.ImageDraw, xs: list[int], ys: list[int], margin: int, args) -> None:
    if not args.major:
        return
    font = ImageFont.load_default(size=max(9, int(margin * 0.42)))
    for k, x in enumerate(xs):
        if k and k % args.major == 0:
            d.text((x, margin // 2), str(k), fill=(0, 0, 0), font=font, anchor="mm")
    for k, y in enumerate(ys):
        if k and k % args.major == 0:
            d.text((margin // 2, y), str(k), fill=(0, 0, 0), font=font, anchor="mm")


def _runs(mask: np.ndarray):
    """Yield (start, stop, value) for runs of equal values in a 1-D bool array."""
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            yield start, i, bool(mask[start])
            start = i


def grid_segments(cells: np.ndarray, major: int, x0: int = 0, y0: int = 0):
    """Grid line pieces in cell units: (vertical, k, a, b, thick, on_dark).

    Vertical line k runs from row a to row b (horizontal: column a to b). A piece is
    `on_dark` when the cells on both sides are dark, so it can be drawn light and
    stay visible. x0/y0 = global index of the first column/row, for thick lines
    every `major` cells when the chart is split in sheets.
    """
    rows, cols = cells.shape[:2]
    lum = cells[..., 0] * 0.299 + cells[..., 1] * 0.587 + cells[..., 2] * 0.114
    dark = lum < 90
    for vertical, n, g0, strip in ((True, cols, x0, lambda k: dark[:, k]),
                                   (False, rows, y0, lambda k: dark[k, :])):
        for k in range(n + 1):
            thick = k in (0, n) or bool(major and (g0 + k) % major == 0)
            both = strip(max(k - 1, 0)) & strip(min(k, n - 1))
            for a, b, on_dark in _runs(both):
                yield vertical, k, a, b, thick, on_dark


def segment_style(thick: bool, on_dark: bool, args):
    if thick:
        return (args.major_color_dark if on_dark else args.major_color), args.major_width
    return (args.line_color_dark if on_dark else args.line_color), args.line_width


def render_chart(cells: np.ndarray, args) -> Image.Image:
    rows, cols = cells.shape[:2]
    px = args.px
    margin = max(2 * px, 36) if args.numbers else args.major_width + 2
    canvas = Image.new("RGB", (cols * px + 2 * margin, rows * px + 2 * margin), "white")
    art = Image.fromarray(cells).resize((cols * px, rows * px), Image.NEAREST)
    canvas.paste(art, (margin, margin))
    d = ImageDraw.Draw(canvas)
    segs = sorted(grid_segments(cells, args.major), key=lambda s: s[4])  # thick on top
    grid = getattr(args, "grid_lines", "full")  # "full", "major" (every --major only) or "none"
    for vertical, k, a, b, thick, on_dark in segs:
        if grid == "none" or (grid == "major" and not thick):
            continue
        col, w = segment_style(thick, on_dark, args)
        p = margin + k * px - w // 2
        lo, hi = margin + a * px, margin + b * px
        if vertical:
            d.rectangle([p, lo, p + w - 1, hi], fill=col)
        else:
            d.rectangle([lo, p, hi, p + w - 1], fill=col)
    if args.numbers:
        xs = [margin + k * px for k in range(cols + 1)]
        ys = [margin + k * px for k in range(rows + 1)]
        draw_numbers(d, xs, ys, margin, args)
    return canvas


def render_overlay(img: np.ndarray, gx: Axis, gy: Axis, args) -> Image.Image:
    H, W = img.shape[:2]
    margin = 40 if args.numbers else args.major_width + 2
    canvas = Image.new("RGB", (W + 2 * margin, H + 2 * margin), "white")
    canvas.paste(Image.fromarray(img), (margin, margin))
    d = ImageDraw.Draw(canvas)
    xs = [margin + int(round(gx.line(k))) for k in range(gx.count + 1)]
    ys = [margin + int(round(gy.line(k))) for k in range(gy.count + 1)]
    draw_grid(d, xs, ys, args)
    if args.numbers:
        draw_numbers(d, xs, ys, margin, args)
    return canvas


def color_runs(cells: np.ndarray):
    """Yield (row, first_col, run_length, rgb) for horizontal runs of equal color."""
    rows, cols = cells.shape[:2]
    for r in range(rows):
        c = 0
        while c < cols:
            e = c
            while e + 1 < cols and np.array_equal(cells[r, e + 1], cells[r, c]):
                e += 1
            yield r, c, e - c + 1, tuple(int(v) for v in cells[r, c])
            c = e + 1


def hex_color(c) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


def write_svg(path: Path, cells: np.ndarray, args) -> None:
    """Vector chart (cells merged in horizontal runs + grid lines), editable in Figma."""
    rows, cols = cells.shape[:2]
    s = args.px
    W, H = cols * s, rows * s
    hexc = hex_color
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" shape-rendering="crispEdges">', '<g id="celle">']
    for r, c, n, rgb in color_runs(cells):
        out.append(f'<rect x="{c * s}" y="{r * s}" width="{n * s}" height="{s}" fill="{hexc(rgb)}"/>')
    out.append("</g>")
    segs = list(grid_segments(cells, args.major))
    for thick in (False, True):
        out.append(f'<g id="{"linee-spesse" if thick else "linee"}">')
        for vertical, k, a, b, t, on_dark in segs:
            if t != thick:
                continue
            col, width = segment_style(t, on_dark, args)
            v = k * s
            xy = (v, a * s, v, b * s) if vertical else (a * s, v, b * s, v)
            out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" ' % xy
                       + f'stroke="{hexc(col)}" stroke-width="{width}"/>')
        out.append("</g>")
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------- PDF

PAGE_SIZES = {"a4": (210.0, 297.0), "a3": (297.0, 420.0)}
PHOTO_WIDTHS = (80, 100, 120, 140, 160)  # chart widths tried for a photo, simplest first
TEXT_CELLS = 5.5        # letter height in cells for text in a drawing to be readable when stitched
TEXT_MAX_WIDTH = 450    # never widen a drawing beyond this many cells just for its text
LINE_PX = 6             # working pixels per cell for the line-art style
# line-art detail level -> (line blur sigma px, dark fill as share of Otsu, edge strength, cell ink cover)
LINE_DETAIL = {1: (2.0, 0.60, 0.035, 0.35), 2: (1.6, 0.75, 0.025, 0.30), 3: (1.2, 0.90, 0.018, 0.25)}
PHOTO_MAX_COLORS = 8
PHOTO_COLOR_TOL = 11.0  # Lab units: auto photo palette stops once the average color difference is below this
PDF_MARGIN, PDF_HEADER, PDF_LABEL = 10.0, 22.0, 7.0  # mm; header holds title + mini map
PDF_THIN, PDF_THICK = 0.15, 0.5  # line widths, mm


def plan_sheets(cols: int, rows: int, args) -> tuple[tuple[float, float], int, int]:
    """Pick orientation and cells per sheet so the chart needs the fewest sheets.

    When the chart does not fit on one sheet, sheets break on thick lines
    (multiples of --major) so neighbouring sheets line up by eye.
    """
    w0, h0 = PAGE_SIZES[args.page]
    best = None
    for w, h in ((w0, h0), (h0, w0)):
        fit = [int((w - 2 * PDF_MARGIN - PDF_LABEL) // args.cell_mm),
               int((h - 2 * PDF_MARGIN - PDF_HEADER - PDF_LABEL) // args.cell_mm)]
        for i, n in enumerate((cols, rows)):
            if n <= fit[i]:
                fit[i] = n
            elif args.major and fit[i] >= args.major:
                fit[i] -= fit[i] % args.major
        if min(fit) < 1:
            continue
        sheets = math.ceil(cols / fit[0]) * math.ceil(rows / fit[1])
        if best is None or sheets < best[0]:
            best = (sheets, (w, h), fit[0], fit[1])
    if best is None:
        raise ValueError("--cell-mm troppo grande per il foglio")
    return best[1], best[2], best[3]


def pdf_cells(page: Page, cells: np.ndarray, ox: float, oy: float, s: float) -> None:
    # Viewers anti-alias abutting rects into hairline seams. Each rect spills into
    # the next cell, which is painted later (row-major order) and covers the spill.
    eps = s * 0.25
    rows, cols = cells.shape[:2]
    for r, c, n, rgb in color_runs(cells):
        page.rect(ox + c * s, oy + r * s, n * s + (eps if c + n < cols else 0),
                  s + (eps if r + 1 < rows else 0), fill=rgb)


def pdf_grid(page: Page, cells: np.ndarray, ox: float, oy: float, s: float,
             x0: int, y0: int, args, label_size: float = 7.0) -> None:
    """Cells, grid and every-`major` numbers at `s` mm per cell, top-left at (ox, oy)."""
    rows, cols = cells.shape[:2]
    pdf_cells(page, cells, ox, oy, s)
    shrink = min(1.0, s / 3.0)  # thinner lines when cells are tiny (whole-chart page)
    segs = sorted(grid_segments(cells, args.major, x0, y0), key=lambda g: g[4])  # thick on top
    for vertical, k, a, b, thick, on_dark in segs:
        col, _ = segment_style(thick, on_dark, args)
        width = (PDF_THICK if thick else PDF_THIN) * shrink
        if vertical:
            page.line(ox + k * s, oy + a * s, ox + k * s, oy + b * s, col, width)
        else:
            page.line(ox + a * s, oy + k * s, ox + b * s, oy + k * s, col, width)

    if args.numbers and args.major:
        for k in range(cols + 1):
            g = x0 + k
            if g and g % args.major == 0:
                page.text(ox + k * s, oy - PDF_LABEL / 2, str(g), size=label_size, align="center")
        for k in range(rows + 1):
            g = y0 + k
            if g and g % args.major == 0:
                page.text(ox - 1, oy + k * s, str(g), size=label_size, align="right")


def pdf_sheet(page: Page, cells: np.ndarray, x0: int, y0: int, args) -> None:
    """One sheet of the chart; x0/y0 = global index of its first column/row."""
    pdf_grid(page, cells, PDF_MARGIN + PDF_LABEL, PDF_MARGIN + PDF_HEADER + PDF_LABEL,
             args.cell_mm, x0, y0, args)


def pdf_whole(doc: Document, cells: np.ndarray, name: str, args) -> None:
    """First page: the whole chart shrunk to one sheet, like the *_clean.png image."""
    rows, cols = cells.shape[:2]
    w, h = PAGE_SIZES[args.page]
    if cols > rows:
        w, h = h, w
    page = doc.add_page(w, h)
    page.text(PDF_MARGIN, PDF_MARGIN + 5, f"{name} – schema intero, {cols} x {rows} quadrati", size=12)
    aw = w - 2 * PDF_MARGIN - PDF_LABEL
    ah = h - 2 * PDF_MARGIN - PDF_HEADER - PDF_LABEL
    s = min(aw / cols, ah / rows)
    pdf_grid(page, cells, PDF_MARGIN + PDF_LABEL, PDF_MARGIN + PDF_HEADER + PDF_LABEL,
             s, 0, 0, args, label_size=min(7.0, max(4.0, s * 10 * 2.2)))


def pdf_minimap(page: Page, cells: np.ndarray, sheets, current: int) -> None:
    """Small copy of the whole chart, top right, with this sheet outlined in red."""
    rows, cols = cells.shape[:2]
    s = min(70.0 / cols, (PDF_HEADER - 4) / rows)
    ox, oy = page.w - PDF_MARGIN - cols * s, PDF_MARGIN
    pdf_cells(page, cells, ox, oy, s)
    for i, (x0, y0, cw, ch) in enumerate(sheets):
        if i != current:
            page.rect(ox + x0 * s, oy + y0 * s, cw * s, ch * s, stroke=(150, 150, 150), width=0.2)
    x0, y0, cw, ch = sheets[current]
    page.rect(ox + x0 * s, oy + y0 * s, cw * s, ch * s, stroke=(220, 0, 0), width=0.7)


def write_pdf(path: Path, cells: np.ndarray, name: str, args) -> int:
    rows, cols = cells.shape[:2]
    (w, h), fx, fy = plan_sheets(cols, rows, args)
    sheets = [(x0, y0, min(fx, cols - x0), min(fy, rows - y0))
              for y0 in range(0, rows, fy) for x0 in range(0, cols, fx)]
    doc = Document()
    if getattr(args, "overview", False) and len(sheets) > 1:
        pdf_whole(doc, cells, name, args)
    for i, (x0, y0, cw, ch) in enumerate(sheets):
        page = doc.add_page(w, h)
        if len(sheets) > 1:
            page.text(PDF_MARGIN, PDF_MARGIN + 5, f"{name} – foglio {i + 1} di {len(sheets)}", size=12)
            page.text(PDF_MARGIN, PDF_MARGIN + 11, f"colonne {x0 + 1}-{x0 + cw}, righe {y0 + 1}-{y0 + ch} "
                      f"(riquadro rosso a destra)", size=9)
            pdf_minimap(page, cells, sheets, i)
        else:
            page.text(PDF_MARGIN, PDF_MARGIN + 5, f"{name} – {cols} x {rows} quadrati", size=12)
        pdf_sheet(page, cells[y0:y0 + ch, x0:x0 + cw], x0, y0, args)
    doc.save(path)
    return len(sheets)


# ---------------------------------------------------------------------- CLI

def pair(kind):
    def parse(s: str):
        parts = s.lower().replace("x", ",").split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"formato atteso A,B oppure AxB: {s}")
        return kind(parts[0]), kind(parts[1])
    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path, help="immagini da elaborare")
    p.add_argument("--out", type=Path, help="cartella di uscita (default: accanto all'immagine)")
    p.add_argument("--mode", choices=["clean", "overlay"], default="clean",
                   help="clean = ridisegna pulita con griglia (default); "
                        "overlay = linee sopra l'immagine originale")
    p.add_argument("--info", action="store_true", help="stampa solo la griglia rilevata")

    g = p.add_argument_group("griglia (manuale, se il rilevamento sbaglia)")
    g.add_argument("--cell", type=float, help="lato del quadrato in pixel (es. 12.5)")
    g.add_argument("--offset", type=pair(float), help="posizione X,Y di un bordo di quadrato")
    g.add_argument("--grid", type=pair(int), metavar="COLONNExRIGHE",
                   help="numero di quadrati, se l'immagine e' ritagliata esattamente sul disegno")
    g.add_argument("--nonsquare", action="store_true", help="quadrati con lati x e y diversi")
    g.add_argument("--min-cell", type=float, default=3.0, help="lato minimo cercato (default 3)")
    g.add_argument("--max-cell", type=float, default=400.0, help="lato massimo cercato")
    g.add_argument("--harmonic-tol", type=float, default=0.8, help=argparse.SUPPRESS)

    ph = p.add_argument_group("foto normali (riconosciute da sole)")
    ph.add_argument("--photo", type=int, nargs="?", const=0, metavar="QUADRATI",
                    help=f"tratta l'immagine come foto; QUADRATI = larghezza dello schema "
                         f"(default automatica {PHOTO_WIDTHS[0]}-{PHOTO_WIDTHS[-1]}, "
                         f"piu' larga se la foto ha piu' dettagli)")
    bg = ph.add_mutually_exclusive_group()
    bg.add_argument("--remove-bg", action="store_true",
                    help="toglie sempre lo sfondo (default: solo se c'e' un soggetto chiaro)")
    bg.add_argument("--keep-bg", action="store_true", help="non toglie mai lo sfondo")
    ph.add_argument("--style", choices=["linee", "colori"], default="linee",
                    help="linee = disegno nero su bianco a china (default); "
                         "colori = pochi colori presi dalla foto")
    ph.add_argument("--detail", type=int, choices=[1, 2, 3], default=2,
                    help="stile linee: 1 = poco dettaglio, 2 = medio (default), 3 = molto")

    c = p.add_argument_group("colori (modalita' clean)")
    c.add_argument("--merge", type=float, default=25.0,
                   help="unisce colori simili entro questa distanza RGB (0 = no, default 25; "
                        "abbassa se sfumature vicine vengono unite)")
    c.add_argument("--colors", type=int, help="esattamente N colori")
    c.add_argument("--inner", type=float,
                   help="parte del quadrato usata per il colore, 0.1-1 (default: 0.5, "
                        "oppure 1 = tutto se l'immagine e' fatta di perline/diamanti)")
    c.add_argument("--trim", action="store_true", help="toglie righe/colonne di solo sfondo ai bordi")
    c.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                   help="se l'immagine ha gia' una griglia disegnata, tiene solo quella zona "
                        "(toglie bordi di screenshot)")

    r = p.add_argument_group("disegno")
    r.add_argument("--px", type=int, default=24, help="pixel per quadrato in uscita (clean, default 24)")
    r.add_argument("--major", type=int, default=10, help="linea spessa ogni N quadrati (0 = no)")
    r.add_argument("--numbers", action=argparse.BooleanOptionalAction, default=True,
                   help="numeri ogni N quadrati sul bordo")
    r.add_argument("--line-color", type=parse_color, default=parse_color("#505050"))
    r.add_argument("--major-color", type=parse_color, default=parse_color("#000000"))
    r.add_argument("--line-color-dark", type=parse_color, default=parse_color("#a0a0a0"),
                   help="linee sottili sopra quadrati scuri")
    r.add_argument("--major-color-dark", type=parse_color, default=parse_color("#ffffff"),
                   help="linee spesse sopra quadrati scuri")
    r.add_argument("--line-width", type=int, default=1)
    r.add_argument("--major-width", type=int, default=3)
    r.add_argument("--svg", action="store_true", help="salva anche SVG vettoriale (clean)")

    f = p.add_argument_group("PDF da stampare (modalita' clean)")
    f.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True,
                   help="salva il PDF diviso in fogli (default si')")
    f.add_argument("--page", choices=sorted(PAGE_SIZES), default="a4", help="formato foglio")
    f.add_argument("--overview", action="store_true",
                   help="aggiunge una prima pagina con lo schema intero (come l'immagine clean)")
    f.add_argument("--cell-mm", type=float, default=3.0,
                   help="lato del quadrato stampato in mm (default 3)")
    return p


BG_MODEL = "u2net"  # rembg keeps it in %USERPROFILE%\.u2net (downloaded on first use)


def _skip_pymatting() -> None:
    """Empty stand-ins for the pymatting modules rembg imports.

    rembg needs pymatting only for alpha matting, which we never use. pymatting is
    compiled by numba on import: minutes, and again on every run when its folder is
    read-only (installed under Program Files), so the real one is never loaded here.
    """
    import types

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("alpha matting non disponibile")

    for name, func in (("pymatting.alpha.estimate_alpha_cf", "estimate_alpha_cf"),
                       ("pymatting.foreground.estimate_foreground_ml", "estimate_foreground_ml"),
                       ("pymatting.util.util", "stack_images")):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        if not hasattr(sys.modules[name], func):
            setattr(sys.modules[name], func, unavailable)


def remove_background(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Photo subject on white, cropped to the subject, with its mask (0-255).

    Also returns the share of the photo the subject covers and how sure the cutout
    is: the solid part of the mask over all of it. A clear subject (a person, a dog
    in a meadow) gives 0.65-0.95; a building, a landscape or a small animal hidden
    in grass gives a faint, smeared one (0.2-0.6).
    """
    _skip_pymatting()
    try:
        from rembg import new_session, remove
    except ImportError:
        raise ValueError("rembg non installato (pip install \"rembg[cpu]\")") from None
    cut = remove(ImageOps.exif_transpose(Image.open(path)).convert("RGB"),
                 session=new_session(BG_MODEL))
    alpha = np.asarray(cut)[..., 3]
    solid = alpha > 200
    ys, xs = np.nonzero(solid)
    if not len(xs):
        raise ValueError("nessun soggetto trovato nella foto")
    sure = float(solid.mean() / max((alpha > 20).mean(), 1e-6))
    cut = cut.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(white, cut).convert("RGB"))
    return rgb, np.asarray(cut)[..., 3], float(solid.mean()), sure


def prepare_photo(path: Path, img: np.ndarray, args) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Cut the subject out when there is a clear one (a person, an object).

    Returns the image, the subject mask (None when the background is kept) and a note.
    With no clear subject (buildings, landscapes, an animal lost in grass) the cutout
    is unsure or covers almost nothing, and the whole photo is kept.
    """
    if args.keep_bg:
        return img, None, "sfondo tenuto"
    try:
        cut, mask, share, sure = remove_background(path)
    except ValueError as e:
        if args.remove_bg:
            raise
        return img, None, f"sfondo tenuto: {e}"
    if args.remove_bg or (sure >= 0.65 and 0.03 < share < 0.85):
        return cut, mask, f"sfondo tolto (il soggetto e' il {share:.0%} della foto)"
    return img, None, "sfondo tenuto: nessun soggetto chiaro (--remove-bg per toglierlo comunque)"


def pixel_art_scale(img: np.ndarray, max_cells: int = 300, max_colors: int = 128) -> int | None:
    """Pixels per cell (1 or 2) for exact pixel art stored at its own size.

    The grid detector only sees cells of 3 px or more. Lossless art with few colors
    whose 2x2 blocks (or single pixels) are the cells is taken as it is, as long as
    the chart stays under `max_cells` a side. JPEG photos have thousands of colors.
    """
    H, W = img.shape[:2]
    if len(np.unique(img.reshape(-1, 3), axis=0)) > max_colors:
        return None
    for f in (2, 1):
        if H % f or W % f or max(H, W) // f > max_cells:
            continue
        small = img[::f, ::f]
        if np.array_equal(np.repeat(np.repeat(small, f, axis=0), f, axis=1), img):
            return f
    return None


def is_chart(img: np.ndarray, gx: Axis, gy: Axis) -> bool:
    """True when the image really is made of cells on the detected grid.

    On a chart the cells line up with the colour changes, so moving the grid by half
    a cell makes every cell straddle two colours and its spread jumps (x3 to x20 on
    charts, beads and drills). A photo has no grid: the shift changes nothing (~x1).
    Charts rendered as little drawn crosses stay textured even when aligned (~x1.5),
    but their grid is found with high confidence (~0.6; photos stay below ~0.25).
    Flat artwork (logos, drawings) can line up with a few huge "cells", so a chart
    also needs at least 20 cells a side.
    """
    if min(gx.count, gy.count) < 20:
        return False
    a = img.astype(np.float32)
    H, W = a.shape[:2]

    def spread(ax: Axis, ay: Axis) -> float:
        xs, ys = cell_ranges(ax, W, 1.0)[1:-1:2], cell_ranges(ay, H, 1.0)[1:-1:2]
        return float(np.mean([a[y0:y1, x0:x1].reshape(-1, 3).var(0).sum()
                              for y0, y1 in ys for x0, x1 in xs] or [0.0]))

    shifted = spread(Axis(gx.start + gx.pitch / 2, gx.pitch, gx.count - 1),
                     Axis(gy.start + gy.pitch / 2, gy.pitch, gy.count - 1))
    ratio = shifted / max(spread(gx, gy), 1e-6)
    return ratio >= 1.8 or (ratio >= 1.3 and min(gx.score, gy.score) >= 0.4)


def photo_grid(W: int, H: int, cols: int) -> tuple[Axis, Axis]:
    """Square cells for an ordinary photo, `cols` cells wide."""
    if not 2 <= cols <= W:
        raise ValueError(f"--photo deve essere tra 2 e {W} quadrati")
    rows = max(1, round(H * cols / W))
    return Axis(0.0, W / cols, cols), Axis(0.0, H / rows, rows)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) to CIE Lab (D65): distances there match how different colors look."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193],
                        [0.3576, 0.7152, 0.1192],
                        [0.1805, 0.0722, 0.9505]]) / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def photo_palette(cells: np.ndarray, n_colors: int | None = None) -> np.ndarray:
    """Few clean colors for a photo.

    k-means in Lab, adding a color (seeded where the error is largest) until the
    average color difference drops below PHOTO_COLOR_TOL, at most PHOTO_MAX_COLORS.
    With n_colors the palette has exactly that many. Each color is the average of
    the cells it stands for.
    """
    flat = cells.reshape(-1, 3).astype(np.float64)
    u, inv, w = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    ul = rgb_to_lab(u)
    centers = ul[[np.argmax(w)]]
    while True:
        centers, lab, dist = _kmeans(ul, w, centers)
        k = len(centers)
        rms = np.sqrt((w * dist ** 2).sum() / w.sum())
        if k >= len(u) or (k >= n_colors if n_colors else rms <= PHOTO_COLOR_TOL or k >= PHOTO_MAX_COLORS):
            break
        centers = np.vstack([centers, ul[np.argmax(w * dist ** 2)]])
    pal = np.array([(u[lab == j] * w[lab == j, None]).sum(0) / w[lab == j].sum()
                    for j in range(k)])
    return np.round(pal).clip(0, 255).astype(np.uint8)[lab][inv].reshape(cells.shape)


def otsu(values: np.ndarray) -> float:
    """Gray level (0-1) that best splits `values` into a dark and a light group."""
    hist, _ = np.histogram(values, 256, (0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    om, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (mu[-1] * om - mu) ** 2 / (om * (1 - om) + 1e-12)
    return float(np.nanargmax(between)) / 255.0


def equalize_light(g: np.ndarray) -> np.ndarray:
    """Local contrast equalization (CLAHE) of a 0-1 gray image.

    With one threshold for the whole photo, the shadow side of a close-up face or
    animal turns into a solid black patch and its eyes and fur disappear. Evening
    out the light region by region keeps those details; only truly dark areas stay
    black. Left unchanged if scikit-image (installed with rembg) is missing.
    """
    try:
        from skimage.exposure import equalize_adapthist
    except ImportError:
        return g
    return equalize_adapthist(g, kernel_size=max(8, min(g.shape) // 6), clip_limit=0.01)


def line_art(img: np.ndarray, mask: np.ndarray | None, cols: int, rows: int,
             detail: int) -> np.ndarray:
    """Black-on-white ink drawing of a photo, like the stencil charts found online.

    Dark areas (hair, clothes, shadows) become solid black, darker than a share of
    the Otsu threshold of the subject; the dark side of sharp edges becomes lines
    (difference of Gaussians), so eyes, mouths and fingers show. Worked out at
    LINE_PX pixels per cell; a cell is black when enough of it is inked.
    """
    from scipy.ndimage import gaussian_filter

    sigma, dark_share, edge, cover = LINE_DETAIL[detail]
    size = (cols * LINE_PX, rows * LINE_PX)
    gray = ImageOps.autocontrast(Image.fromarray(img).convert("L").resize(size, Image.LANCZOS), cutoff=1)
    g = equalize_light(np.asarray(gray) / 255.0)
    subject = (np.ones(g.shape, bool) if mask is None else
               np.asarray(Image.fromarray(mask).resize(size, Image.BOX)) >= 128)
    dark = gaussian_filter(g, 1.0) < otsu(g[subject]) * dark_share
    lines = gaussian_filter(g, sigma) - gaussian_filter(g, sigma * 1.6) < -edge
    ink = (dark | lines) & subject
    black = ink.reshape(rows, LINE_PX, cols, LINE_PX).mean(axis=(1, 3)) >= cover
    cells = np.where(black[..., None], np.uint8(20), np.uint8(255)) * np.ones(3, np.uint8)
    return despeckle(cells)


def is_line_drawing(img: np.ndarray) -> bool:
    """True for black-on-white artwork such as a coloring page.

    Nearly every pixel is close to white or to black and hardly any has color;
    photos (even old black-and-white ones) are full of mid grays.
    """
    a = img.astype(np.int16)
    g = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    return bool(((g > 200) | (g < 70)).mean() > 0.85 and (sat < 40).mean() > 0.9
                and (g > 200).mean() > 0.5)


def drawing_cells(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Copy the lines of a drawing onto the grid: a cell is black when a line crosses enough of it.

    A frame around the picture is dropped: thick black areas joined to the image
    border. Lines of the drawing are thin, so they stay even where they touch it.
    """
    from scipy import ndimage

    size = (cols * LINE_PX, rows * LINE_PX)
    g = np.asarray(Image.fromarray(img).convert("L").resize(size, Image.LANCZOS))
    black = g < 128
    # Opening wipes out anything thinner than k px (the drawn lines) and keeps solid areas.
    k = max(5, round(0.015 * min(size)))
    solid = ndimage.binary_opening(black, structure=np.ones((k, k)))
    lab, n = ndimage.label(solid)
    edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    frame = np.isin(lab, edge_ids[edge_ids > 0])
    frame = ndimage.binary_dilation(frame, iterations=2)  # include its anti-aliased rim
    ink = (black & ~frame).reshape(rows, LINE_PX, cols, LINE_PX).mean(axis=(1, 3)) >= 0.2
    if frame.any():
        # Where the frame is thin its inner rim survives as a line hugging the edge:
        # with a frame there, the outer 3 cells are frame, not drawing.
        ink[:3] = ink[-3:] = False
        ink[:, :3] = ink[:, -3:] = False
    return np.where(ink[..., None], np.uint8(20), np.uint8(255)) * np.ones(3, np.uint8)


def text_height(img: np.ndarray) -> float | None:
    """Letter height in pixels of the smallest line of text in a drawing, or None.

    Text = at least 4 separate small black shapes of about the same height, side by
    side on one baseline with gaps narrower than a letter, drawn with thin strokes
    (letters fill well under 60% of their box). Letters also differ from each other
    in width or fill; a row of identical windows, fence posts or beads does not.
    """
    from scipy import ndimage

    ink = np.asarray(Image.fromarray(img).convert("L")) < 128
    lab, n = ndimage.label(ink)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        fill = (lab[sl] == i).mean()
        if 6 <= h <= 60 and 0.15 <= w / h <= 1.4 and fill < 0.6:
            boxes.append((sl[1].start, sl[1].stop, (sl[0].start + sl[0].stop) / 2, h, fill))
    boxes.sort()
    heights = []
    for k, (x0, x1, cy, h, fill) in enumerate(boxes):
        line, right = [(h, x1 - x0, fill)], x1
        for bx0, bx1, bcy, bh, bfill in boxes[k + 1:]:
            if bx0 - right > 1.2 * h:
                break
            if abs(bcy - cy) < 0.4 * h and 0.67 < bh / h < 1.5:
                line.append((bh, bx1 - bx0, bfill))
                right = max(right, bx1)
        if len(line) < 4:
            continue
        hs, ws, fs = (np.array(v, float) for v in zip(*line))
        if ws.std() / ws.mean() > 0.1 or fs.std() / fs.mean() > 0.1:
            heights.append(float(np.median(hs)))
    return min(heights) if heights else None


def smooth_texture(img: np.ndarray) -> np.ndarray:
    """Flatten grass, leaves and gravel while keeping strong outlines (bilateral filter).

    In a scene with no single subject every blade of grass is an edge and would
    become a black dash. Left unchanged if OpenCV (installed with rembg) is missing.
    """
    try:
        import cv2
    except ImportError:
        return img
    H, W = img.shape[:2]
    if max(H, W) > 1000:  # the filter is slow on big photos; the chart needs far less
        f = 1000 / max(H, W)
        img = np.asarray(Image.fromarray(img).resize((round(W * f), round(H * f)), Image.LANCZOS))
    out = np.ascontiguousarray(img)
    for _ in range(3):
        out = cv2.bilateralFilter(out, 9, 60, 9)
    return out


def drop_small_blobs(cells: np.ndarray, min_cells: int = 6) -> np.ndarray:
    """Remove black specks and fill white holes smaller than `min_cells` cells.

    Only for scenes without a subject: in a portrait an eye can be 2-3 cells.
    """
    from scipy import ndimage

    ink = cells[..., 0] < 128
    for value in (True, False):
        region = ink == value
        lab, n = ndimage.label(region)
        sizes = ndimage.sum(region, lab, range(1, n + 1))
        ink[np.isin(lab, np.nonzero(sizes < min_cells)[0] + 1)] = not value
    return np.where(ink[..., None], np.uint8(20), np.uint8(255)) * np.ones(3, np.uint8)


def auto_photo_width(img: np.ndarray, mask: np.ndarray | None, drawing: bool = False) -> int:
    """Smallest chart width that keeps the drawing's detail.

    The line-art drawing at PHOTO_WIDTHS is compared with a 200-cell reference; the
    first width whose ink overlaps the reference by 90% wins. A simple portrait gets
    ~100 cells, a busy or large subject goes up to the last width.
    """
    H, W = img.shape[:2]

    def ink(cols: int) -> np.ndarray:
        rows = max(1, round(H * cols / W))
        cells = drawing_cells(img, cols, rows) if drawing else line_art(img, mask, cols, rows, 2)
        return cells[..., 0] < 128

    ref = ink(200)
    for cols in PHOTO_WIDTHS:
        up = np.asarray(Image.fromarray(ink(cols).astype(np.uint8) * 255)
                        .resize(ref.shape[::-1], Image.NEAREST)) > 0
        if (up & ref).sum() >= 0.9 * max((up | ref).sum(), 1):
            return cols
    return PHOTO_WIDTHS[-1]


def despeckle(cells: np.ndarray, passes: int = 2) -> np.ndarray:
    """Give lone cells (no side neighbour of the same color) their neighbours' color.

    A single stitch of a color is tedious to sew and usually photo noise.
    """
    c = cells.copy()
    H, W = c.shape[:2]
    for _ in range(passes):
        pad = np.pad(c, ((1, 1), (1, 1), (0, 0)), mode="edge")
        sides = [pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        lonely = ~np.any([np.all(s == c, axis=2) for s in sides], axis=0)
        if not lonely.any():
            break
        for y, x in zip(*np.nonzero(lonely)):
            opts = [tuple(s[y, x]) for s in sides]
            c[y, x] = max(set(opts), key=opts.count)
    return c


@dataclass
class Layout:
    """What an image is and where its cells are, before colors are worked out."""
    img: np.ndarray
    gx: Axis
    gy: Axis
    kind: str                       # "chart", "native", "drawing" or "photo"
    mask: np.ndarray | None = None  # photo subject mask when the background was removed
    native: int | None = None       # pixels per cell for native pixel art


def analyze(path: Path, args, say=print) -> Layout:
    """Decide what the image is (chart, pixel art, drawing, photo) and lay out its grid."""
    img = load_rgb(path)
    H, W = img.shape[:2]
    photo = args.photo is not None
    native = None
    if not photo:
        warnings = io.StringIO()
        try:
            with contextlib.redirect_stderr(warnings):
                gx, gy = detect_grid(img, args)
            photo = not (args.grid or args.cell) and not is_chart(img, gx, gy)
        except ValueError:
            if args.grid or args.cell:
                raise
            photo = True
        # No grid of 3+ px cells: it may still be pixel art stored at 1-2 px per cell.
        native = pixel_art_scale(img) if photo else None
        if native:
            photo = False
            gx = Axis(0.0, float(native), W // native)
            gy = Axis(0.0, float(native), H // native)
        elif not photo and warnings.getvalue():
            say(warnings.getvalue().rstrip())
    mask = None
    drawing = photo and args.style == "linee" and is_line_drawing(img)
    if drawing:
        # Already black lines on white: no background removal, no filters.
        how = ""
        width = args.photo
        if not width:
            width = auto_photo_width(img, None, drawing=True)
            how = " (larghezza scelta dai dettagli)"
            letters = text_height(img)
            if letters:
                # Wide enough for letters TEXT_CELLS tall, rounded up to a thick line.
                need = min(TEXT_MAX_WIDTH, math.ceil(TEXT_CELLS * W / letters / 10) * 10)
                if need > width:
                    width = need
                    how = f" (larghezza scelta per rendere leggibile la scritta, lettere di {letters:.0f}px)"
        gx, gy = photo_grid(W, H, width)
        say(f"{path.name}: disegno a linee -> {gx.count} x {gy.count} quadrati{how}")
        return Layout(img, gx, gy, "drawing")
    if photo:
        img, mask, note = prepare_photo(path, img, args)
        if mask is None and args.style == "linee":
            img = smooth_texture(img)
        H, W = img.shape[:2]
        width = args.photo or auto_photo_width(img, mask)
        gx, gy = photo_grid(W, H, width)
        how = "" if args.photo else " (larghezza scelta dai dettagli)"
        say(f"{path.name}: foto, {note} -> {gx.count} x {gy.count} quadrati{how}")
        return Layout(img, gx, gy, "photo", mask=mask)
    if native:
        say(f"{path.name}: pixel art {W}x{H}px, {native}px per quadrato -> "
            f"{gx.count} x {gy.count} quadrati")
        return Layout(img, gx, gy, "native", native=native)
    say(f"{path.name}: {W}x{H}px -> {gx.count} x {gy.count} quadrati, "
        f"lato {gx.pitch:.3f} x {gy.pitch:.3f}px, primo bordo a "
        f"({gx.start % gx.pitch:.2f}, {gy.start % gy.pitch:.2f}), "
        f"affidabilita' {min(gx.score, gy.score):.2f}")
    if min(gx.score, gy.score) < 0.35:
        say("  attenzione: griglia poco sicura, controlla il risultato "
            "(eventualmente usa --cell o --grid)")
    return Layout(img, gx, gy, "chart")


def chart_cells(lay: Layout, args, say=print) -> np.ndarray:
    """One color per cell for the laid-out image, cleaned up for stitching."""
    img, gx, gy, mask, native = lay.img, lay.gx, lay.gy, lay.mask, lay.native
    drawing, photo = lay.kind == "drawing", lay.kind == "photo"
    if drawing:
        cells = drawing_cells(img, gx.count, gy.count)
    elif photo and args.style == "linee":
        cells = line_art(img, mask, gx.count, gy.count, args.detail)
        if mask is None:
            cells = drop_small_blobs(cells)
    elif photo:
        # Area average of each cell; a photo has no grid to keep or crop.
        cells = np.asarray(Image.fromarray(img).resize((gx.count, gy.count), Image.BOX))
        if mask is None:
            cells = despeckle(photo_palette(cells, args.colors))
        else:
            # Colors are chosen on the subject only; the background stays plain white.
            bg = np.asarray(Image.fromarray(mask).resize((gx.count, gy.count), Image.BOX)) < 128
            subject = photo_palette(cells[~bg][None], args.colors)[0]
            cells = np.full_like(cells, 255)
            cells[~bg] = subject
            cells = despeckle(cells)
    elif native:
        # Exact colors, one per cell: nothing to clean up unless asked for fewer colors.
        cells = img[::native, ::native].copy()
        if args.colors:
            cells = build_palette(cells, args.merge, args.colors)
    else:
        inner = args.inner
        if inner is None:
            inner = 0.5
            if cell_texture(img, gx, gy) > 40:
                inner = 1.0
                say("  quadrati con texture (perline/diamanti): colore preso da tutto il quadrato")
        cells = sample_cells(img, gx, gy, inner)
        extent = drawn_grid_extent(img, gx, gy) if args.crop else None
        if extent and (len(extent[0]), len(extent[1])) != cells.shape[:2]:
            rr, cc = extent
            cells = cells[rr.start:rr.stop, cc.start:cc.stop]
            say(f"  griglia gia' disegnata: tengo righe {rr.start + 1}-{rr.stop}, "
                f"colonne {cc.start + 1}-{cc.stop} (--no-crop per tenere tutto)")
        if args.colors or args.merge > 0:
            cells = build_palette(cells, args.merge, args.colors)
        if not args.colors:
            cells = fold_blends(cells)
            cells, dropped = drop_edge_text(cells)
            if dropped:
                say("  tolta una scritta sul bordo (es. nome del sito)")
    if args.trim:
        cells, _, _ = trim_background(cells)
    n_colors = len(np.unique(cells.reshape(-1, 3), axis=0))
    say(f"  schema {cells.shape[1]} x {cells.shape[0]} quadrati, {n_colors} colori")
    if n_colors > 30 and not args.colors:
        say("  molti colori: se sono sfumature di rumore prova --colors N o --merge 35")
    return cells


def process(path: Path, args) -> None:
    lay = analyze(path, args)
    if args.info:
        return
    out_dir = args.out or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{path.stem}_{args.mode}"
    if args.mode == "overlay":
        render_overlay(lay.img, lay.gx, lay.gy, args).save(stem.with_suffix(".png"))
        print(f"  -> {stem.with_suffix('.png')}")
        return
    cells = chart_cells(lay, args)
    render_chart(cells, args).save(stem.with_suffix(".png"))
    print(f"  -> {stem.with_suffix('.png')}")
    if args.pdf:
        n = write_pdf(stem.with_suffix(".pdf"), cells, path.stem, args)
        print(f"  -> {stem.with_suffix('.pdf')} ({n} {'foglio' if n == 1 else 'fogli'})")
    if args.svg:
        write_svg(stem.with_suffix(".svg"), cells, args)
        print(f"  -> {stem.with_suffix('.svg')}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    failed = 0
    for path in args.images:
        try:
            process(path, args)
        except Exception as e:  # keep going on batch runs
            failed += 1
            print(f"{path}: errore: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
