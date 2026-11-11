"""Synthetic checks for grid detection and chart output.

Run:  python tests/test_gridify.py          (or with pytest)
Each case builds a pixel-art image with a known grid, degrades it like a picture
found online (non-integer upscale, blur, JPEG, margins, drawn grid lines), then
checks that the detected cell size and the reconstructed cell colors match.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gridify  # noqa: E402

PALETTE = np.array([[255, 255, 255], [20, 20, 20], [200, 40, 40], [40, 150, 60],
                    [50, 80, 200], [240, 200, 40], [150, 90, 40], [230, 140, 180]], np.uint8)


def pixel_art(cols, rows, seed, n_colors=8):
    """Blobby random art: smoothed noise snapped to a palette."""
    rng = np.random.default_rng(seed)
    noise = rng.random((rows // 3 + 2, cols // 3 + 2, 3))
    small = Image.fromarray((noise * 255).astype(np.uint8)).resize((cols, rows), Image.BICUBIC)
    a = np.asarray(small).astype(int)
    pal = PALETTE[:n_colors].astype(int)
    idx = np.argmin(((a[:, :, None, :] - pal[None, None]) ** 2).sum(-1), axis=2)
    # sprinkle single cells so edges are not only on blob outlines
    mask = rng.random((rows, cols)) < 0.05
    idx[mask] = rng.integers(0, n_colors, mask.sum())
    return PALETTE[idx]


def degrade(art, pitch, margin=(0, 0), blur=0.0, jpeg=None, gridlines=False, bg=(255, 255, 255)):
    rows, cols = art.shape[:2]
    big = Image.fromarray(art).resize((round(cols * pitch), round(rows * pitch)), Image.NEAREST)
    if gridlines:
        d = ImageDraw.Draw(big)
        for k in range(cols + 1):
            x = round(k * big.width / cols)
            d.line([(x, 0), (x, big.height)], fill=(160, 160, 160))
        for k in range(rows + 1):
            y = round(k * big.height / rows)
            d.line([(0, y), (big.width, y)], fill=(160, 160, 160))
    mx, my = margin
    canvas = Image.new("RGB", (big.width + 2 * mx, big.height + 2 * my), bg)
    canvas.paste(big, (mx, my))
    if blur:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    if jpeg:
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=jpeg)
        canvas = Image.open(buf).convert("RGB")
    true_pitch = (big.width / cols, big.height / rows)
    return np.asarray(canvas), true_pitch


def run_case(name, art, pitch, colors=None, **kw):
    img, (tpx, tpy) = degrade(art, pitch, **kw)
    args = gridify.build_parser().parse_args(["x.png"])
    gx, gy = gridify.detect_grid(img, args)
    assert gridify.cell_texture(img, gx, gy) < 40  # pixel art: centre sampling
    cells = gridify.build_palette(gridify.sample_cells(img, gx, gy, 0.5), args.merge, colors)
    n_found = len(np.unique(cells.reshape(-1, 3), axis=0))
    n_true = len(np.unique(art.reshape(-1, 3), axis=0))

    mx, my = kw.get("margin", (0, 0))
    sx, sy = round((mx - gx.start) / gx.pitch), round((my - gy.start) / gy.pitch)
    rows, cols = art.shape[:2]
    got = cells[sy:sy + rows, sx:sx + cols].astype(int)
    ok_shape = got.shape[:2] == (rows, cols)
    acc = (np.linalg.norm(got - art.astype(int), axis=2) < 40).mean() if ok_shape else 0.0
    err = max(abs(gx.pitch - tpx) / tpx, abs(gy.pitch - tpy) / tpy)
    # Margins add their own background color, hence the +1.
    passed = err < 0.005 and acc > 0.98 and n_found <= n_true + 1
    print(f"{'OK ' if passed else 'FAIL'} {name:28s} pitch {gx.pitch:7.3f}/{tpx:7.3f} "
          f"err {err * 100:5.2f}%  cells {gx.count}x{gy.count}  match {acc * 100:6.2f}%  "
          f"colors {n_found}/{n_true}  score {min(gx.score, gy.score):.2f}")
    return passed


CASES = [
    ("crisp x10", dict(art=(40, 30, 1), pitch=10)),
    ("x7.3 blur jpeg60 margin", dict(art=(32, 24, 2), pitch=7.3, margin=(13, 5), blur=1.0, jpeg=60)),
    # Tiny cells + JPEG chroma subsampling smear colors: needs an explicit --colors.
    ("x4.6 blur jpeg70 --colors 8", dict(art=(60, 45, 3), pitch=4.6, blur=0.8, jpeg=70, colors=8)),
    ("x12 drawn gridlines jpeg80", dict(art=(30, 30, 4), pitch=12, gridlines=True, jpeg=80)),
    ("x9 sprite big margin", dict(art=(20, 20, 5), pitch=9, margin=(180, 140))),
    ("x6 large jpeg75", dict(art=(150, 110, 6), pitch=6, jpeg=75)),
    ("x8 aligned to jpeg blocks", dict(art=(40, 40, 7), pitch=8, jpeg=50)),
    ("x15.5 heavy blur", dict(art=(25, 20, 8), pitch=15.5, blur=2.0, jpeg=85)),
    ("x5 three colors", dict(art=(50, 40, 9, 3), pitch=5, jpeg=80)),
]


def test_detection():
    failed = []
    for name, c in CASES:
        cols, rows, seed, *nc = c["art"]
        art = pixel_art(cols, rows, seed, *nc)
        kw = {k: v for k, v in c.items() if k != "art"}
        if not run_case(name, art, **kw):
            failed.append(name)
    assert not failed, failed


def test_outputs():
    """End-to-end CLI run: PNG + multi-sheet PDF + SVG are written and well formed."""
    art = pixel_art(150, 110, 6)
    img, _ = degrade(art, 6, jpeg=80)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "big.jpg"
        Image.fromarray(img).save(src, quality=90)
        assert gridify.main([str(src), "--svg"]) == 0
        pdf = (Path(d) / "big_clean.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
        n_pages = int(pdf.split(b"/Count ")[1].split(b" ")[0])
        # 150x110 cells at 3 mm on A4 -> 6 sheets, nothing else
        assert n_pages == 6, n_pages
        assert Image.open(Path(d) / "big_clean.png").size[0] > 150 * 24
        assert (Path(d) / "big_clean.svg").read_text().startswith("<svg")
        # --overview adds one whole-chart page in front of the sheets
        assert gridify.main([str(src), "--overview"]) == 0
        pdf = (Path(d) / "big_clean.pdf").read_bytes()
        assert int(pdf.split(b"/Count ")[1].split(b" ")[0]) == n_pages + 1
        assert gridify.main([str(src), "--mode", "overlay"]) == 0
        assert (Path(d) / "big_overlay.png").exists()


def fake_photo(w=480, h=640, seed=0):
    """Smooth gradients, a dark figure with a light face, noise: no grid anywhere."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([120 + 80 * xx / w, 140 + 60 * yy / h, 160 - 40 * xx / w], -1)
    body = ((xx - w / 2) / (w * 0.3)) ** 2 + ((yy - h * 0.75) / (h * 0.3)) ** 2 < 1
    face = ((xx - w / 2) / (w * 0.15)) ** 2 + ((yy - h * 0.35) / (h * 0.13)) ** 2 < 1
    eyes = ((((xx - w * 0.45) ** 2 + (yy - h * 0.33) ** 2) < (w * 0.02) ** 2) |
            (((xx - w * 0.55) ** 2 + (yy - h * 0.33) ** 2) < (w * 0.02) ** 2))
    img[body] = (25, 25, 30)
    img[face] = (235, 200, 180)
    img[eyes] = (20, 15, 15)
    img = img + rng.normal(0, 6, img.shape)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))


def test_photo_mode():
    """A photo is recognised as such and becomes a 2-color line drawing of auto width."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "ritratto.jpg"
        fake_photo().save(src, quality=90)
        img = gridify.load_rgb(src)
        args = gridify.build_parser().parse_args([str(src)])
        gx, gy = gridify.detect_grid(img, args)
        assert not gridify.is_chart(img, gx, gy)
        assert gridify.auto_photo_width(img, None) in gridify.PHOTO_WIDTHS

        assert gridify.main([str(src), "--keep-bg", "--no-pdf"]) == 0
        out = np.asarray(Image.open(Path(d) / "ritratto_clean.png").convert("RGB"))
        assert out.shape[1] > 80 * 24  # at least 80 cells of 24 px
        cells = gridify.line_art(img, None, 80, 107, 2)
        assert len(np.unique(cells.reshape(-1, 3), axis=0)) == 2
        assert (cells[..., 0] < 128).mean() > 0.2  # the dark figure is inked

        assert gridify.main([str(src), "--keep-bg", "--no-pdf", "--style", "colori",
                             "--colors", "3"]) == 0


def test_chart_cleanup():
    """Blend colors fold into their neighbours; a watermark on the edge goes away."""
    black, white, grey, red = (0, 0, 0), (255, 255, 255), (128, 128, 128), (220, 30, 30)
    cells = np.full((40, 40, 3), white, np.uint8)
    cells[5:30, 5:30] = black
    cells[10, 30] = grey                      # half-stitched edge cell
    folded = gridify.fold_blends(cells)
    assert len(np.unique(folded.reshape(-1, 3), axis=0)) == 2

    marked = folded.copy()
    marked[37:39, 20:35] = red                # site name along the bottom edge
    clean, dropped = gridify.drop_edge_text(marked)
    assert dropped and not np.all(clean == red, axis=2).any()

    accent = folded.copy()
    accent[15, 15] = red                      # red mouth inside the drawing stays
    kept, dropped = gridify.drop_edge_text(accent)
    assert not dropped and np.all(kept[15, 15] == red)


def test_native_pixel_art():
    """Pixel art saved at 1 or 2 px per cell (too small for the grid detector) is kept cell for cell."""
    art = pixel_art(60, 40, 12)
    for scale in (1, 2):
        img = np.repeat(np.repeat(art, scale, axis=0), scale, axis=1)
        assert gridify.pixel_art_scale(img) == scale
    # a photo has far too many colors to be taken as pixel art
    assert gridify.pixel_art_scale(np.asarray(fake_photo(120, 160).convert("RGB"))) is None
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "sprite.png"
        Image.fromarray(art).save(src)
        assert gridify.main([str(src), "--no-pdf"]) == 0
        with Image.open(Path(d) / "sprite_clean.png") as out:  # closed before cleanup (Windows)
            assert out.size[0] >= 60 * 24


def test_line_drawing():
    """A coloring page keeps its lines, loses its frame, and is not taken for a photo."""
    im = Image.new("RGB", (600, 400), "black")
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([12, 12, 588, 388], radius=30, fill="white")   # black frame outside
    d.rectangle([150, 120, 450, 330], outline="black", width=3)        # house
    d.line([(150, 120), (300, 50), (450, 120)], fill="black", width=3)  # roof
    d.line([(20, 300), (150, 330)], fill="black", width=3)              # path touching nothing
    img = np.asarray(im)
    assert gridify.is_line_drawing(img)
    assert not gridify.is_line_drawing(np.asarray(fake_photo().convert("RGB")))
    cells = gridify.drawing_cells(img, 120, 80)
    ink = cells[..., 0] < 128
    assert not ink[0].any() and not ink[:, 0].any()   # frame gone
    assert ink[24:66, 30].any() and ink[24, 30:90].any()  # house walls kept

    # identical windows are not text; a real line of letters is, with its height
    for x in range(170, 440, 40):
        d.rectangle([x, 150, x + 25, 190], outline="black", width=2)
    assert gridify.text_height(np.asarray(im)) is None
    d.text((180, 250), "CASA FAMIGLIA", fill="black", font_size=24)
    assert 10 <= gridify.text_height(np.asarray(im)) <= 24


def test_fine_drawn_grid():
    """Chart screenshot with 2-3 px cells, 1 px grid lines (darker every 10 cells,
    as chart programs draw them at that zoom) and a ruler on top: found as cells, not as a photo or as 10x10 blocks."""
    cols, rows, p, ruler = 120, 90, 2.86, 18
    blob = np.all(pixel_art(cols, rows, 21, 2) == PALETTE[1], axis=2)
    inside = blob.copy()                                # line art: blob outlines only
    inside[1:-1, 1:-1] &= blob[:-2, 1:-1] & blob[2:, 1:-1] & blob[1:-1, :-2] & blob[1:-1, 2:]
    stitch = blob & ~inside
    W, H = round(cols * p), round(rows * p)
    im = Image.new("RGB", (W, H + ruler), "white")
    d = ImageDraw.Draw(im)
    for r, c in zip(*np.nonzero(~stitch | stitch)):
        x0, y0 = round(c * p), ruler + round(r * p)
        d.rectangle([x0, y0, round((c + 1) * p), ruler + round((r + 1) * p)],
                    fill=(110,) * 3 if stitch[r, c] else (215,) * 3)
    for k in range(cols + 1):
        d.line([(round(k * p), ruler), (round(k * p), H + ruler)],
               fill=(70,) * 3 if k % 10 == 0 else (175,) * 3)
    for k in range(rows + 1):
        y = ruler + round(k * p)
        d.line([(0, y), (W, y)], fill=(70,) * 3 if k % 10 == 0 else (175,) * 3)
    for k in range(10, cols, 10):                       # ruler: ticks and numbers
        d.line([(round(k * p), 2), (round(k * p), ruler - 4)], fill="black")
        d.text((round(k * p) + 2, 2), str(k), fill="black")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "chart.jpg"
        src.write_bytes(buf.getvalue())
        args = gridify.build_parser().parse_args([str(src)])
        lay = gridify.analyze(src, args, say=lambda *a: None)
        cells = gridify.chart_cells(lay, args, say=lambda *a: None)
    assert lay.kind == "chart" and lay.gx.drawn
    assert abs(lay.gx.pitch - p) / p < 0.01 and abs(lay.gy.pitch - p) / p < 0.01
    assert len(np.unique(cells.reshape(-1, 3), axis=0)) == 2
    got = cells.astype(int).sum(axis=2) < 3 * 160
    assert abs(got.shape[0] - rows) <= 2 and abs(got.shape[1] - cols) <= 2  # ruler cropped
    def match(dy, dx):  # got[r + dy, c + dx] is art cell (r, c)
        a, b = got[max(0, dy):, max(0, dx):], stitch[max(0, -dy):, max(0, -dx):]
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        return (a[:h, :w] == b[:h, :w]).mean()
    best = max(match(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3))
    assert best > 0.97, best


def test_print_options():
    """--one-page picks the largest cell that fits one sheet; --contrast sets the grid
    unless a line color is given explicitly."""
    cells = np.full((167, 95, 3), 220, np.uint8)
    args = gridify.build_parser().parse_args(["x", "--cell-mm", "3", "--one-page"])
    mm = gridify.fit_one_page(95, 167, args)
    probe = gridify.build_parser().parse_args(["x", "--cell-mm", str(mm)])
    assert gridify.plan_sheets(95, 167, probe)[1:] == (95, 167)          # fits at mm
    probe.cell_mm = round(mm + 0.1, 1)
    _, fx, fy = gridify.plan_sheets(95, 167, probe)
    assert fx < 95 or fy < 167                                            # not at mm + 0.1
    with tempfile.TemporaryDirectory() as d:
        assert gridify.write_pdf(Path(d) / "one.pdf", cells, "one", args) == 1
    assert args.cell_mm == mm
    big = gridify.build_parser().parse_args(["x", "--cell-mm", "3", "--one-page"])
    assert gridify.fit_one_page(10, 10, big) == 3.0                       # never enlarges
    thin = gridify.build_parser().parse_args(["x", "--cell-mm", "3", "--one-page", "--margin", "4"])
    assert gridify.fit_one_page(95, 167, thin) > mm                       # less margin, bigger cells

    strong = gridify.build_parser().parse_args(["x", "--contrast", "3"])
    assert gridify.thin_line(strong, False) == ((0, 0, 0), 2)
    assert gridify.thin_line(strong, True)[0] == (255, 255, 255)
    mine = gridify.build_parser().parse_args(["x", "--contrast", "3", "--line-color", "#123456"])
    assert gridify.thin_line(mine, False)[0] == (0x12, 0x34, 0x56)


if __name__ == "__main__":
    test_detection()
    test_outputs()
    test_photo_mode()
    test_chart_cleanup()
    test_native_pixel_art()
    test_line_drawing()
    test_fine_drawn_grid()
    test_print_options()
    print("tutti i test ok")
