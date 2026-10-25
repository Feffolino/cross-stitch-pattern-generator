"""Minimal PDF writer: filled rectangles, lines, Helvetica text and RGB images.

No external dependencies. Coordinates are in millimetres from the top-left corner.
"""
from __future__ import annotations

import zlib
from pathlib import Path

MM = 72 / 25.4  # points per millimetre

# Helvetica advance widths (1/1000 em) for the characters we print most; others ~0.55 em.
_WIDTHS = {**{d: 556 for d in "0123456789"}, " ": 278, "-": 333, ".": 278, ",": 278,
           "/": 278, "#": 556, "(": 333, ")": 333, ":": 278, "–": 556, "—": 1000}


def text_width_mm(s: str, size: float) -> float:
    return sum(_WIDTHS.get(ch, 550) for ch in s) / 1000 * size / MM


def _escape(s: str) -> bytes:
    b = s.encode("cp1252", "replace")
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _rgb(c) -> str:
    return " ".join(f"{v / 255:.3f}" for v in c)


class Page:
    def __init__(self, w_mm: float, h_mm: float):
        self.w, self.h = w_mm, h_mm
        self.ops: list[bytes] = []
        self.images: list = []  # PIL images drawn on this page, as /Im1, /Im2...
        self._fill = self._stroke = self._width = None

    def _y(self, y: float) -> float:
        return (self.h - y) * MM

    def _set_fill(self, c):
        if c != self._fill:
            self.ops.append(f"{_rgb(c)} rg".encode())
            self._fill = c

    def _set_stroke(self, c, width):
        if c != self._stroke:
            self.ops.append(f"{_rgb(c)} RG".encode())
            self._stroke = c
        if width != self._width:
            self.ops.append(f"{width * MM:.3f} w".encode())
            self._width = width

    def rect(self, x, y, w, h, fill=None, stroke=None, width=0.2):
        if fill is not None:
            self._set_fill(tuple(fill))
        if stroke is not None:
            self._set_stroke(tuple(stroke), width)
        op = "B" if fill is not None and stroke is not None else ("f" if fill is not None else "S")
        self.ops.append(f"{x * MM:.3f} {self._y(y + h):.3f} {w * MM:.3f} {h * MM:.3f} re {op}".encode())

    def line(self, x1, y1, x2, y2, color, width):
        self._set_stroke(tuple(color), width)
        self.ops.append(f"{x1 * MM:.3f} {self._y(y1):.3f} m {x2 * MM:.3f} {self._y(y2):.3f} l S".encode())

    def text(self, x, y, s: str, size=8.0, align="left", color=(0, 0, 0)):
        """Draw text with its vertical middle at y; align is left, center or right."""
        if align == "center":
            x -= text_width_mm(s, size) / 2
        elif align == "right":
            x -= text_width_mm(s, size)
        self._set_fill(tuple(color))
        base = y + 0.35 * size / MM
        self.ops.append(b"BT /F1 %.2f Tf %.3f %.3f Td (" % (size, x * MM, self._y(base))
                        + _escape(s) + b") Tj ET")

    def image(self, x, y, w, h, im):
        """Draw a PIL image stretched to the w x h mm box whose top-left is (x, y)."""
        self.images.append(im.convert("RGB"))
        self.ops.append(f"q {w * MM:.3f} 0 0 {h * MM:.3f} {x * MM:.3f} {self._y(y + h):.3f} cm "
                        f"/Im{len(self.images)} Do Q".encode())


class Document:
    def __init__(self):
        self.pages: list[Page] = []

    def add_page(self, w_mm: float, h_mm: float) -> Page:
        p = Page(w_mm, h_mm)
        self.pages.append(p)
        return p

    def save(self, path: Path) -> None:
        # Object numbers: 1 catalog, 2 page tree, 3 font, then (page, content) pairs,
        # then one image XObject per drawn image.
        n = len(self.pages)
        kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
        images, next_obj = [], 4 + 2 * n
        for i, p in enumerate(self.pages):
            xobj = " ".join(f"/Im{k + 1} {next_obj + len(images) + k} 0 R" for k in range(len(p.images)))
            images += p.images
            res = "/Font << /F1 3 0 R >>" + (f" /XObject << {xobj} >>" if xobj else "")
            objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {p.w * MM:.2f} {p.h * MM:.2f}] "
                        f"/Resources << {res} >> /Contents {5 + 2 * i} 0 R >>".encode())
            data = zlib.compress(b"\n".join(p.ops))
            objs.append(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data)
                        + data + b"\nendstream")
        for im in images:
            data = zlib.compress(im.tobytes(), 6)
            objs.append(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                        b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n"
                        % (im.width, im.height, len(data)) + data + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, 1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
        out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
        Path(path).write_bytes(bytes(out))
