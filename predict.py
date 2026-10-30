"""Find the regions of a drawing sheet: the drawing itself, legends, tables, notes, abbreviation lists
and the title block.

    python predict.py sheet.pdf                      # every page, boxes as JSON on stdout
    python predict.py sheet.pdf --pages 1 3 --draw out/   # pages 1 and 3, plus a picture of each
    python predict.py scan.png --draw out/           # an image works too

Each box comes back in pixels of the rendered page and as fractions of its width and height (0 to 1),
so it can be drawn on a render of any size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):  # doclayout_yolo 0.0.4 still calls np.trapz, removed in NumPy 2
    np.trapz = np.trapezoid

WEIGHTS = Path(__file__).resolve().parent / "weights" / "sheet_regions.pt"
LONG_EDGE = 2400   # pages are rendered so their longest side is this many pixels, as in training
IMGSZ = 768        # the model's input size
HIDDEN = {"unknown"}  # a small mark that recurs at the edge of many training sheets; not useful
COLOURS = {"drawing": (31, 119, 180), "legend": (214, 39, 40), "table": (44, 160, 44),
           "notes": (255, 127, 14), "abbreviation": (148, 103, 189), "title_block": (23, 190, 207)}


def load_model(weights: Path = WEIGHTS):
    import torch
    from doclayout_yolo import YOLOv10

    real = torch.load  # torch >= 2.6 loads weights_only by default; this file holds a whole model
    torch.load = lambda *a, **k: real(*a, **{**k, "weights_only": False})
    try:
        return YOLOv10(str(weights))
    finally:
        torch.load = real


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pages_of(path: Path, wanted: list[int] | None):
    """Yield (page number from 1, RGB image) for a PDF or an image file."""
    from PIL import Image

    if path.suffix.lower() != ".pdf":
        img = Image.open(path).convert("RGB")
        s = LONG_EDGE / max(img.size)
        yield 1, img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        return
    import pymupdf

    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            if wanted and i not in wanted:
                continue
            s = LONG_EDGE / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False)
            yield i, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect(model, img, conf: float = 0.25, device: str = "cpu") -> list[dict]:
    res = model.predict(img, imgsz=IMGSZ, conf=conf, device=device, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        label = res.names[int(b.cls[0])]
        if label in HIDDEN:
            continue
        x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
        boxes.append({"label": label, "conf": round(float(b.conf[0]), 3),
                      "box_px": [round(x0), round(y0), round(x1), round(y1)],
                      "box": [round(x0 / img.width, 4), round(y0 / img.height, 4),
                              round(x1 / img.width, 4), round(y1 / img.height, 4)]})
    # one region sometimes gets two labels (say table 0.40 and abbreviation 0.36): keep the stronger
    # box and record the other label as a second opinion rather than drawing both
    kept: list[dict] = []
    for b in sorted(boxes, key=lambda d: -d["conf"]):
        for k in kept:
            if k["label"] != b["label"] and _iou(k["box_px"], b["box_px"]) >= 0.8:
                k.setdefault("also", []).append({"label": b["label"], "conf": b["conf"]})
                break
        else:
            kept.append(b)
    return kept


def draw(img, boxes: list[dict], out: Path, max_side: int = 1800) -> None:
    from PIL import ImageDraw, ImageFont

    s = min(1.0, max_side / max(img.size))
    im = img.resize((round(img.width * s), round(img.height * s))) if s < 1 else img.copy()
    d = ImageDraw.Draw(im, "RGBA")
    lw = max(3, round(max(im.size) / 450))
    try:
        font = ImageFont.load_default(size=max(14, round(max(im.size) / 90)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for b in sorted(boxes, key=lambda d: d["label"] == "drawing", reverse=True):
        c = COLOURS.get(b["label"], (90, 90, 90))
        x0, y0, x1, y1 = (v * s for v in b["box_px"])
        d.rectangle([x0, y0, x1, y1], outline=c + (255,), width=lw, fill=c + (28,))
        tag = f"{b['label']} {b['conf']:.2f}"
        tw, th = d.textbbox((0, 0), tag, font=font)[2:]
        ty = y0 - th - 6 if y0 - th - 6 > 0 else y0 + lw
        d.rectangle([x0, ty, x0 + tw + 8, ty + th + 6], fill=c + (235,))
        d.text((x0 + 4, ty + 2), tag, fill=(255, 255, 255), font=font)
    im.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", type=Path, help="a PDF, PNG or JPEG")
    ap.add_argument("--pages", type=int, nargs="*", help="PDF pages to read, counting from 1 (default: all)")
    ap.add_argument("--conf", type=float, default=0.25, help="lowest confidence kept (default 0.25)")
    ap.add_argument("--draw", type=Path, help="folder to write a picture of each page with its boxes")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: the fastest available)")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    a = ap.parse_args()
    model, device = load_model(a.weights), a.device or pick_device()
    if a.draw:
        a.draw.mkdir(parents=True, exist_ok=True)
    found = 0
    for n, img in pages_of(a.file, a.pages):
        boxes = detect(model, img, a.conf, device)
        found += 1
        print(json.dumps({"file": a.file.name, "page": n, "width": img.width, "height": img.height,
                          "boxes": boxes}))
        if a.draw:
            draw(img, boxes, a.draw / f"{a.file.stem}_p{n}.png")
    if not found:
        print("no pages read", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
