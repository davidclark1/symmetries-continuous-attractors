"""
Pixel-level equivalence check for matplotlib-produced PNGs and SVGs.

PNGs: load both, compare numpy arrays.
SVGs: extract base64-embedded PNG payload (matplotlib raster-fallback output) and
      compare the embedded raster pixels. Pure-vector SVGs without embedded raster
      are reported as VECTOR-ONLY (text diff is the only signal).

Usage:
    python tools/check_figure_equivalence.py <baseline_path> <regen_path>

Exit code 0 = pixel-exact match (or pure-vector text-equal), 1 otherwise.
"""
import sys
import re
import io
import base64
import numpy as np
from PIL import Image


def extract_embedded_png(svg_text):
    m = re.search(r'iVBORw0K[A-Za-z0-9+/=]+', svg_text)
    return base64.b64decode(m.group(0)) if m else None


def compare_pngs(a_bytes, b_bytes):
    a = np.array(Image.open(io.BytesIO(a_bytes)))
    b = np.array(Image.open(io.BytesIO(b_bytes)))
    if a.shape != b.shape:
        return None, f"shape differ: {a.shape} vs {b.shape}"
    d = np.abs(a.astype(int) - b.astype(int))
    pct = float((d.sum(axis=-1) > 0).mean() * 100) if a.ndim == 3 else float((d > 0).mean() * 100)
    maxch = int(d.max())
    return (pct, maxch), f"shape={a.shape}, pix_diff={pct:.4f}%, max_ch_diff={maxch}"


def compare(base_path, new_path):
    base_size = open(base_path, 'rb').seek(0, 2)
    print(f"baseline: {base_path}")
    print(f"regen:    {new_path}")
    print()

    if base_path.endswith('.png'):
        with open(base_path, 'rb') as f: a = f.read()
        with open(new_path, 'rb') as f: b = f.read()
        result, msg = compare_pngs(a, b)
        print(f"PNG: {msg}")
        if result is None:
            return 1
        pct, _ = result
        return 0 if pct == 0 else 1

    elif base_path.endswith('.svg'):
        with open(base_path) as f: bs = f.read()
        with open(new_path) as f: ns = f.read()
        text_eq = (bs == ns)
        print(f"text-equal: {text_eq}")
        bp = extract_embedded_png(bs)
        np_ = extract_embedded_png(ns)
        if bp is None or np_ is None:
            print("VECTOR-ONLY SVG (no embedded raster); text diff is only signal")
            return 0 if text_eq else 1
        result, msg = compare_pngs(bp, np_)
        print(f"embedded raster: {msg}")
        if result is None:
            return 1
        pct, _ = result
        return 0 if pct == 0 else 1

    else:
        with open(base_path, 'rb') as f: a = f.read()
        with open(new_path, 'rb') as f: b = f.read()
        print(f"byte-equal: {a == b}")
        return 0 if a == b else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
