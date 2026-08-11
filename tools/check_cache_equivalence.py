"""
Compare a cache you rebuilt against the one released on Zenodo.

Run it after rerunning a notebook with REGENERATE_CACHES = True, to check that
your rebuilt file matches ours:

    python tools/check_cache_equivalence.py released.npz mine.npz

It walks both files key by key and reports any key that appears in only one of
them, then the shape, dtype and exact equality of each shared key. Where a key is
not exactly equal it also reports the largest absolute and relative differences,
and whether numpy considers the arrays close at its default tolerance.

Exit code 0 means every key matches exactly and 1 means at least one does not. A
few cached arrays are fresh random draws rather than functions of the data, so
they are expected to differ; FIGURES.md says which.
"""
import sys
import numpy as np


def compare(orig_path, regen_path):
    orig = np.load(orig_path, allow_pickle=False)
    regen = np.load(regen_path, allow_pickle=False)

    orig_keys = set(orig.files)
    regen_keys = set(regen.files)

    only_orig = orig_keys - regen_keys
    only_regen = regen_keys - orig_keys
    common = sorted(orig_keys & regen_keys)

    print(f"orig:  {orig_path}")
    print(f"regen: {regen_path}")
    print(f"orig keys:  {sorted(orig_keys)}")
    print(f"regen keys: {sorted(regen_keys)}")
    if only_orig:
        print(f"  ONLY IN ORIG:  {sorted(only_orig)}")
    if only_regen:
        print(f"  ONLY IN REGEN: {sorted(only_regen)}")

    all_exact = (not only_orig) and (not only_regen)
    print()
    print(f"{'key':<20s} {'shape':<20s} {'dtype':<10s} {'exact':<7s} {'allclose':<10s} {'maxabs':<12s} {'maxrel':<12s}")
    print("-" * 100)

    for k in common:
        a, b = orig[k], regen[k]
        shape_match = a.shape == b.shape
        dtype_match = a.dtype == b.dtype
        if not shape_match:
            print(f"{k:<20s} SHAPE MISMATCH orig={a.shape} regen={b.shape}")
            all_exact = False
            continue
        # equal_nan=True treats matching-NaN positions as equal, which these arrays
        # need because NaN is a meaningful value in them (wraparound markers).
        exact = bool(np.array_equal(a, b, equal_nan=True)) if np.issubdtype(a.dtype, np.floating) else bool(np.array_equal(a, b))
        if exact:
            print(f"{k:<20s} {str(a.shape):<20s} {str(a.dtype):<10s} {'YES':<7s}")
            continue
        all_exact = False
        if np.issubdtype(a.dtype, np.number):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            maxabs = float(diff.max()) if diff.size else 0.0
            denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(denom > 0, diff / denom, 0.0)
            maxrel = float(rel.max()) if rel.size else 0.0
            close = bool(np.allclose(a, b))
            print(f"{k:<20s} {str(a.shape):<20s} {str(a.dtype):<10s} {'NO':<7s} {('YES' if close else 'NO'):<10s} {maxabs:<12.4e} {maxrel:<12.4e}")
        else:
            print(f"{k:<20s} {str(a.shape):<20s} {str(a.dtype):<10s} {'NO':<7s} (non-numeric)")

    print()
    print("RESULT: every key matches exactly" if all_exact
          else "RESULT: the files differ, see the per-key lines above")
    return 0 if all_exact else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
