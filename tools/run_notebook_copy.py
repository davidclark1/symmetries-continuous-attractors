"""Run a disposable copy of a notebook with the regeneration switches flipped on.

The copy is written next to the original, inside ``notebooks/``, so that every
``../data`` and ``../figures`` relative path still resolves. It is removed when the
run finishes, whether or not the run succeeded.

Usage
-----
    conda run -n ring-local python tools/run_notebook_copy.py notebooks/spectra.ipynb
    conda run -n ring-local python tools/run_notebook_copy.py notebooks/spectra.ipynb --no-caches
    conda run -n ring-local python tools/run_notebook_copy.py notebooks/spectra.ipynb --keep-output

By default both ``SAVE_FIGURES`` and ``REGENERATE_CACHES`` are set to ``True`` in the
copy. The committed notebook is never modified.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


def flip_switches(nb, save_figures=True, regenerate_caches=True):
    """Set the two regeneration switches in every cell that assigns them."""
    found = {"SAVE_FIGURES": 0, "REGENERATE_CACHES": 0}
    wanted = {"SAVE_FIGURES": save_figures, "REGENERATE_CACHES": regenerate_caches}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new = src
        for name, value in wanted.items():
            pattern = rf"^(\s*){name}\s*=\s*(True|False)\s*$"

            def repl(m, name=name, value=value):
                found[name] += 1
                return f"{m.group(1)}{name} = {value}"

            new = re.sub(pattern, repl, new, flags=re.MULTILINE)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--no-figures", action="store_true",
                    help="leave SAVE_FIGURES False")
    ap.add_argument("--no-caches", action="store_true",
                    help="leave REGENERATE_CACHES False")
    ap.add_argument("--keep-output", action="store_true",
                    help="keep the executed copy instead of deleting it")
    ap.add_argument("--timeout", type=int, default=100000)
    args = ap.parse_args()

    src_path = os.path.abspath(args.notebook)
    directory = os.path.dirname(src_path)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    copy_path = os.path.join(directory, f"_disposable_{stem}.ipynb")

    nb = json.load(open(src_path))
    found = flip_switches(nb,
                          save_figures=not args.no_figures,
                          regenerate_caches=not args.no_caches)
    with open(copy_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"[run_notebook_copy] {os.path.basename(src_path)} -> {os.path.basename(copy_path)}")
    print(f"[run_notebook_copy] switches found: {found}")
    sys.stdout.flush()

    t0 = time.time()
    rc = 1
    try:
        proc = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             f"--ExecutePreprocessor.timeout={args.timeout}", copy_path],
            cwd=directory)
        rc = proc.returncode
    finally:
        elapsed = time.time() - t0
        print(f"[run_notebook_copy] rc={rc} elapsed={elapsed:.1f}s ({elapsed/60:.1f} min)")
        if not args.keep_output and os.path.exists(copy_path):
            os.remove(copy_path)
        checkpoints = os.path.join(directory, ".ipynb_checkpoints")
        if os.path.isdir(checkpoints):
            shutil.rmtree(checkpoints, ignore_errors=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
