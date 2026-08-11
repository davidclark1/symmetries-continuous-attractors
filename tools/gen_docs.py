"""Generate the figure page from docs/figure_manifest.yaml.

The manifest is the source of truth for which notebook produces which figure.
This script renders it, so the page and the manifest cannot drift apart. Re-run
after editing the manifest:

    python tools/gen_docs.py

Reads:  docs/figure_manifest.yaml
Writes: FIGURES.md (repo root, for browsing on GitHub)
        docs/figures.md (the same page, for the documentation site)
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "figure_manifest.yaml"


def _order(e):
    """Sort main figures 1-8 first, then supplementary S1-S16.

    Supplementary ids are strings like "S10", so a plain string sort would put
    S10 before S2.
    """
    f = e["figure"]
    if isinstance(f, int):
        return (0, f)
    s = str(f)
    if s.startswith("S") and s[1:].isdigit():
        return (1, int(s[1:]))
    raise ValueError(f"unrecognised figure id: {f!r}")


def load():
    return sorted(yaml.safe_load(MANIFEST.read_text()), key=_order)


def anchor(e):
    """The heading anchor GitHub and MkDocs both generate for this figure."""
    return f"#figure-{str(e['figure']).lower()}"


def entry(e):
    """Render one figure's section."""
    L = [f"### Figure {e['figure']}\n", f"{e['title']}.\n"]
    L.append(f"- **Image:** `{e['file']}`")
    L.append(f"- **Notebook:** `{e['notebook']}`")
    L.append(f"- **Cell:** the one containing `{e['cell']}`")

    caches = e.get("caches") or []
    if caches:
        L.append("- **Reads:** " + ", ".join(f"`{c}`" for c in caches))
    else:
        L.append("- **Reads:** nothing; everything it needs is computed in the notebook")

    panels = e.get("panels") or []
    if panels:
        L.append("- **Panels:**")
        for p in panels:
            L.append(f"    - {p}")

    if e.get("note"):
        L.append("")
        L.append(e["note"].strip())

    if e.get("regen"):
        L.append("")
        L.append(f"*On rerunning:* {e['regen'].strip()}")

    L.append("")
    return L


def page(figs, manifest_link):
    main = [e for e in figs if isinstance(e["figure"], int)]
    supp = [e for e in figs if not isinstance(e["figure"], int)]

    L = ["# Figures\n"]
    L.append(f"Generated from [`{manifest_link}`]({manifest_link}) by `tools/gen_docs.py`. "
             "Edit the manifest, not this page.\n")
    L.append("Every figure in the paper, the notebook that draws it, and what that notebook "
             "reads. To reproduce one, install the package, unpack the cached results into "
             "`data/`, and run the notebook from top to bottom; the figure appears inline. "
             "Set `SAVE_FIGURES = True` at the top of the notebook to write it to `figures/` "
             "as well.\n")

    L.append("| Figure | What it shows | Notebook |")
    L.append("|--------|---------------|----------|")
    for e in figs:
        nb = e["notebook"].replace("notebooks/", "")
        L.append(f"| [{e['figure']}]({anchor(e)}) | {e['title']} | `{nb}` |")
    L.append("")

    L.append("## Main text\n")
    for e in main:
        L += entry(e)
    L.append("## Supplementary figures\n")
    for e in supp:
        L += entry(e)

    return "\n".join(L) + "\n"


def main():
    figs = load()
    (ROOT / "FIGURES.md").write_text(page(figs, "docs/figure_manifest.yaml"))
    (ROOT / "docs" / "figures.md").write_text(page(figs, "figure_manifest.yaml"))
    print(f"wrote FIGURES.md and docs/figures.md from {len(figs)} figures")


if __name__ == "__main__":
    main()
