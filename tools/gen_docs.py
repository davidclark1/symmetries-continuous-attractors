"""Generate PROVENANCE.md and FIGURES.md from docs/figure_manifest.yaml.

Single source of truth -> two rendered docs, so they cannot drift. Re-run
after editing the manifest:

    python tools/gen_docs.py

Reads:  docs/figure_manifest.yaml
Writes: PROVENANCE.md, FIGURES.md   (repo root)
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "figure_manifest.yaml"

# Plain-language glossary of the terms these generated docs use, so a reader can
# understand a row without prior context. Only terms actually in use are shown, so
# the glossary cannot drift from the table.
_KIND_DEFS = {
    "direct": "a single notebook cell saves the whole figure; reproducible from the notebook alone.",
    "composite": "several panels are drawn by one notebook cell and assembled into the final image in code.",
    "response": "an additional figure produced by the notebooks; not a numbered figure in the paper.",
}
_VERDICT_DEFS = {
    "BIT-EXACT": "regenerates a byte-identical file.",
    "PIXEL-EXACT": "regenerates a pixel-identical image; only invisible file metadata differs.",
    "PDF-METADATA-ONLY": "identical image; only PDF timestamps and ids differ.",
    "TIER-3-STATISTICAL": "a different random draw with the same statistics, so it looks equivalent without being pixel-identical.",
    "NON-DETERMINISTIC": "small run-to-run variation remains even with a fixed seed; the per-figure note says why.",
}


def glossary(figs):
    kinds = {e.get("kind") for e in figs}
    verdicts = {(e.get("verification") or {}).get("verdict") for e in figs}
    L = ["## Terms used here\n", "**Figure kind**"]
    for k, v in _KIND_DEFS.items():
        if k in kinds:
            L.append(f"- *{k}* — {v}")
    L.append("\n**Verification verdict** — how faithfully re-running the code reproduces "
             "the committed figure:")
    for k, v in _VERDICT_DEFS.items():
        if k in verdicts:
            L.append(f"- *{k}* — {v}")
    L.append("\n**Other terms**")
    L.append("- *Producer* — the notebook, and specific cell, that generates a figure or panel.")
    L.append("- *Input caches* — precomputed results in `data/` that a notebook loads instead of recomputing.")
    if any((e.get("verification") or {}).get("f_fixes") for e in figs):
        L.append("- *F-fixes* — seed fixes that make a figure regenerate deterministically, "
                 "described in `EQUIVALENCE_LEDGER.md`.")
    return "\n".join(L) + "\n"


def _order(e):
    """Sort main figures 1-8, then supplementary S1-S16, then anything else.

    Supplementary ids are strings like "S10", so a plain string sort would put
    S10 before S2.
    """
    f = e["figure"]
    if isinstance(f, int):
        return (0, f, "")
    s = str(f)
    if s.startswith("S") and s[1:].isdigit():
        return (1, int(s[1:]), "")
    return (2, 0, s)


def load():
    return sorted(yaml.safe_load(MANIFEST.read_text()), key=_order)


def _is_paper(e):
    """True for figures the paper actually contains.

    Excludes the additional figures, and the two retired schematics whose
    artwork was merged into other figures; their files no longer exist, so
    listing them would send a reader looking for something that is not there.
    """
    return e.get("kind") not in ("response", "merged")


def gen_audit(figs, links):
    paper = [e for e in figs if _is_paper(e)]
    response = [e for e in figs if e.get("kind") == "response"]
    n = len(paper)
    kinds = {}
    verdicts = {}
    for e in paper:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        v = (e.get("verification") or {}).get("verdict", "?")
        verdicts[v] = verdicts.get(v, 0) + 1

    man = links["manifest"]
    L = []
    L.append("# Figure provenance\n")
    L.append(f"Auto-generated from [`{man}`]({man}) "
             "by `tools/gen_docs.py` — **do not edit by hand.**\n")
    L.append("Every paper figure mapped to its producer notebook and its verification verdict "
             f"from the equivalence-checking pass. See [figure recipes]({links['figures']}) for "
             "per-figure reproduction recipes.\n")

    L.append("## Summary\n")
    L.append(f"- **{n}** paper figures "
             f"({kinds.get('direct',0)} direct, {kinds.get('composite',0)} composite).")
    L.append("- Verdicts: " + ", ".join(f"{k} ({v})" for k, v in sorted(verdicts.items())) + ".\n")

    L.append(glossary(paper + response))

    L.append("## Figures\n")
    L.append("| Fig | Manuscript file | Producer | Verdict | Fixes |")
    L.append("|----:|-----------------|----------|---------|-------|")
    for e in paper:
        p = e.get("producer") or {}
        nb = p.get("notebook") or "—"
        nb_short = nb.replace("notebooks/", "") if nb != "—" else "—"
        v = e.get("verification") or {}
        verdict = v.get("verdict", "?")
        fixes = ", ".join(v.get("f_fixes") or []) or "—"
        L.append(f"| {e['figure']} | `{e['manuscript_file']}` | "
                 f"{'`'+nb_short+'`' if nb_short!='—' else '—'} | {verdict} | {fixes} |")
    L.append("")

    L.append("## Applied reproducibility fixes\n")
    L.append("The `Fixes` column references the seed fixes that make each figure regenerate "
             "deterministically. They are described in `EQUIVALENCE_LEDGER.md`.\n")

    if response:
        L.append("## Additional figures\n")
        L.append("Figures produced by the analysis notebooks that are not numbered figures "
                 "in the paper.\n")
        L.append("| Ref | Manuscript file | Producer | Verdict |")
        L.append("|-----|-----------------|----------|---------|")
        for e in response:
            p = e.get("producer") or {}
            nb = (p.get("notebook") or "—").replace("notebooks/", "")
            verdict = (e.get("verification") or {}).get("verdict", "?")
            L.append(f"| {e['figure']} | `{e['manuscript_file']}` | `{nb}` | {verdict} |")
        L.append("")
    return "\n".join(L) + "\n"


def _fig_recipe(e, header, caveats):
    """Render one figure's reproduction recipe (shared by paper + response figs)."""
    L = [f"{header}\n"]
    L.append(f"- **Manuscript file:** `{e['manuscript_file']}`")
    L.append(f"- **Kind:** {e['kind']}")

    p = e.get("producer") or {}
    if not p.get("notebook"):
        L.append("- **Producer:** none; no notebook reproduces this figure.")
    else:
        out = p.get("output")
        odir = p.get("output_dir") or "figures/"
        L.append(f"- **Producer notebook:** `{p['notebook']}`")
        if out:
            L.append(f"- **Output:** `{out}` (written to `{odir}`)")
        sig = p.get("cell_signature")
        if sig:
            L.append(f"- **Producer cell:** the cell containing `{sig}`")
        caches = e.get("input_caches") or []
        if caches:
            L.append("- **Input caches:** " + ", ".join(f"`{c}`" for c in caches))
        if out:
            L.append(f"- **Reproduce:** run `{p['notebook']}` top-to-bottom; "
                     f"the figure is written to `{odir}{out}`.")
        else:
            L.append(f"- **Reproduce:** run `{p['notebook']}` top-to-bottom to regenerate "
                     "the panels listed below.")

    panels = e.get("composite_panels") or []
    if panels:
        L.append("- **Composite panels:**")
        for panel in panels:
            L.append(f"    - {panel}")

    v = e.get("verification") or {}
    verdict = v.get("verdict", "?")
    entry = v.get("ledger_entry")
    fixes = ", ".join(v.get("f_fixes") or [])
    vline = f"- **Verification:** {verdict}"
    if entry:
        vline += f" ({entry})"
    if fixes:
        vline += f"; fixes: {fixes}"
    L.append(vline)

    note = v.get("notes")
    if note:
        L.append(f"- **Notes:** {note}")
        low = note.lower()
        if any(w in low for w in ("non-determin", "sign-flip", "sign degener",
                                  "not reproducible", "color-cycle", "patch version")):
            caveats.append((e["figure"], e["title"], note))
    L.append("")
    return L


def gen_figures(figs, links):
    man = links["manifest"]
    paper = [e for e in figs if _is_paper(e)]
    response = [e for e in figs if e.get("kind") == "response"]
    L = []
    L.append("# Figure reproduction recipes\n")
    L.append(f"Auto-generated from [`{man}`]({man}) "
             "by `tools/gen_docs.py` — **do not edit by hand.**\n")
    L.append("How each paper figure is produced. Notebooks live in `notebooks/`, importable "
             "helpers in the `ring/` package, cached inputs in `data/`, and figure outputs in "
             "`figures/` (see each figure's *Output*). Some composite-figure panels are "
             "intermediate files gated behind `SAVE_FIGURES` (default off) to protect the locked "
             "baseline; they are not kept on disk and are regenerated by setting "
             "`SAVE_FIGURES = True` and running the producer cell. Reproducibility caveats are collected at the "
             "[end](#reproducibility-caveats).\n")

    L.append(glossary(paper + response))

    caveats = []
    for e in paper:
        L += _fig_recipe(e, f"## Figure {e['figure']} — {e['title']}", caveats)

    L.append("## Reproducibility caveats\n")
    L.append("Figures whose exact pixels are not bit-reproducible from the notebook alone. "
             "Each was reviewed and accepted during the equivalence pass; details in "
             "`EQUIVALENCE_LEDGER.md`.\n")
    if caveats:
        for fig, title, note in caveats:
            L.append(f"- **Figure {fig} ({title}).** {note}")
    else:
        L.append("- None.")
    L.append("")

    if response:
        L.append("## Additional figures\n")
        L.append("Figures produced by the analysis notebooks that are not numbered figures "
                 "in the paper.\n")
        for e in response:
            L += _fig_recipe(e, f"### {e['figure']} — {e['title']}", caveats)
    return "\n".join(L) + "\n"


def main():
    figs = load()
    # Root copies (GitHub-browsable): sibling links + docs/ path to the manifest.
    root_links = {"figures": "FIGURES.md", "audit": "PROVENANCE.md",
                  "manifest": "docs/figure_manifest.yaml"}
    # docs/ copies (MkDocs site): lowercase page links + manifest is a sibling.
    site_links = {"figures": "figures.md", "audit": "provenance.md",
                  "manifest": "figure_manifest.yaml"}
    (ROOT / "PROVENANCE.md").write_text(gen_audit(figs, root_links))
    (ROOT / "FIGURES.md").write_text(gen_figures(figs, root_links))
    (ROOT / "docs" / "provenance.md").write_text(gen_audit(figs, site_links))
    (ROOT / "docs" / "figures.md").write_text(gen_figures(figs, site_links))
    print(f"wrote PROVENANCE.md + FIGURES.md (root + docs/) from {len(figs)} manifest entries")


if __name__ == "__main__":
    main()
