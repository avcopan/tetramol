# Drawings

One folder per molecule, with one script per method of making its geometry:

```
drawings/
  methylamine/
    ideal.py              # builds, orients, and renders the drawing
    ideal.xyz             # the oriented geometry
    ideal.svg             # the drawing
  tham/
    ideal.py
    crest.py
    crest-conformers.xyz  # CREST's conformers (a cache; delete it to rerun CREST)
    crest.xyz
    crest.svg
```

- **Folder names.** Use the molecule's common name if it has a simple, recognizable one (`tham`, `tert-butanol`), and its IUPAC name otherwise (`4-methylbicyclo[2.2.2]octan-1-amine`).
  Write it in lowercase with hyphens.
  For several molecules in one drawing (a complex or a reacting pair), join the names with `+`, e.g. `methane+hydroxyl`.
- **Methods.** Each script and its outputs are named after how the geometry is made:
  - `ideal`: an idealized geometry on the tetrahedral grid, built atom by atom with `Builder`.
  - `crest`: a conformer from a CREST search, with `crest.conformers`.

  For variants of one drawing, add a suffix, e.g. `ideal-inverted.svg`.
- **Scripts.** Each script is a `.py` file with `# %%` cells, to run cell by cell in a REPL or all at once with `pixi run -e dev python drawings/<name>/<method>.py`.
  Copy an existing script as a starting point.
- **Orienting.** `render` looks down the z axis with x to the right and y up, so `Builder(up=(0, 1, 0))` points "up" up the page.
  To rotate a molecule by hand, run `orient(geo)`: it opens the molecule in a browser (py3Dmol), and when you click "Done" it prints a `Rotation.from_quat([...])` line to paste into the script.
  The script then reproduces the drawing without the viewer.
- **Spinning about an axis.** To put a molecular axis along y, rotate by `geom.axis_alignment_rotation(geo, (i, j), axis=(0, 1, 0))`.
  Then choose a spin about y with `render_spins(geo, out)`, which renders several spins side by side in the final style, or with `orient(geo, axis=(0, 1, 0))`, which spins with a slider.
  See `4-methylbicyclo[2.2.2]octan-1-amine/crest.py`.
- **CREST.** `crest.conformers(smiles, path)` runs a CREST conformer search the first time, in a temporary directory, and caches the conformers at `path` (lowest energy first).
  It takes minutes to tens of minutes.
  CREST runs in its own Pixi environment (`crest`), because it needs an older libgfortran than SciPy.
- **Umbrella motions.** `b.umbrella_image(n)` adds the mirror image of the branch on atom `n` and keeps both sides; fade one in Inkscape.
  See `ethane-1,1,1-triamine/ideal.py`.
- **Paths.** Scripts write next to themselves, using `__file__`.
  In a REPL, where `__file__` is undefined, they assume the REPL was started at the repository root.
