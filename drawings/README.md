# Drawings

One folder per drawing, named after the molecule:

```
drawings/
  methylamine/
    methylamine.py     # builds, orients, and renders the drawing
    methylamine.xyz    # the oriented geometry
    methylamine.svg    # the drawing
```

- **Names.** Use the molecule's name, lowercase, with hyphens (`tert-butanol`, `ethane-1,1,1-triamine`).
  For several molecules in one drawing (a complex or a reacting pair), join the names with `+`, e.g. `methane+hydroxyl`.
  For variants of one drawing, add a suffix to the file names in the same folder, e.g. `methylamine-inverted.svg`.
- **Scripts.** Each script is a `.py` file with `# %%` cells, to run cell by cell in a REPL or all at once with `pixi run -e dev python drawings/<name>/<name>.py`.
  Copy an existing drawing as a starting point.
- **Orienting.** `render` looks down the z axis with x to the right and y up, so `Builder(up=(0, 1, 0))` points "up" up the page.
  To rotate a molecule by hand, run `orient(geo)`: it opens the molecule in a browser (py3Dmol), and when you click "Done" it prints a `Rotation.from_quat([...])` line to paste into the script.
  The script then reproduces the drawing without the viewer.
- **Umbrella motions.** `b.umbrella_image(n)` adds the mirror image of the branch on atom `n` and keeps both sides; fade one in Inkscape.
  See `ethane-1,1,1-triamine`.
- **Paths.** Scripts write next to themselves, using `__file__`.
  In a REPL, where `__file__` is undefined, they assume the REPL was started at the repository root.
