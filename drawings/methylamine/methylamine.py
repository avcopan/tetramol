"""Methylamine, with the C-N bond pointing up the page and the lone pair to the back."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, orient, render

NAME = "methylamine"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Build: the drawing looks down z with y up, so up=(0, 1, 0) is up the page
b = Builder(up=(0, 1, 0), left=(-1, 0, 0))
c = b.add("C")
n = b.add("N", c, toward=(0, 1, 0))
b.add_hydrogens([n], avoid=(0, 0, -1))  # leave the slot pointing back empty
b.add_hydrogens([c])
geo = b.geometry()

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.identity()
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{NAME}.xyz")
render(geo, HERE / f"{NAME}.svg")
