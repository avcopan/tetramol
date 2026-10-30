"""Methane, tilted slightly so that all four hydrogens are visible."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, orient, render

NAME = "methane"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Build: a C-H bond pointing up
b = Builder(up=(0, 0, 1), left=(-1, 0, 0))
c = b.add("C")
b.add_hydrogens()
geo = b.geometry()

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.from_euler("yx", [-10, 20], degrees=True)
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{NAME}.xyz")
render(geo, HERE / f"{NAME}.svg")
