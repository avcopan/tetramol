"""tert-Butanol, with the C-O bond pointing up and the O-H tilted right."""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, orient, render

NAME = "tert-butanol"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Build: C-O up, a methyl tripod with one methyl to the left, and the O-H in
# the same plane, pointing right (anti to the left methyl)
b = Builder(up=(0, 0, 1), left=(-1, 0, 0))
c = b.add("C")
o = b.add("O", c, toward=(0, 0, 1))
b.add("H", o, toward=(1, 0, 0))
for slot in b.open_slots(c):
    b.add("C", c, slot)
b.add_hydrogens()
geo = b.geometry()

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.from_euler("x", 40, degrees=True) * Rotation.from_euler(
    "xy", [-90, -30], degrees=True
)
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{NAME}.xyz")
render(geo, HERE / f"{NAME}.svg")
