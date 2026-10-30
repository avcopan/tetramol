"""Ethane-1,1,1-triamine, with both amine inversion images drawn on each N.

Each amine keeps the H pointing down and one upward H, leaving its third slot
(the lone pair) empty. `umbrella_image` then adds the mirror images of both H's,
to show the amine inversion (fade one side by hand afterwards).
"""

# %%
from pathlib import Path

from automol import geom
from scipy.spatial.transform import Rotation

from tetramol import Builder, orient, render

NAME = "ethane-1,1,1-triamine"
# `__file__` is undefined when running cells in a REPL started at the repo root
HERE = Path(globals().get("__file__", f"drawings/{NAME}/_")).parent

# %% Build: C-C up, then an amine tripod, each with the H in slot 0 pointing down
b = Builder(up=(0, 0, 1), left=(-1, 0, 0))
c = b.add("C")
c_top = b.add("C", c, 0)
b.add_hydrogens([c_top])

amines = []
for slot in (1, 2, 3):
    n = b.add("N", c, slot)
    keep = (slot + 1) % 3 + 1  # the previous slot; the next one is the lone pair
    b.add("H", n, 0)
    b.add("H", n, keep)
    amines.append(n)

# Add the umbrella-inversion images of the amine H's
geo = b.geometry()
for n in amines:
    geo, _ = b.umbrella_image(n, geo=geo)

# %% Orient: to change it, run `orient(geo)` and paste the printed rotation here
rot = Rotation.from_euler("x", 40, degrees=True) * Rotation.from_euler(
    "xy", [-90, -30], degrees=True
)
geo = geom.rotate(geo, rot)

# %% Write
geo.xyz_file(path=HERE / f"{NAME}.xyz")
render(geo, HERE / f"{NAME}.svg")
