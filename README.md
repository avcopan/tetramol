# tetramol

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)

Idealized molecular structures for illustrations, rendered as SVGs with
[xyzrender](https://github.com/aligfellow/xyzrender).

Atoms are placed on a tetrahedral "diamond" grid, with ideal bond lengths (1.1 Å to
hydrogen, 1.5 Å otherwise) and angles (arccos(-1/3) ≈ 109.47°), so neighboring
centers are always staggered. This is meant for clean, schematic pictures, not for
real geometries.

## Installation

```sh
pixi run init      # first time only
pixi run local start   # use the local ../automol checkout (see below)
```

tetramol depends on [automol](https://github.com/avcopan/automol) (from the `avcopan`
conda channel) for geometry handling, and on xyzrender (from PyPI) for rendering.
To develop against a local automol checkout next to this repository, toggle it with
`pixi run local start` and `pixi run local stop`.

## Usage

```python
from automol import geom
from tetramol import Builder, orient, render

b = Builder(up=(0, 1, 0), left=(-1, 0, 0))  # y is up the page, z toward you
c = b.add("C")  # the first atom goes at the origin
n = b.add("N", c, toward=(0, 1, 0))  # bond through the slot closest to a direction
b.add("C", c, toward=(-1, 0, 0))  # ... or through a slot by index: b.add("C", c, 2)
b.add_hydrogens(avoid=(0, 0, -1))  # fill valences; N's lone pair points back
geo = b.geometry()  # an automol Geometry

rot = orient(geo)  # optional: rotate by hand in the browser,
geo = geom.rotate(geo, rot)  # ... and apply the rotation
render(geo, "drawing.svg")
```

Each atom has four slots, one per tetrahedral direction, and bonded atoms bond back
through the same slot, which fills in the diamond lattice. Use `automol.geom` for
anything off the grid (e.g. `geom.rotate(geo, rot, origin=..., keys=[...])`).

To draw an umbrella inversion, `b.umbrella_image(n)` adds the mirror image of the
branch on atom `n` (through the plane of its parent bond, perpendicular to its lone
pair), keeping both, so that one side can be faded afterwards.

Drawings live in [`drawings/`](https://github.com/avcopan/tetramol/tree/main/drawings), one folder per molecule.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Credits

This package was created with [copier](https://copier.readthedocs.io/) and the [avcopan/copier-pixi-python](https://github.com/avcopan/copier-pixi-python) project template.
