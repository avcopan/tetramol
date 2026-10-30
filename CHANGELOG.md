# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]
### Added
- `Builder`, which places atoms on a tetrahedral "diamond" grid with ideal bond lengths (1.1 Å to H, 1.5 Å otherwise) and angles, choosing bond slots by index or by direction (`toward`), and filling valences with `add_hydrogens`.
- `Builder.umbrella_image`, which adds the mirror image of a branch (e.g. the H's of an amine) to draw an umbrella inversion.
- `tetrahedral_basis(up, left)` and the `TETRAHEDRAL_ANGLE`, `XH_LENGTH`, and `XY_LENGTH` constants.
- `render`, which renders a geometry as an SVG in a flat house style (`STYLE`) with xyzrender.
- `orient`, which opens a geometry in py3Dmol in the browser and returns the rotation you set by hand, printing it as code to paste into a drawing script.
- `drawings/`, with methane, methylamine, tert-butanol, and ethane-1,1,1-triamine.

### Removed
- The `test-lowest` environment, since xyzrender's unbounded transitive PyPI dependencies don't solve to their lowest versions.
