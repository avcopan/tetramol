"""Umbrella-inversion images of any geometry, e.g. for an amine.

Unlike `Builder.umbrella_image`, these work on a plain geometry in its final
orientation, so they also work for CREST conformers, and they can turn the group
so that its mirror plane is seen edge-on.
"""

import numpy as np
from automol import Geometry, geom
from scipy.sparse.csgraph import connected_components

VIEW_AXIS = np.array([0.0, 0.0, 1.0])
"""The direction toward the viewer in `render`."""


def umbrella_image(
    geo: Geometry, key: int, *, parent: int | None = None
) -> tuple[Geometry, list[int]]:
    """Add the mirror image of the group on an atom, e.g. to draw an amine inversion.

    The group (everything bonded beyond `key`, away from `parent`) is reflected
    through the plane that contains the `parent`-`key` bond and is perpendicular
    to the lone pair of `key`. This swaps the sides of the substituents and the
    lone pair. Both the group and its image are kept, so one can be faded.

    Args:
        geo: Geometry.
        key: Center of the group, e.g. an amine nitrogen.
        parent: The atom on the other side of the bond. Defaults to the only
            heavy atom bonded to `key`.

    Returns:
        The geometry with the image atoms appended, and their indices.

    Raises:
        ValueError: If `parent` isn't given and `key` doesn't have exactly one
            heavy-atom neighbor.
    """
    parent, branch = _group(geo, key, parent)
    image = geom.reflect(
        geom.select_atoms(geo, branch),
        _mirror_normal(geo, key, parent),
        origin=geo.coordinates[key],
    )
    keys = list(range(geo.atom_count, geo.atom_count + len(branch)))
    return geom.join([geo, image]), keys


def umbrella_edge_on(geo: Geometry, key: int, *, parent: int | None = None) -> Geometry:
    """Twist the group on an atom so that its umbrella mirror plane is seen edge-on.

    The mirror plane of `umbrella_image` contains the `parent`-`key` bond. Of the
    planes that do, one also contains the view axis (z, see `render`), so it is
    seen edge-on, with each atom and its image at the same depth. Both planes
    contain the bond, so they differ by a twist about it: the signed angle
    between their normals. The group is twisted by the smaller of the two such
    angles (the normal can point either way). Bond lengths and angles are kept.

    Args:
        geo: Geometry, in its final orientation.
        key: Center of the group, e.g. an amine nitrogen.
        parent: The atom on the other side of the bond (see `umbrella_image`).

    Returns:
        The geometry with the group twisted.

    Raises:
        ValueError: If the bond points straight at the viewer, so that no mirror
            plane containing it is seen edge-on.
    """
    parent, _ = _group(geo, key, parent)
    axis = geo.coordinates[key] - geo.coordinates[parent]
    edge_on_normal = np.cross(axis, VIEW_AXIS)
    if np.isclose(np.linalg.norm(edge_on_normal), 0.0):
        msg = f"The bond ({parent}, {key}) points at the viewer: no edge-on plane."
        raise ValueError(msg)
    normal = _mirror_normal(geo, key, parent)
    angle = geom.twist_angle(geo, (parent, key), normal, edge_on_normal)
    if abs(angle) > np.pi / 2:  # the plane's normal can point either way
        angle -= np.copysign(np.pi, angle)
    return geom.twist(geo, (parent, key), angle)


def _group(geo: Geometry, key: int, parent: int | None) -> tuple[int, list[int]]:
    """Get the parent of `key` and the atoms bonded beyond `key`, away from it."""
    amat = geom.adjacency_matrix(geo)
    if parent is None:
        heavy = [k for k in np.flatnonzero(amat[key]) if geo.symbols[k] != "H"]
        if len(heavy) != 1:
            msg = (
                f"Atom {key} has heavy-atom neighbors {heavy}; give the `parent` "
                "on the other side of the bond."
            )
            raise ValueError(msg)
        parent = int(heavy[0])
    amat[key, parent] = amat[parent, key] = 0
    _, labels = connected_components(amat, directed=False)
    branch = [k for k in np.flatnonzero(labels == labels[key]) if k != key]
    return parent, [int(k) for k in branch]


def _mirror_normal(geo: Geometry, key: int, parent: int) -> np.ndarray:
    """Get the mirror plane normal: the lone pair, minus its part along the bond.

    The lone pair points opposite the sum of the unit bond vectors of `key`,
    which is exact for a tetrahedral center with three neighbors.
    """
    amat = geom.adjacency_matrix(geo)
    bonds = geo.coordinates[np.flatnonzero(amat[key])] - geo.coordinates[key]
    lone_pair = -np.sum(bonds / np.linalg.norm(bonds, axis=1, keepdims=True), axis=0)
    axis = geo.coordinates[parent] - geo.coordinates[key]
    axis /= np.linalg.norm(axis)
    return lone_pair - np.dot(lone_pair, axis) * axis
