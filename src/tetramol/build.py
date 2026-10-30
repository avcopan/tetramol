"""Build idealized molecules on a tetrahedral grid."""

from collections.abc import Collection, Sequence

import numpy as np
from automol import Geometry, geom
from automol.utils import element
from numpy.typing import ArrayLike

from .grid import XH_LENGTH, XY_LENGTH, tetrahedral_basis

VALENCES = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}
"""Default number of bonds to each element, used by `Builder.add_hydrogens`."""

_CLASH_DISTANCE = 0.5


class Builder:
    """Build a molecule atom by atom on a tetrahedral grid.

    Every atom has four bond slots, one per direction of the
    `tetrahedral_basis`. Bonded atoms have opposite parities: an atom of parity
    +1 bonds along the basis directions, and one of parity -1 along their
    negatives. A child added through slot ``i`` therefore bonds back to its
    parent through its own slot ``i``. Filling in carbons gives the diamond
    lattice, and neighboring centers are always staggered.

    Example:
        A C-N bond pointing up, with a methyl group to the left of the carbon:

        >>> b = Builder(up=(0, 0, 1), left=(-1, 0, 0))
        >>> c = b.add("C")
        >>> n = b.add("N", c, toward=(0, 0, 1))
        >>> c2 = b.add("C", c, toward=(-1, 0, 0))
        >>> b.add_hydrogens()
        [3, 4, 5, 6, 7, 8, 9]
        >>> b.geometry().symbols
        ['C', 'N', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
    """

    def __init__(
        self,
        up: ArrayLike = (0.0, 0.0, 1.0),
        left: ArrayLike = (-1.0, 0.0, 0.0),
        *,
        xh_length: float = XH_LENGTH,
        xy_length: float = XY_LENGTH,
    ) -> None:
        """Start an empty molecule.

        Args:
            up: Direction of slot 0 for the first atom (see `tetrahedral_basis`).
            left: Direction in the plane of slots 0 and 1 for the first atom.
            xh_length: Bond length to hydrogen, in Angstroms.
            xy_length: Bond length between heavy atoms, in Angstroms.
        """
        self.basis = tetrahedral_basis(up, left)
        self.xh_length = xh_length
        self.xy_length = xy_length
        self.symbols: list[str] = []
        self.positions: list[np.ndarray] = []
        self.parities: list[int] = []
        self.bonds: list[dict[int, int]] = []  # slot -> neighbor, per atom
        self.parents: list[int | None] = []

    @property
    def atom_count(self) -> int:
        """Number of atoms."""
        return len(self.symbols)

    def slot_direction(self, key: int, slot: int) -> np.ndarray:
        """Get the unit direction of a slot of an atom.

        Args:
            key: Atom index.
            slot: Slot index, from 0 to 3.

        Returns:
            The direction.
        """
        return self.parities[key] * self.basis[slot]

    def open_slots(self, key: int) -> list[int]:
        """List the empty slots of an atom.

        Args:
            key: Atom index.

        Returns:
            The empty slots, in order.
        """
        return [s for s in range(4) if s not in self.bonds[key]]

    def add(
        self,
        symbol: str,
        parent: int | None = None,
        slot: int | None = None,
        *,
        toward: ArrayLike | None = None,
        length: float | None = None,
    ) -> int:
        """Add an atom.

        Args:
            symbol: Atomic symbol.
            parent: Atom to bond to. Must be `None` for the first atom, which
                goes at the origin.
            slot: Slot of `parent` to bond through. If neither this nor
                `toward` is given, use the first empty slot.
            toward: Direction to bond in. The empty slot of `parent` closest
                to this direction is used.
            length: Bond length. Defaults to `xh_length` for bonds to hydrogen
                and `xy_length` otherwise.

        Returns:
            Index of the new atom.

        Raises:
            ValueError: If the parent, slot, or position is invalid.
        """
        symbol = element.symbol(symbol)
        if parent is None:
            if self.atom_count:
                msg = "Only the first atom can be added without a parent."
                raise ValueError(msg)
            return self._append(symbol, np.zeros(3), parity=1)

        slot = self._choose_slot(parent, slot, toward)
        if length is None:
            has_h = "H" in (symbol, self.symbols[parent])
            length = self.xh_length if has_h else self.xy_length
        position = self.positions[parent] + length * self.slot_direction(parent, slot)
        self._check_clash(position)

        key = self._append(symbol, position, parity=-self.parities[parent])
        self.parents[key] = parent
        self.bonds[parent][slot] = key
        self.bonds[key][slot] = parent
        return key

    def add_hydrogens(
        self,
        keys: Collection[int] | None = None,
        *,
        avoid: ArrayLike | None = None,
    ) -> list[int]:
        """Fill atoms up to their default valence (see `VALENCES`) with hydrogens.

        Args:
            keys: Atoms to fill. Defaults to all heavy atoms.
            avoid: When an atom needs fewer hydrogens than it has empty slots
                (e.g. an amine nitrogen), leave the slots closest to this
                direction empty. By default, the last empty slots are left.

        Returns:
            Indices of the new hydrogens.
        """
        if keys is None:
            keys = [k for k, s in enumerate(self.symbols) if s != "H"]

        added = []
        for key in keys:
            count = VALENCES[self.symbols[key]] - len(self.bonds[key])
            slots = self.open_slots(key)
            if avoid is not None:
                slots.sort(key=lambda s: np.dot(self.slot_direction(key, s), avoid))
            added.extend(self.add("H", key, s) for s in slots[: max(count, 0)])
        return added

    def geometry(self, *, charge: int = 0, spin: int | None = None) -> Geometry:
        """Get the geometry.

        Args:
            charge: Charge.
            spin: Number of unpaired electrons. Defaults to the lowest spin
                consistent with the electron count (0 or 1).

        Returns:
            The geometry.
        """
        if spin is None:
            spin = (sum(map(element.number, self.symbols)) - charge) % 2
        return Geometry(
            symbols=self.symbols,
            coordinates=np.reshape(self.positions, (-1, 3)),
            charge=charge,
            spin=spin,
        )

    def umbrella_image(
        self,
        key: int,
        keys: Sequence[int] | None = None,
        *,
        geo: Geometry | None = None,
        normal: ArrayLike | None = None,
    ) -> tuple[Geometry, list[int]]:
        """Add the mirror image of a branch, e.g. to draw an umbrella inversion.

        The branch is reflected through a plane that contains the bond from
        `key` to its parent, and that is perpendicular to the empty slot of
        `key` (its lone pair). This swaps the sides of the substituents and
        the lone pair, as in the inversion of an amine. Both the branch and its
        image are kept, so one can be faded afterwards.

        Args:
            key: Center of the branch, e.g. an amine nitrogen.
            keys: Atoms to reflect. Defaults to everything bonded beyond `key`,
                away from its parent.
            geo: Geometry to add the image to, e.g. one that already has other
                images. Defaults to `geometry()`. Its first atoms must be the
                builder's.
            normal: Normal of the mirror plane through `key`, overriding the
                default.

        Returns:
            The geometry with the image atoms appended, and their indices.

        Raises:
            ValueError: If `key` has no parent, or, without `normal`, doesn't
                have exactly one empty slot.
        """
        parent = self.parents[key]
        if parent is None:
            msg = f"Atom {key} has no parent, so there is no bond to mirror about."
            raise ValueError(msg)
        if normal is None:
            empty = self.open_slots(key)
            if len(empty) != 1:
                msg = (
                    f"Atom {key} has empty slots {empty}; the mirror plane needs "
                    "exactly one (or pass `normal`)."
                )
                raise ValueError(msg)
            axis = self.positions[parent] - self.positions[key]
            axis /= np.linalg.norm(axis)
            lone_pair = self.slot_direction(key, empty[0])
            normal = lone_pair - np.dot(lone_pair, axis) * axis

        keys = self._branch(key) if keys is None else list(keys)
        geo = self.geometry() if geo is None else geo
        image = geom.reflect(
            geom.select_atoms(geo, keys), normal, origin=self.positions[key]
        )
        image_keys = list(range(geo.atom_count, geo.atom_count + len(keys)))
        return geom.join([geo, image]), image_keys

    def _branch(self, key: int) -> list[int]:
        """List the atoms bonded beyond `key`, away from its parent."""
        seen = {key, self.parents[key]}
        branch, stack = [], [key]
        while stack:
            for nbr in self.bonds[stack.pop()].values():
                if nbr not in seen:
                    seen.add(nbr)
                    branch.append(nbr)
                    stack.append(nbr)
        return sorted(branch)

    def _append(self, symbol: str, position: np.ndarray, parity: int) -> int:
        """Append an atom and return its index."""
        self.symbols.append(symbol)
        self.positions.append(position)
        self.parities.append(parity)
        self.bonds.append({})
        self.parents.append(None)
        return self.atom_count - 1

    def _choose_slot(
        self, parent: int, slot: int | None, toward: ArrayLike | None
    ) -> int:
        """Choose an empty slot of `parent` by index or direction."""
        if not 0 <= parent < self.atom_count:
            msg = f"No atom {parent} in a molecule of {self.atom_count} atoms."
            raise ValueError(msg)
        if slot is not None and toward is not None:
            msg = "Give either a slot or a direction (`toward`), not both."
            raise ValueError(msg)

        open_slots = self.open_slots(parent)
        if not open_slots:
            msg = f"Atom {parent} has no empty slots."
            raise ValueError(msg)
        if toward is not None:
            dirs = [self.slot_direction(parent, s) for s in open_slots]
            return open_slots[int(np.argmax(np.dot(dirs, toward)))]
        if slot is None:
            return open_slots[0]
        if slot not in open_slots:
            msg = f"Slot {slot} of atom {parent} is not empty (empty: {open_slots})."
            raise ValueError(msg)
        return slot

    def _check_clash(self, position: np.ndarray) -> None:
        """Raise `ValueError` if `position` is on top of an existing atom."""
        for key, other in enumerate(self.positions):
            if np.linalg.norm(position - other) < _CLASH_DISTANCE:
                msg = f"Position {position} is occupied by atom {key}."
                raise ValueError(msg)
