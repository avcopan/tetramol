"""Motion arrows on rendered drawings: torsions, umbrella inversions, rotations.

`add_arrows` draws arrows into an SVG made by `render`, or into one edited by
hand afterwards (moved, scaled, or with atoms faded). It needs the geometry that
was drawn, to know the 3D directions and depths, and it finds where each atom
landed on the page by fitting the atom centers to the SVG's circles. The fit is
an exact similarity transform (the render is orthographic), so the arrows are
exact projections of 3D circles:

- `Torsion(i, j)`: rotation about the bond i -> j (j on the moving side). A ring
  normal to the bond, by default the front half of a small one on the visible
  part of the bond.
- `Umbrella(key, images)`: inversion of the group on `key`, drawn after
  `umbrella_image`. One double-headed arrow along the path that one H takes to
  its image: a circle about `key`, crossing the mirror plane halfway.
- `ExternalRotation(i, j)`: overall rotation about the i -> j axis. An axis line beyond
  j, with a ring around it that the line passes through, between its back and
  front halves.

Arcs are layered by depth against the atoms and bonds they cover, and each
visible run is an Inkscape arc, so it can still be edited. Arrowheads are
fitted to the arcs (they point along the chord to the tip, not the end tangent),
which keeps them natural on tight curves. All lengths in `ArrowStyle` are in
Angstroms, so the same style works at any page scale.

Example:
    In a drawing script, after `render`::

        umbrella_geo, images = umbrella_image(geo, n)
        svg = render(umbrella_geo)
        arrows = [*torsions(umbrella_geo), Umbrella(n, images), ExternalRotation(c, n)]
        add_arrows(svg, umbrella_geo, arrows).save("ideal-arrows.svg")
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import xyzrender
from automol import Geometry, geom
from scipy.optimize import brentq

_MM = 1 / 7.0502618784215
"""Angstroms per mm on the page of the THAM graphic, where the house style was
designed (its lengths are given below in mm on that page)."""


@dataclass(frozen=True)
class Head:
    """A fitted arrowhead's size.

    Attributes:
        length: From the base to the tip, in Angstroms.
        width: Across the barbs, in Angstroms.
        notch: Depth of the concave back, as a fraction of the length.
    """

    length: float
    width: float
    notch: float


@dataclass(frozen=True)
class ArrowStyle:
    """How motion arrows are drawn.

    Lengths are in Angstroms (page lengths at the drawing's scale).

    The defaults are the house style. The head and the rotation ring match
    Inkscape's "Concave triangle arrow" on a 0.375 mm stroke, on a page at
    7.05 mm/Å. To change a field for one drawing, use e.g.
    ``dataclasses.replace(ARROW_STYLE, ring_radius=0.3)``.

    Attributes:
        color: Stroke and arrowhead color.
        stroke: Arc stroke width.
        head: Arrowhead size.
        ring_radius: Radius of torsion rings on bonds, and of the rotation ring.
        atom_ring_scale: Radius of torsion rings around atoms, as a multiple of
            the atom's drawn radius (``placement="atom"``).
        gap: For full torsion rings, the angle left open on each side of the
            moving atom's outer bond, in radians.
        front_trim: For front-half torsion rings, the angle trimmed off each end.
        sense: Turning sense of torsions about i -> j: +1 counterclockwise
            looking down the bond at j, -1 clockwise. It sets which end of each
            arc gets the arrowhead.
        piece: Longest piece of arc that is layered as one, in radians.
        layer_margin: How near (beyond its stroke) an arc must pass to an atom or
            bond to be layered against it.
        umbrella_clearance: How far outside the H discs the umbrella path runs.
        umbrella_margin: Smallest gap between an umbrella arrowhead and its H.
        rotation_angles: Start and end of the rotation arc, in radians from e1
            (across the projected axis) toward e2 (along it, beyond j).
        rotation_sense: +1 runs the rotation arc from the first angle to the
            second; -1 the other way.
        axis_stroke: Width of the rotation axis line.
        axis_overhang: How far the axis line runs past the ring at each end.
        axis_clearance: Gap between the axis line and anything drawn in its way.
    """

    color: str = "#0066ff"
    stroke: float = 0.375 * _MM
    head: Head = Head(2.165625 * _MM, 1.575 * _MM, 1.5 / 11)
    ring_radius: float = 2.6353059 * _MM
    atom_ring_scale: float = 1.6
    gap: float = float(np.radians(35))
    front_trim: float = float(np.radians(12))
    sense: int = -1
    piece: float = float(np.radians(15))
    layer_margin: float = 0.1 * _MM
    umbrella_clearance: float = 0.3 * _MM
    umbrella_margin: float = 0.6 * _MM
    rotation_angles: tuple[float, float] = (0.18046987, 6.022068026299929)
    rotation_sense: int = 1
    axis_stroke: float = 0.470001 * _MM
    axis_overhang: float = 1.8 * _MM
    axis_clearance: float = 1.5 * _MM


ARROW_STYLE = ArrowStyle()

_TWO_NEIGHBORS = 2  # a rotor's atoms each have a neighbor besides each other
_TOL = 1e-9
"""The house style for motion arrows."""


@dataclass(frozen=True)
class Torsion:
    """Rotation about the bond i -> j, with j on the moving side.

    Attributes:
        i: Atom on the fixed side.
        j: Atom on the moving side.
        placement: ``"bond-front"`` (the front half of a small ring on the
            visible part of the bond), ``"bond"`` (the whole small ring, open
            at the moving atom's outer bond), or ``"atom"`` (a ring around j).
        sense: Turning sense, overriding the style's.
    """

    i: int
    j: int
    placement: str = "bond-front"
    sense: int | None = None


@dataclass(frozen=True)
class Umbrella:
    """Umbrella inversion of the group on `key`, as drawn with `umbrella_image`.

    Attributes:
        key: Center of the group, e.g. an amine N.
        images: Keys of the image atoms (as returned by `umbrella_image`). Each
            is paired with the original it mirrors.
        side: Which pair to draw the arrow on: ``"back"`` (the farthest from
            the viewer) or ``"front"``.
    """

    key: int
    images: Sequence[int]
    side: str = "back"

    def __post_init__(self) -> None:
        """Store the images as a tuple of ints."""
        object.__setattr__(self, "images", tuple(int(k) for k in self.images))


@dataclass(frozen=True)
class ExternalRotation:
    """Overall rotation about the axis i -> j, drawn beyond j.

    Attributes:
        i: Atom at the start of the axis.
        j: Atom at the end of the axis; the symbol goes past it.
    """

    i: int
    j: int


Motion = Torsion | Umbrella | ExternalRotation


def torsions(
    geo: Geometry, *, placement: str = "bond-front", sense: int | None = None
) -> list[Torsion]:
    """A `Torsion` for every rotor of a geometry.

    The rotors are the bonds between two heavy atoms that both have other
    neighbors. Each torsion points toward its bond's smaller side.

    Args:
        geo: Geometry.
        placement: Placement of each torsion (see `Torsion`).
        sense: Turning sense of each torsion, overriding the style's.

    Returns:
        The torsions, in bond order.
    """
    neighbors = _neighbors(geo)
    found = []
    for bond in _bonds(geo):
        i, j = bond
        if (
            "H" in (geo.symbols[i], geo.symbols[j])
            or min(len(neighbors[i]), len(neighbors[j])) < _TWO_NEIGHBORS
        ):
            continue
        if len(_side(neighbors, i, j)) > len(_side(neighbors, j, i)):
            i, j = j, i
        found.append(Torsion(i, j, placement=placement, sense=sense))
    return found


def add_arrows(
    svg: str | Path | xyzrender.SVGResult,
    geo: Geometry,
    arrows: Sequence[Motion],
    *,
    style: ArrowStyle = ARROW_STYLE,
) -> xyzrender.SVGResult:
    """Draw motion arrows into a rendered SVG.

    The SVG may be straight from `render`, or edited afterwards, as long as each
    atom of `geo` is still a circle (moved or scaled is fine). Everything that
    isn't an arrow is kept byte for byte, except that the root gets the sodipodi
    namespace if it lacks it (the arcs are Inkscape arcs, to stay editable), and
    its viewBox grows to hold any arrow drawn outside it.

    Args:
        svg: The SVG, its text, or its path.
        geo: The geometry that was drawn, in the drawn orientation.
        arrows: The motions to draw (see `Torsion`, `Umbrella`, `ExternalRotation`).
        style: How to draw them.

    Returns:
        The SVG with the arrows.

    Raises:
        ValueError: If the atoms of `geo` don't match the SVG's circles.
    """
    text = Path(svg).read_text() if isinstance(svg, Path) else str(svg)
    drawing = _Drawing(text, geo, style)
    inserts: list[tuple[int, str]] = []
    drawn: list[np.ndarray] = []  # the arrows' extents so far, for `ExternalRotation`
    for motion in sorted(arrows, key=lambda m: isinstance(m, ExternalRotation)):
        if isinstance(motion, Torsion):
            new, points = drawing.torsion(motion)
        elif isinstance(motion, Umbrella):
            new, points = drawing.umbrella(motion)
        elif isinstance(motion, ExternalRotation):
            new, points = drawing.external_rotation(motion, drawn)
        else:
            msg = f"Unknown motion: {motion!r}"
            raise TypeError(msg)
        inserts += new
        drawn.append(points)
    for offset, piece in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:offset] + piece + text[offset:]
    if not inserts:
        return xyzrender.SVGResult(text)
    extents = np.vstack(drawn)
    return xyzrender.SVGResult(
        _fit_view(_declare_sodipodi(text), extents, drawing.scale)
    )


def _fit_view(text: str, extents: np.ndarray, scale: float) -> str:
    """Grow the root viewBox, if needed, to hold the arrows.

    The width and height grow in proportion, and the margin is 0.1 Å.
    """
    root = re.search(r"<svg\b[^>]*>", text)
    if root is None:
        return text
    view_box = re.search(r'viewBox="([^"]*)"', root.group(0))
    if view_box is None:
        return text
    x, y, w, h = (float(v) for v in view_box.group(1).replace(",", " ").split())
    pad = extents[:, 2] + 0.1 * scale
    lo = np.minimum([x, y], (extents[:, :2] - pad[:, None]).min(0))
    hi = np.maximum([x + w, y + h], (extents[:, :2] + pad[:, None]).max(0))
    if np.allclose(lo, [x, y]) and np.allclose(hi, [x + w, y + h]):
        return text
    tag = root.group(0)
    new_w, new_h = hi - lo
    tag = tag.replace(
        view_box.group(0), f'viewBox="{lo[0]:.1f} {lo[1]:.1f} {new_w:.1f} {new_h:.1f}"'
    )
    for name, factor in (("width", new_w / w), ("height", new_h / h)):
        size = re.search(rf'\s{name}="([\d.]+)([a-z%]*)"', tag)
        if size and size.group(2) != "%":
            tag = tag.replace(
                size.group(0),
                f' {name}="{float(size.group(1)) * factor:.1f}{size.group(2)}"',
            )
    return text[: root.start()] + tag + text[root.end() :]


_SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"


def _declare_sodipodi(text: str) -> str:
    """Declare the sodipodi namespace on the root, if needed (for Inkscape arcs)."""
    root = re.search(r"<svg\b[^>]*>", text)
    if root is None or "xmlns:sodipodi=" in root.group(0):
        return text
    end = root.start() + 4
    return text[:end] + f' xmlns:sodipodi="{_SODIPODI}"' + text[end:]


# Geometry helpers
def _bonds(geo: Geometry) -> list[tuple[int, int]]:
    amat = geom.adjacency_matrix(geo)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(amat)), strict=True)]


def _neighbors(geo: Geometry) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = {i: set() for i in range(geo.atom_count)}
    for i, j in _bonds(geo):
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def _side(neighbors: dict[int, set[int]], i: int, j: int) -> set[int]:
    """The atoms on j's side of the i-j bond."""
    seen, todo = {j}, [j]
    while todo:
        for m in neighbors[todo.pop()] - seen - {i}:
            seen.add(m)
            todo.append(m)
    return seen


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


@dataclass
class _Circle:
    """A 3D circle P(phi) = c + r (cos(phi) e1 + sin(phi) e2), normal to u.

    e1 = u x z lies in the page and e2 = u x e1, so (e1, e2, u) is right-handed
    and phi increases counterclockwise looking down u. On the page it's an
    ellipse with semi-axes r and r |u_z| along e1 and e2, and its depth relative
    to c is r sin(phi) e2_z.
    """

    center: np.ndarray
    radius: float
    u: np.ndarray
    e1: np.ndarray
    e2: np.ndarray

    @classmethod
    def about(cls, center: np.ndarray, u: np.ndarray, radius: float) -> "_Circle":
        e1 = _unit(np.cross(u, [0.0, 0.0, 1.0]))
        return cls(center, radius, u, e1, np.cross(u, e1))

    def at(self, phi: float | np.ndarray) -> np.ndarray:
        phi = np.asarray(phi)[..., None]
        return self.center + self.radius * (
            np.cos(phi) * self.e1 + np.sin(phi) * self.e2
        )

    def angle_of(self, vec: np.ndarray) -> float:
        return float(np.arctan2(vec @ self.e2, vec @ self.e1))


_ELEMENT = re.compile(r"<(line|circle|g|path|rect)\b[^>]*?/?>", re.DOTALL)


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf'\s{name}="([^"]*)"', tag)
    return match.group(1) if match else None


def _num(tag: str, name: str) -> float:
    """A numeric attribute of an SVG element."""
    value = _attr(tag, name)
    if value is None:
        msg = f"Expected a {name} attribute on {tag[:40]}..."
        raise ValueError(msg)
    return float(value)


def _segment_distance(page: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    t = np.clip((page - p) @ (q - p) / ((q - p) @ (q - p)), 0, 1)
    return np.linalg.norm(page - (p + t[:, None] * (q - p)), axis=1)


@dataclass
class _Occluder:
    """An atom's circle or a bond's line in the SVG, with its depth."""

    start: int
    end: int
    atom: int | None = None
    bond: tuple[int, int] | None = None
    center: np.ndarray | None = None
    radius: float = 0.0
    ends: tuple[np.ndarray, np.ndarray] | None = None
    width: float = 0.0


class _Drawing:
    """An SVG, its geometry, and the map between them."""

    def __init__(self, text: str, geo: Geometry, style: ArrowStyle) -> None:
        self.text, self.geo, self.style = text, geo, style
        self.xyz = np.asarray(geo.coordinates, dtype=float)
        defs = [
            (m.start(), m.end())
            for m in re.finditer(r"<defs\b.*?</defs>", text, re.DOTALL)
        ]
        self.elements = [
            (m.start(), m.end(), m.group(1), m.group(0))
            for m in _ELEMENT.finditer(text)
            if not any(a <= m.start() < b for a, b in defs)
        ]
        plain = [
            e
            for e in self.elements
            if _attr(e[3], "transform") is None and _attr(e[3], "clip-path") is None
        ]
        circles = [e for e in plain if e[2] == "circle"]
        if not circles:
            msg = "The SVG has no atom circles."
            raise ValueError(msg)
        centers = np.array([[_num(c[3], "cx"), _num(c[3], "cy")] for c in circles])
        self._fit(centers)
        self.atom_radius = np.array([_num(circles[k][3], "r") for k in self.circle_of])
        self.bonds = _bonds(geo)
        self.neighbors = _neighbors(geo)
        self.occluders = self._occluders(plain, circles)

    # The 3D -> page map
    def _fit(self, centers: np.ndarray) -> None:
        """Fit page = A @ (x, y) + t, matching each atom to its circle.

        Start from the render's orientation (x right, y down the page) with the
        scale and shift that match the spreads and centroids, then match each
        atom to its nearest circle and refit, a few times.
        """
        xy = self.xyz[:, :2]
        spread = np.sqrt(((centers - centers.mean(0)) ** 2).sum(1).mean())
        spread /= np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean())
        a_mat = spread * np.diag([1.0, -1.0])
        shift = centers.mean(0) - a_mat @ xy.mean(0)
        for _ in range(5):
            proj = xy @ a_mat.T + shift
            match = np.argmin(
                np.linalg.norm(proj[:, None] - centers[None], axis=-1), axis=1
            )
            design = np.hstack([xy, np.ones((len(xy), 1))])
            sol, *_ = np.linalg.lstsq(design, centers[match], rcond=None)
            a_mat, shift = sol[:2].T, sol[2]
        scale = float(np.sqrt(abs(np.linalg.det(a_mat))))
        residual = float(np.abs(xy @ a_mat.T + shift - centers[match]).max())
        if len(set(match.tolist())) < len(xy) or residual > 0.005 * scale:
            msg = (
                "The geometry doesn't match the SVG's circles "
                f"(off by {residual / scale:.3g} Å). "
                "Pass the geometry that was drawn, in the drawn orientation."
            )
            raise ValueError(msg)
        if not np.allclose(a_mat.T @ a_mat / scale**2, np.eye(2), atol=1e-3):
            msg = "The SVG is skewed or stretched relative to the geometry."
            raise ValueError(msg)
        self.a_mat, self.shift, self.scale, self.circle_of = a_mat, shift, scale, match

    def page(self, point: np.ndarray) -> np.ndarray:
        """Project a 3D point (or points) onto the page."""
        return np.asarray(point)[..., :2] @ self.a_mat.T + self.shift

    def _occluders(self, plain: list, circles: list) -> list[_Occluder]:
        atom_of = {int(k): a for a, k in enumerate(self.circle_of)}
        tol = 0.005 * self.scale
        found = []
        for start, end, kind, tag in plain:
            if kind == "circle":
                k = circles.index((start, end, kind, tag))
                if k in atom_of:
                    center = np.array([_num(tag, "cx"), _num(tag, "cy")])
                    found.append(
                        _Occluder(
                            start,
                            end,
                            atom=atom_of[k],
                            center=center,
                            radius=_num(tag, "r"),
                        )
                    )
            elif kind == "line" and _attr(tag, "x1") is not None:
                p = np.array([_num(tag, "x1"), _num(tag, "y1")])
                q = np.array([_num(tag, "x2"), _num(tag, "y2")])
                for i, j in self.bonds:
                    pi, pj = self.page(self.xyz[i]), self.page(self.xyz[j])
                    if _segment_distance(np.array([p, q]), pi, pj).max() < tol:
                        width = float(_attr(tag, "stroke-width") or 0)
                        found.append(
                            _Occluder(start, end, bond=(i, j), ends=(p, q), width=width)
                        )
                        break
        return found

    # Layering
    def _covers(self, occ: _Occluder, page: np.ndarray, margin: float) -> bool:
        if occ.atom is not None:
            return bool(
                (np.linalg.norm(page - occ.center, axis=1) < occ.radius + margin).any()
            )
        if occ.ends is None:
            msg = "A bond occluder needs its ends."
            raise ValueError(msg)
        p, q = occ.ends
        return bool((_segment_distance(page, p, q) < occ.width / 2 + margin).any())

    def _depth(self, occ: _Occluder, page: np.ndarray) -> float:
        if occ.atom is not None:
            return float(self.xyz[occ.atom, 2])
        if occ.bond is None:
            msg = "A bond occluder needs its bond."
            raise ValueError(msg)
        i, j = occ.bond
        pi, pj = self.page(self.xyz[i]), self.page(self.xyz[j])
        t = np.clip((page - pi) @ (pj - pi) / ((pj - pi) @ (pj - pi)), 0, 1)
        return float(np.mean(self.xyz[i, 2] + t * (self.xyz[j, 2] - self.xyz[i, 2])))

    def _slot(self, page: np.ndarray, depth: float) -> int | None:
        """Text offset for a piece of arc, by depth, or None if it covers nothing.

        The piece goes behind the atoms and bonds it covers that are in front of
        it, and in front of those behind it.
        """
        margin = (self.style.stroke / 2 + self.style.layer_margin) * self.scale
        hits = [o for o in self.occluders if self._covers(o, page, margin)]
        if not hits:
            return None
        depths = [self._depth(o, page) for o in hits]
        costs = [
            sum(d > depth for d in depths[:s]) + sum(d < depth for d in depths[s:])
            for s in range(len(hits) + 1)
        ]
        s = max(range(len(costs)), key=lambda s: (-costs[s], s))
        return hits[s].start if s < len(hits) else hits[-1].end

    # Drawing pieces
    def _page_at(self, circle: _Circle, phi: float) -> np.ndarray:
        return self.page(circle.at(phi))

    def _arc_path(
        self, circle: _Circle, a: float, b: float, ident: str, width: float
    ) -> str:
        """An arc from angle a to b, as an Inkscape arc in the ellipse's frame.

        Inkscape arcs run toward increasing angle, so a backward one is written as
        psi = -phi in the frame (e1, -e2), which traces the same points.
        """
        rx = self.scale * circle.radius
        ry = rx * abs(circle.u[2])
        f1 = self.a_mat @ circle.e1[:2] / self.scale
        f2 = _unit(self.a_mat @ circle.e2[:2])
        if b < a:
            a, b, f2 = -a, -b, -f2
        cx, cy = self.page(circle.center)
        x0, y0 = rx * np.cos(a), ry * np.sin(a)
        x1, y1 = rx * np.cos(b), ry * np.sin(b)
        style = (
            f"fill:none;stroke:{self.style.color};"
            f"stroke-width:{width * self.scale:.6g};"
            "stroke-linecap:round;stroke-linejoin:bevel;stroke-opacity:1"
        )
        return (
            f'<path style="{style}" id="{ident}" sodipodi:type="arc" '
            'sodipodi:cx="0" sodipodi:cy="0" '
            f'sodipodi:rx="{rx:.6f}" sodipodi:ry="{ry:.6f}" '
            f'sodipodi:start="{a % (2 * np.pi):.6f}" '
            f'sodipodi:end="{b % (2 * np.pi):.6f}" '
            'sodipodi:arc-type="arc" sodipodi:open="true" '
            f'd="M {x0:.6f},{y0:.6f} A {rx:.6f},{ry:.6f} 0 {int(b - a > np.pi)} 1 '
            f'{x1:.6f},{y1:.6f}" '
            f'transform="matrix({f1[0]:.6f},{f1[1]:.6f},{f2[0]:.6f},{f2[1]:.6f},'
            f'{cx:.6f},{cy:.6f})" />'
        )

    def _back_along(
        self, circle: _Circle, start: float, end: float, chord: float
    ) -> float:
        """The angle back from `end` whose page point is `chord` from the end point.

        It's the single root between `start` and `end`, since the distance only
        grows going back.
        """
        tip = self._page_at(circle, end)
        gap = lambda phi: np.linalg.norm(self._page_at(circle, phi) - tip) - chord  # noqa: E731
        if gap(start) <= 0:
            msg = "An arc is shorter than its head; use a bigger ring or smaller head."
            raise ValueError(msg)
        return float(brentq(gap, start, end, xtol=1e-12))

    def _head_points(self, circle: _Circle, start: float, end: float) -> np.ndarray:
        """The tip, base, and barbs of the fitted head at `end`."""
        head = self.style.head
        length = head.length * self.scale
        tip = self._page_at(circle, end)
        base = self._page_at(circle, self._back_along(circle, start, end, length))
        along = (tip - base) / length
        across = np.array([-along[1], along[0]])
        half = head.width * self.scale / 2
        return np.array([tip, base, base + half * across, base - half * across])

    def _head(
        self, circle: _Circle, start: float, end: float, ident: str
    ) -> tuple[str, float, np.ndarray]:
        """A head fitted to the arc's end.

        Returns its path, the angle to stop the stroke at (the notch), and its
        points.

        The tip is the arc's end and the base is on the same ellipse a head length
        back, so the head points along that chord and sits on the curve.
        """
        head = self.style.head
        length = head.length * self.scale
        tip, base, barb1, barb2 = self._head_points(circle, start, end)
        notch = base + head.notch * length * (tip - base) / length
        control = (
            2 * notch - base
        )  # the quadratic barb1 -> barb2 passes through the notch
        stop = self._back_along(circle, start, end, (1 - head.notch) * length)
        x = [f"{p[0]:.6f},{p[1]:.6f}" for p in (tip, barb1, control, barb2)]
        style = f"fill:{self.style.color};fill-opacity:1;stroke:none"
        d = f"M {x[0]} L {x[1]} Q {x[2]} {x[3]} Z"
        path = f'<path style="{style}" id="{ident}" d="{d}" />'
        return path, stop, np.array([tip, barb1, barb2])

    def _arc(
        self,
        circle: _Circle,
        ends: Sequence[float] | np.ndarray,
        ident: str,
        *,
        both_ends: bool = False,
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        """An arrow along a circle, with one or two fitted heads.

        The stroke is split into pieces layered by depth, and neighbors in the
        same layer are merged.
        """
        start, end = ends
        head, stop, points = self._head(circle, start, end, f"{ident}-head")
        tail = None
        if both_ends:
            tail, new_start, tail_points = self._head(
                circle, end, start, f"{ident}-tail"
            )
            points = np.vstack([points, tail_points])
            start = new_start
        end = stop
        count = int(np.ceil(abs(end - start) / self.style.piece))
        bounds = np.linspace(start, end, count + 1)
        slots = []
        for a, b in pairwise(bounds):
            samples = circle.at(np.linspace(a, b, 7))
            slots.append(self._slot(self.page(samples), float(samples[:, 2].mean())))
        known = [s for s in slots if s is not None] or [self.elements[-1][1]]
        for k in range(len(slots)):
            if slots[k] is None:
                slots[k] = next((s for s in slots[k::-1] if s is not None), known[0])
        runs: list[list] = []
        for k, slot in enumerate(slots):
            if runs and runs[-1][2] == slot:
                runs[-1][1] = k + 1
            else:
                runs.append([k, k + 1, slot])
        inserts = [
            (
                slot,
                self._arc_path(
                    circle, bounds[a], bounds[b], f"{ident}-{n}", self.style.stroke
                ),
            )
            for n, (a, b, slot) in enumerate(runs)
        ]
        inserts[-1] = (inserts[-1][0], inserts[-1][1] + head)
        if tail is not None:
            inserts[0] = (inserts[0][0], inserts[0][1] + tail)
        samples = self.page(circle.at(np.linspace(start, end, 101)))
        return inserts, self._extent(samples, points, self.style.stroke)

    def _extent(
        self, stroked: np.ndarray, filled: np.ndarray, stroke: float
    ) -> np.ndarray:
        """Page points of an arrow, each with how far its ink reaches past it.

        That's half the stroke for points on a stroke, and nothing for points on
        a filled head.
        """
        pads = np.r_[
            np.full(len(stroked), stroke * self.scale / 2), np.zeros(len(filled))
        ]
        return np.column_stack([np.vstack([stroked, filled]), pads])

    # Motions
    def torsion(self, motion: Torsion) -> tuple[list[tuple[int, str]], np.ndarray]:
        i, j = motion.i, motion.j
        style = self.style
        sense = style.sense if motion.sense is None else motion.sense
        u = _unit(self.xyz[j] - self.xyz[i])
        if motion.placement == "atom":
            radius = style.atom_ring_scale * self.atom_radius[j] / self.scale
            circle = _Circle.about(self.xyz[j], u, radius)
        elif motion.placement in ("bond", "bond-front"):
            circle = _Circle.about(self._bond_center(i, j), u, style.ring_radius)
        else:
            msg = f"Unknown torsion placement: {motion.placement!r}"
            raise ValueError(msg)
        if motion.placement == "bond-front":
            low = 0.0 if circle.e2[2] > 0 else np.pi
            ends = np.array([low + style.front_trim, low + np.pi - style.front_trim])
            ends = ends if sense > 0 else ends[::-1]
        else:
            phi = circle.angle_of(self.xyz[self._gap_atom(i, j)] - self.xyz[j])
            ends = phi + sense * (
                style.gap + np.array([0.0, 1.0]) * (2 * np.pi - 2 * style.gap)
            )
        return self._arc(circle, ends, f"torsion-{i}-{j}")

    def _bond_center(self, i: int, j: int) -> np.ndarray:
        """The middle of the bond's visible part.

        That's halfway between where it comes out of the front atom's disc and
        goes into the back atom's, or the front atom's edge, for a bond hidden
        behind it.
        """
        front, back = (i, j) if self.xyz[i, 2] >= self.xyz[j, 2] else (j, i)
        dist = np.linalg.norm(self.page(self.xyz[back]) - self.page(self.xyz[front]))
        s_f = self.atom_radius[front] / dist
        s_b = 1 - self.atom_radius[back] / dist
        s = (s_f + max(s_f, s_b)) / 2
        return self.xyz[front] + s * (self.xyz[back] - self.xyz[front])

    def _gap_atom(self, i: int, j: int) -> int:
        """The bond of j that a full ring starts and ends at.

        It's a heavy atom, else the front-most H.
        """
        subs = sorted(self.neighbors[j] - {i})
        heavy = [k for k in subs if self.geo.symbols[k] != "H"]
        return heavy[0] if heavy else max(subs, key=lambda k: self.xyz[k, 2])

    def umbrella(self, motion: Umbrella) -> tuple[list[tuple[int, str]], np.ndarray]:
        """One double-headed arrow along an H's inversion path.

        The path is a circle about the center, in the plane of the H and its
        image. It's scaled out so its ends clear the H discs, and shortened until
        its heads do too.
        """
        key, style = motion.key, self.style
        pairs = self._umbrella_pairs(key, list(motion.images))
        pick = min if motion.side == "back" else max
        h, image = pick(pairs, key=lambda p: self.xyz[p[1], 2])
        vh, vi = self.xyz[h] - self.xyz[key], self.xyz[image] - self.xyz[key]
        reach = np.linalg.norm(self.page(self.xyz[h]) - self.page(self.xyz[key]))
        k = 1 + (self.atom_radius[h] + style.umbrella_clearance * self.scale) / reach
        circle = _Circle.about(
            self.xyz[key], _unit(np.cross(vi, vh)), k * np.linalg.norm(vh)
        )
        phi_i, phi_h = circle.angle_of(vi), circle.angle_of(vh)
        phi_h += 2 * np.pi * np.round((phi_i - phi_h) / (2 * np.pi))  # the short way
        mirror = _unit(self.xyz[image] - self.xyz[h])
        if abs((circle.at((phi_i + phi_h) / 2) - self.xyz[key]) @ mirror) > _TOL:
            msg = "The umbrella image isn't a mirror image of its original."
            raise ValueError(msg)
        mid = (phi_i + phi_h) / 2
        margin = style.umbrella_margin * self.scale

        def clearance(far: float, phi: float, atom: int) -> float:
            tip, _, barb1, barb2 = self._head_points(circle, far, phi)
            center = self.page(self.xyz[atom])[None]
            sides = ((tip, barb1), (tip, barb2), (barb1, barb2))
            return (
                min(_segment_distance(center, p, q)[0] for p, q in sides)
                - self.atom_radius[atom]
                - margin
            )

        def shorten(far: float, near: float, atom: int) -> float:
            if clearance(far, near, atom) >= 0:
                return near
            return float(
                brentq(lambda phi: clearance(far, phi, atom), near, mid, xtol=1e-12)
            )

        start, end = shorten(phi_h, phi_i, image), shorten(phi_i, phi_h, h)
        return self._arc(circle, (start, end), f"umbrella-{key}", both_ends=True)

    def _umbrella_pairs(self, key: int, images: list[int]) -> list[tuple[int, int]]:
        """Pair each image with the original it mirrors.

        All pairs share one mirror plane through `key`, normal to (image -
        original). Try each original for the first image, and keep the plane that
        maps every image onto an original.
        """
        originals = [k for k in sorted(self.neighbors[key]) if k not in images]
        center = self.xyz[key]
        tol = 1e-6
        for first in originals:
            normal = _unit(self.xyz[images[0]] - self.xyz[first])
            pairs = []
            for image in images:
                mirrored = center + _reflect(self.xyz[image] - center, normal)
                dists = [np.linalg.norm(self.xyz[k] - mirrored) for k in originals]
                if min(dists) < tol:
                    pairs.append((originals[int(np.argmin(dists))], image))
            if len(pairs) == len(images):
                return pairs
        msg = f"The atoms {images} aren't mirror images of atoms bonded to {key}."
        raise ValueError(msg)

    def external_rotation(
        self, motion: ExternalRotation, drawn: list[np.ndarray]
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        """An axis line beyond j, with a ring around it.

        The line starts clear of everything drawn in its way. The ring is
        centered on it, and the line passes between the ring's back and front
        halves.
        """
        i, j, style = motion.i, motion.j, self.style
        u = _unit(self.xyz[j] - self.xyz[i])
        step = self.a_mat @ u[:2]  # page step per Å along the axis
        along = _unit(step)
        across = np.array([-along[1], along[0]])
        base = self.page(self.xyz[j])
        rx = style.ring_radius * self.scale
        ry = rx * abs(u[2])
        corridor = rx + style.stroke * self.scale

        # How far along the axis (from j, on the page) things in its way reach
        reach = 0.0
        for k in range(len(self.xyz)):
            rel = self.page(self.xyz[k]) - base
            if abs(rel @ across) - self.atom_radius[k] < corridor:
                reach = max(reach, rel @ along + self.atom_radius[k])
        for points in drawn:  # (x, y, half the stroke width there)
            rel = points[:, :2] - base
            near = np.abs(rel @ across) < corridor
            if near.any():
                reach = max(reach, float((rel[near] @ along + points[near, 2]).max()))
        near_end = reach + (style.axis_clearance + style.axis_stroke / 2) * self.scale
        center = near_end + ry + style.axis_overhang * self.scale
        far_end = center + ry + style.axis_overhang * self.scale
        on_axis = lambda s: self.xyz[j] + s / np.linalg.norm(step) * u  # noqa: E731
        circle = _Circle.about(on_axis(center), u, style.ring_radius)

        start, end = (
            style.rotation_angles
            if style.rotation_sense > 0
            else style.rotation_angles[::-1]
        )
        # A long arc: fit the head within the last quarter turn
        head_start = start + np.sign(end - start) * max(
            0.0, abs(end - start) - np.pi / 2
        )
        head, stop, head_points = self._head(
            circle, head_start, end, f"rotation-{i}-{j}-head"
        )
        cuts = [
            np.pi * n
            for n in range(-4, 5)
            if min(start, stop) < np.pi * n < max(start, stop)
        ]
        bounds = sorted([start, stop, *cuts], reverse=start > stop)
        pieces = list(pairwise(bounds))
        back = [p for p in pieces if np.sin(sum(p) / 2) * circle.e2[2] < 0]
        front = [p for p in pieces if p not in back]
        p0, p1 = self.page(on_axis(near_end)), self.page(on_axis(far_end))
        line = (
            f'<path style="fill:none;stroke:{style.color};'
            f"stroke-width:{style.axis_stroke * self.scale:.6g};"
            "stroke-linecap:round;stroke-linejoin:bevel;stroke-dasharray:none;"
            f'stroke-opacity:1" id="rotation-{i}-{j}-axis" '
            f'd="M {p0[0]:.6f},{p0[1]:.6f} L {p1[0]:.6f},{p1[1]:.6f}" />'
        )

        def arcs(ps: list, tag: str) -> str:
            return "".join(
                self._arc_path(circle, a, b, f"rotation-{i}-{j}-{tag}{n}", style.stroke)
                for n, (a, b) in enumerate(ps)
            )

        symbol = arcs(back, "back") + line + arcs(front, "front") + head
        samples = self.page(circle.at(np.linspace(start, stop, 101)))
        # Behind the molecule, unless it covers something
        slot = self._slot(
            np.vstack([samples, p0[None], p1[None]]), float(on_axis(center)[2])
        )
        offset = min(o.start for o in self.occluders) if slot is None else slot
        line_points = self._extent(
            np.array([p0, p1]), np.empty((0, 2)), style.axis_stroke
        )
        return [(offset, symbol)], np.vstack(
            [self._extent(samples, head_points, style.stroke), line_points]
        )


def _reflect(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect a vector through the plane normal to `normal`."""
    normal = _unit(normal)
    return vec - 2 * (vec @ normal) * normal


__all__ = [
    "ARROW_STYLE",
    "ArrowStyle",
    "ExternalRotation",
    "Head",
    "Torsion",
    "Umbrella",
    "add_arrows",
    "torsions",
]
