"""Orient and render geometries as illustrations."""

import json
import re
import threading
import webbrowser
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import py3Dmol
import xyzrender
from automol import Geometry, geom
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

STYLE = xyzrender.build_config(
    "flat",
    orient=False,
    atom_scale=3,
    bond_width=25,
    bond_color="#000000",
    atom_stroke_width=3,
    bond_outline_color="#ffffff",
    bond_outline_width=2,
    hy=True,
    bo=False,
    fog=False,
)
"""House style: flat atoms with black bonds, drawn as oriented (no auto-rotation).

Bonds have a thin white outline, so that where bonds cross, the front one cuts a
gap through the one behind it. Fog is off, so that fading an atom (e.g. an
umbrella image) can only mean one thing. To change a field for one drawing,
use e.g. ``dataclasses.replace(STYLE, bond_outline_width=3)``.
"""

SPINS_COLUMNS = 4
"""Number of panels per row in `render_spins`."""

SPINS_PANEL_HEIGHT = 300.0
"""Height of each panel in `render_spins`."""


def render(
    geo: Geometry,
    out: str | Path | None = None,
    *,
    config: xyzrender.RenderConfig = STYLE,
) -> xyzrender.SVGResult:
    """Render a geometry as an SVG, looking down the z axis (x right, y up).

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.svg``.
        config: xyzrender configuration.

    Returns:
        The SVG, which displays inline in Jupyter-like REPLs.
    """
    return geom.render_svg(geo, out=out, config=config)


def render_png(
    svg: str | Path | xyzrender.SVGResult,
    out: str | Path | None = None,
    *,
    crop: Sequence[float] | None = None,
    width: int = 1600,
) -> bytes:
    """Rasterize an SVG to a PNG, for a quick look, e.g. at arrows from `add_arrows`.

    Args:
        svg: The SVG, its text, or its path.
        out: Output path. The suffix is replaced with ``.png``.
        crop: Part of the drawing to show, as ``(x, y, width, height)`` in the SVG's
            own units. Defaults to the whole drawing.
        width: Width of the PNG, in pixels.

    Returns:
        The PNG.

    Note:
        This uses cairosvg, which doesn't draw ``context-stroke`` marker fills
        (Inkscape's default arrowheads); they come out black.
    """
    import cairosvg  # noqa: PLC0415  (slow to import; only needed here)

    text = Path(svg).read_text() if isinstance(svg, Path) else str(svg)
    root = re.search(r"<svg\b[^>]*>", text)
    if root is None:
        msg = "Expected an <svg> element."
        raise ValueError(msg)
    tag = root.group(0)
    if crop is None:
        view_box = re.search(r'viewBox="([^"]*)"', tag)
        crop = [float(x) for x in view_box.group(1).split()] if view_box else None
    if crop is not None:
        x, y, w, h = crop
        tag = re.sub(r'\s(viewBox|width|height)="[^"]*"', "", tag)
        height = round(width * h / w)
        size = f' viewBox="{x} {y} {w} {h}" width="{width}" height="{height}"'
        tag = tag[:4] + size + tag[4:]
    png = cairosvg.svg2png(
        bytestring=(text[: root.start()] + tag + text[root.end() :]).encode(),
        background_color="white",
    )
    if out is not None:
        Path(out).with_suffix(".png").write_bytes(png)
    return png


def render_spins(
    geo: Geometry,
    out: str | Path | None = None,
    *,
    axis: ArrayLike = (0.0, 1.0, 0.0),
    angles: Sequence[float] = range(0, 120, 15),
    config: xyzrender.RenderConfig = STYLE,
) -> xyzrender.SVGResult:
    """Render a geometry spun about an axis by several angles, side by side.

    Use it to pick a spin in the final style, e.g. after putting a molecular axis
    along y with `automol.geom.axis_alignment_rotation`. Each panel is labeled
    with its angle, for ``Rotation.from_euler("y", angle, degrees=True)`` (or
    the equivalent rotation about `axis`). Each panel is scaled to fit on its
    own, so compare shapes, not sizes.

    Args:
        geo: Geometry.
        out: Output path. The suffix is replaced with ``.svg``.
        axis: Axis to spin about. The default is y, up the page.
        angles: Angles, in degrees.
        config: xyzrender configuration.

    Returns:
        The tiled SVG, which displays inline in Jupyter-like REPLs.
    """
    unit = _unit(axis)
    height = SPINS_PANEL_HEIGHT
    label_height = 0.12 * height
    panels = []
    x = y = width = 0.0
    for i, angle in enumerate(angles):
        if i and not i % SPINS_COLUMNS:
            x, y = 0.0, y + height + label_height
        rot = Rotation.from_rotvec(np.radians(angle) * unit)
        svg = str(render(geom.rotate(geo, rot), config=config))
        view_box = re.search(r'viewBox="([^"]*)"', svg)
        if view_box is None:
            msg = "Expected the rendered SVG to have a viewBox."
            raise ValueError(msg)
        *_, vb_width, vb_height = map(float, view_box.group(1).split())
        panel_width = height * vb_width / vb_height
        svg = re.sub(
            r"<svg\b[^>]*>",
            f'<svg x="{x:.1f}" y="{y + label_height:.1f}" width="{panel_width:.1f}" '
            f'height="{height:.1f}" viewBox="{view_box.group(1)}">',
            svg,
            count=1,
        )
        label = (
            f'<text x="{x + panel_width / 2:.1f}" y="{y + 0.8 * label_height:.1f}" '
            f'font-family="sans-serif" font-size="{0.7 * label_height:.1f}" '
            f'text-anchor="middle">{angle:g}°</text>'
        )
        panels.extend([label, svg])
        x += panel_width
        width = max(width, x)
    total_height = y + height + label_height
    sheet = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {width:.1f} {total_height:.1f}" '
        f'width="{width:.1f}" height="{total_height:.1f}">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(panels)
        + "\n</svg>\n"
    )
    result = xyzrender.SVGResult(sheet)
    if out is not None:
        result.save(Path(out).with_suffix(".svg"))
    return result


def orient(geo: Geometry, *, axis: ArrayLike | None = None, port: int = 0) -> Rotation:
    """Rotate a geometry by hand in the browser and get the rotation.

    This opens the geometry in py3Dmol in a web browser. Rotate the molecule
    with the mouse, then click "Done". The view matches `render`: both look
    down the z axis with x to the right and y up.

    With `axis`, the page only spins the molecule about that axis, with a slider,
    e.g. after putting a molecular axis along y with
    `automol.geom.axis_alignment_rotation`.

    The rotation is also printed as code, to paste into a drawing script so that
    it reproduces the orientation without the viewer.

    Args:
        geo: Geometry.
        axis: Axis to spin about, if any. By default, rotate freely.
        port: Port for the local server behind the page. The default (0) picks a
            free port.

    Returns:
        The rotation, to apply with ``automol.geom.rotate(geo, rot)``.
    """
    _, rot = _orient([geo], labels=None, axis=axis, port=port)
    return rot


def orient_multi(
    geos: Sequence[Geometry],
    *,
    labels: Sequence[str] | None = None,
    axis: ArrayLike | None = None,
    port: int = 0,
) -> tuple[int, Rotation]:
    """Choose one of several geometries and rotate it by hand in the browser.

    Like `orient`, with a slider (or the arrow keys) to switch between the
    geometries, e.g. CREST conformers. The view is shared, so switching keeps
    the current rotation. Click "Done" on the one you want.

    The choice and the rotation are also printed as code, to paste into a
    drawing script (``geo = geos[i]`` and ``rot = ...``).

    Args:
        geos: Geometries.
        labels: Text to show for each geometry, e.g. its relative energy.
            Defaults to none (just the index).
        axis: Axis to spin about, if any (see `orient`). By default, rotate
            freely.
        port: Port for the local server behind the page. The default (0) picks a
            free port.

    Returns:
        The index of the chosen geometry, and the rotation, to apply with
        ``automol.geom.rotate(geos[index], rot)``.
    """
    return _orient(geos, labels=labels, axis=axis, port=port)


def _orient(
    geos: Sequence[Geometry],
    *,
    labels: Sequence[str] | None,
    axis: ArrayLike | None,
    port: int,
) -> tuple[int, Rotation]:
    """Serve the orientation page, wait for "Done", and print the result as code."""
    server, result = serve_orient_page(geos, labels=labels, axis=axis, port=port)
    url = f"http://localhost:{server.server_address[1]}"
    print(f"Orient the molecule at {url}, then click Done.")  # noqa: T201
    webbrowser.open(url)
    try:
        while not result:
            threading.Event().wait(0.2)
    finally:
        server.shutdown()
        server.server_close()

    index = int(result[0]["index"])
    rot = Rotation.from_quat(result[0]["quat"])
    if len(geos) > 1:
        print(f"geo = geos[{index}]")  # noqa: T201
    if axis is None:
        quat = ", ".join(f"{x:.6f}" for x in rot.as_quat())
        print(f"rot = Rotation.from_quat([{quat}])")  # noqa: T201
        return index, rot

    unit = _unit(axis)
    angle = float(np.degrees(np.dot(rot.as_rotvec(), unit))) % 360
    print(f"rot = {_spin_code(unit, angle)}")  # noqa: T201
    return index, Rotation.from_rotvec(np.radians(angle) * unit)


def _unit(axis: ArrayLike) -> np.ndarray:
    """Normalize an axis."""
    vec = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        msg = f"The axis must be non-zero, got {vec}."
        raise ValueError(msg)
    return vec / norm


def _spin_code(unit: np.ndarray, angle: float) -> str:
    """Format a spin about an axis as code."""
    for name, basis in zip("xyz", np.eye(3), strict=True):
        if np.allclose(unit, basis):
            return f'Rotation.from_euler("{name}", {angle:.1f}, degrees=True)'
    vec = ", ".join(f"{x:.6f}" for x in unit)
    return f"Rotation.from_rotvec(np.radians({angle:.1f}) * np.array([{vec}]))"


# Browser viewer: py3Dmol can't send data back to Python, so serve the page from
# a local server, and have it post the view quaternion from 3Dmol.js
# (`viewer.getView()` -> [x, y, z, zoom, qx, qy, qz, qw]) back when done
_PAGE = """<!DOCTYPE html>
<html>
<head><title>tetramol: orient</title></head>
<body style="margin: 0; font-family: sans-serif;">
<div style="padding: 8px; display: flex; gap: 16px; align-items: center;">
  <button id="done" style="font-size: 1.2em;">Done</button>
  {controls}
  <span>Quaternion: <code id="quat">0, 0, 0, 1</code></span>
</div>
{viewer_html}
<script>
$3Dmolpromise.then(function() {{
  const viewer = viewer_{uniqueid};
  const quat = () => viewer.getView().slice(4);
  const show = () => {{
    document.getElementById("quat").textContent =
      quat().map((x) => x.toFixed(4)).join(", ");
  }};
  viewer.setViewChangeCallback(show);
  let index = () => 0;
  {pick_script}
  {spin_script}
  document.getElementById("done").onclick = () => {{
    const body = JSON.stringify({{index: index(), quat: quat()}});
    fetch("/done", {{method: "POST", body: body}}).then(() => {{
      document.body.innerHTML = "<p style='padding: 8px'>Done. Close this tab.</p>";
    }});
  }};
}});
</script>
</body>
</html>
"""

_FREE_CONTROLS = "<span>Rotate with the mouse, then click Done.</span>"

_PICK_CONTROLS = """<label>Geometry:
  <input id="pick" type="range" min="0" max="{last}" step="1" value="0"
         style="width: 240px;">
  <code id="picked"></code></label>"""

# Show only the picked model. The view is shared by all models, so switching
# keeps the rotation. The arrow keys step through the models.
_PICK_SCRIPT = """const labels = {labels_js};
  const picker = document.getElementById("pick");
  index = () => Number(picker.value);
  const pick = () => {{
    labels.forEach((_, i) => {{
      const model = viewer.getModel(i);
      if (i === index()) model.show(); else model.hide();
    }});
    viewer.render();
    const label = labels[index()] ? ": " + labels[index()] : "";
    document.getElementById("picked").textContent =
      `${{index()}} / ${{labels.length - 1}}${{label}}`;
  }};
  picker.oninput = pick;
  document.addEventListener("keydown", (event) => {{
    if (document.activeElement === picker) return;
    const step = {{ArrowRight: 1, ArrowLeft: -1}}[event.key];
    if (!step) return;
    picker.value = Math.min(Math.max(index() + step, 0), labels.length - 1);
    pick();
  }});
  pick();"""

_SPIN_CONTROLS = """<label>Spin about ({axis_text}):
  <input id="angle" type="range" min="0" max="360" step="1" value="0"
         style="width: 360px;">
  <code id="degrees">0</code>&deg;</label>"""

# Set the view to the slider's spin, keeping the zoom and position. Mouse
# rotations are undone on the next view change, so the view is always a spin.
_SPIN_SCRIPT = """const axis = [{axis_js}];
  const slider = document.getElementById("angle");
  let setting = false;
  const spin = () => {{
    const half = (Math.PI / 360) * slider.value;
    const view = viewer.getView();
    view.splice(4, 4, ...axis.map((a) => a * Math.sin(half)), Math.cos(half));
    setting = true;
    viewer.setView(view);
    viewer.render();
    setting = false;
    document.getElementById("degrees").textContent = slider.value;
    show();
  }};
  viewer.setViewChangeCallback(() => {{ if (!setting) spin(); }});
  slider.oninput = spin;
  spin();"""


def orient_page(
    geos: Geometry | Sequence[Geometry],
    *,
    labels: Sequence[str] | None = None,
    axis: ArrayLike | None = None,
) -> str:
    """Make the web page for orienting a geometry in the browser.

    Args:
        geos: Geometry, or several to choose from with a slider.
        labels: Text to show for each geometry (see `orient_multi`).
        axis: Axis to spin about with a slider, if any. By default, the page
            rotates freely with the mouse.

    Returns:
        HTML for a page with a py3Dmol viewer and a "Done" button, which posts
        ``{"index": i, "quat": [qx, qy, qz, qw]}`` as JSON to ``/done``: the
        chosen geometry and the view quaternion.

    Raises:
        ValueError: If there are no geometries, or `labels` doesn't have one
            entry per geometry.
    """
    geos = [geos] if isinstance(geos, Geometry) else list(geos)
    if not geos:
        msg = "Expected at least one geometry."
        raise ValueError(msg)
    if labels is not None and len(labels) != len(geos):
        msg = f"Got {len(labels)} labels for {len(geos)} geometries."
        raise ValueError(msg)

    view = py3Dmol.view(width="100%", height="90vh")
    for geo in geos:
        geom.view(geo, view=view)
    controls = []
    pick_script = spin_script = ""
    if len(geos) > 1:
        labels_js = json.dumps(list(labels or [""] * len(geos))).replace("</", "<\\/")
        controls.append(_PICK_CONTROLS.format(last=len(geos) - 1))
        pick_script = _PICK_SCRIPT.format(labels_js=labels_js)
    if axis is None:
        controls.append(_FREE_CONTROLS)
    else:
        unit = _unit(axis)
        text = ", ".join(f"{x:g}" for x in unit.round(3))
        controls.append(_SPIN_CONTROLS.format(axis_text=text))
        spin_script = _SPIN_SCRIPT.format(axis_js=", ".join(map(repr, unit.tolist())))
    return _PAGE.format(
        viewer_html=view.write_html(),
        uniqueid=view.uniqueid,
        controls="\n  ".join(controls),
        pick_script=pick_script,
        spin_script=spin_script,
    )


def serve_orient_page(
    geos: Geometry | Sequence[Geometry],
    *,
    labels: Sequence[str] | None = None,
    axis: ArrayLike | None = None,
    port: int = 0,
) -> tuple[ThreadingHTTPServer, list]:
    """Serve the orientation page from a local server, in a background thread.

    Args:
        geos: Geometry, or several to choose from (see `orient_page`).
        labels: Text to show for each geometry (see `orient_multi`).
        axis: Axis to spin about, if any (see `orient_page`).
        port: Port to serve on. The default (0) picks a free port.

    Returns:
        The server, and a list that receives the page's result,
        ``{"index": i, "quat": [qx, qy, qz, qw]}``, when it posts it. The server
        shuts itself down after that.
    """
    page = orient_page(geos, labels=labels, axis=axis).encode()
    result: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            result.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()
            threading.Thread(target=self.server.shutdown).start()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("localhost", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, result
