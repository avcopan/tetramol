"""Orient and render geometries as illustrations."""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import py3Dmol
import xyzrender
from automol import Geometry, geom
from scipy.spatial.transform import Rotation

STYLE = xyzrender.build_config(
    "flat",
    orient=False,
    atom_scale=3,
    bond_width=25,
    bond_color="#000000",
    atom_stroke_width=3,
    hy=True,
    bo=False,
    fog=False,
)
"""House style: flat atoms with black bonds, drawn as oriented (no auto-rotation)."""


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


def orient(geo: Geometry, *, viewer: str = "py3dmol", port: int = 0) -> Rotation:
    """Rotate a geometry by hand and get the rotation.

    With the default viewer, this opens py3Dmol in a web browser. Rotate the
    molecule with the mouse, then click "Done". The view matches `render`:
    both look down the z axis with x to the right and y up.

    The rotation is also printed as code, to paste into a drawing script so that
    it reproduces the orientation without the viewer.

    Args:
        geo: Geometry.
        viewer: ``"py3dmol"`` (in a browser) or ``"vmol"`` (via xyzrender,
            which requires the `vmol` package; press ``z`` to confirm, then
            ``q`` to quit).
        port: Port for the local server behind the browser viewer. The default
            (0) picks a free port.

    Returns:
        The rotation, to apply with ``automol.geom.rotate(geo, rot)``.

    Raises:
        ValueError: If `viewer` is unknown.
    """
    if viewer == "py3dmol":
        rot = _orient_py3dmol(geo, port=port)
    elif viewer == "vmol":
        rot = _orient_vmol(geo)
    else:
        msg = f"Unknown viewer {viewer!r}; use 'py3dmol' or 'vmol'."
        raise ValueError(msg)

    quat = ", ".join(f"{x:.6f}" for x in rot.as_quat())
    print(f"rot = Rotation.from_quat([{quat}])")  # noqa: T201
    return rot


# Browser viewer: py3Dmol can't send data back to Python, so serve the page from
# a local server, and have it post the view quaternion from 3Dmol.js
# (`viewer.getView()` -> [x, y, z, zoom, qx, qy, qz, qw]) back when done
_PAGE = """<!DOCTYPE html>
<html>
<head><title>tetramol: orient</title></head>
<body style="margin: 0; font-family: sans-serif;">
<div style="padding: 8px; display: flex; gap: 16px; align-items: center;">
  <button id="done" style="font-size: 1.2em;">Done</button>
  <span>Rotate with the mouse, then click Done. Quaternion:
  <code id="quat">0, 0, 0, 1</code></span>
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
  document.getElementById("done").onclick = () => {{
    fetch("/done", {{method: "POST", body: JSON.stringify(quat())}}).then(() => {{
      document.body.innerHTML = "<p style='padding: 8px'>Done. Close this tab.</p>";
    }});
  }};
}});
</script>
</body>
</html>
"""


def orient_page(geo: Geometry) -> str:
    """Make the web page for orienting a geometry in the browser.

    Args:
        geo: Geometry.

    Returns:
        HTML for a page with a py3Dmol viewer and a "Done" button, which posts
        the view quaternion ``[qx, qy, qz, qw]`` as JSON to ``/done``.
    """
    view = py3Dmol.view(width="100%", height="90vh")
    geom.view(geo, view=view)
    viewer_html = view.write_html()
    return _PAGE.format(viewer_html=viewer_html, uniqueid=view.uniqueid)


def serve_orient_page(
    geo: Geometry, *, port: int = 0
) -> tuple[ThreadingHTTPServer, list]:
    """Serve the orientation page from a local server, in a background thread.

    Args:
        geo: Geometry.
        port: Port to serve on. The default (0) picks a free port.

    Returns:
        The server, and a list that receives the view quaternion when the page
        posts it. The server shuts itself down after that.
    """
    page = orient_page(geo).encode()
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


def _orient_py3dmol(geo: Geometry, *, port: int) -> Rotation:
    """Orient a geometry in py3Dmol, in a web browser."""
    server, result = serve_orient_page(geo, port=port)
    url = f"http://localhost:{server.server_address[1]}"
    print(f"Orient the molecule at {url}, then click Done.")  # noqa: T201
    webbrowser.open(url)
    try:
        while not result:
            threading.Event().wait(0.2)
    finally:
        server.shutdown()
        server.server_close()
    return Rotation.from_quat(result[0])


def _orient_vmol(geo: Geometry) -> Rotation:
    """Orient a geometry in vmol, via xyzrender."""
    mol = geom._xyzrender_molecule(geo)  # noqa: SLF001
    xyzrender.orient(mol)
    coords = np.array([mol.graph.nodes[i]["position"] for i in mol.graph.nodes()])
    oriented = Geometry(
        symbols=geo.symbols, coordinates=coords, charge=geo.charge, spin=geo.spin
    )
    return geom.alignment_rotation(oriented, geo)
