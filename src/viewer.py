"""Write a self-contained HTML page that shows a point cloud in 3D.

Usage:
    python -m src.viewer outputs/c00a170fe1/pointcloud.ply

Writes viewer.html next to the PLY. Open it in any browser: drag to rotate,
scroll to zoom, right-drag to pan. A slider hides everything above a chosen
height so the ceiling can be cut away to look into the rooms.
The points are embedded in the page, so it works by double-clicking the file
(three.js itself is loaded from a CDN, so an internet connection is needed).
"""

import argparse
import base64
from pathlib import Path

import numpy as np

from src.pointcloud import read_ply

MAX_POINTS = 1_500_000   # keeps the page under ~30 MB and smooth on a laptop

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #1d1f21; font-family: system-ui, sans-serif; }}
  #panel {{ position: absolute; top: 12px; left: 12px; background: #fff; padding: 10px 14px;
           border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.2); font-size: 14px; }}
  #panel input {{ width: 220px; }}
</style></head>
<body>
<div id="panel">
  <b>{title}</b> &middot; {count} points<br>
  Drag: rotate &middot; Scroll: zoom &middot; Right-drag: pan<br>
  Cut above height: <input id="cut" type="range" min="{ymin}" max="{ymax}" step="0.01" value="{ymax}">
  <span id="cutval"></span>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const bytes = Uint8Array.from(atob("{data}"), c => c.charCodeAt(0));
const n = {count};
const positions = new Float32Array(bytes.buffer, 0, n * 3);
const colors = new Uint8Array(bytes.buffer, n * 12, n * 3);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
geometry.computeBoundingSphere();

const cut = new THREE.Plane(new THREE.Vector3(0, -1, 0), {ymax});
const material = new THREE.PointsMaterial({{ size: 0.018, vertexColors: true, clippingPlanes: [cut] }});
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d1f21);
scene.add(new THREE.Points(geometry, material));

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 200);
const centre = geometry.boundingSphere.center, radius = geometry.boundingSphere.radius;
camera.position.set(centre.x, centre.y + radius * 1.6, centre.z + radius * 0.6);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.localClippingEnabled = true;
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.copy(centre);

const slider = document.getElementById("cut"), label = document.getElementById("cutval");
function updateCut() {{ cut.constant = parseFloat(slider.value); label.textContent = "y = " + slider.value + " m"; }}
slider.oninput = updateCut; updateCut();
addEventListener("resize", () => {{
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
(function loop() {{ controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); }})();
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", help="path to a pointcloud.ply written by src.pointcloud")
    args = parser.parse_args()

    points, colors = read_ply(args.ply)
    if len(points) > MAX_POINTS:
        keep = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
        points, colors = points[keep], colors[keep]

    payload = np.ascontiguousarray(points, dtype="<f4").tobytes() + np.ascontiguousarray(colors).tobytes()
    out = Path(args.ply).with_name("viewer.html")
    out.write_text(PAGE.format(
        title=Path(args.ply).parent.name,
        count=len(points),
        ymin=f"{points[:, 1].min():.2f}",
        ymax=f"{points[:, 1].max():.2f}",
        data=base64.b64encode(payload).decode("ascii"),
    ), encoding="utf-8")
    print(f"wrote {out} ({len(points):,} points)")


if __name__ == "__main__":
    main()
