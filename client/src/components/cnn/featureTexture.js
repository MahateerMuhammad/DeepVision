import * as THREE from "three";

// Perceptual sequential ramp: pale → sky → cerulean → indigo. Feature maps are
// ≥0 after ReLU (and inputs are 0–1), so a sequential ramp reads cleanly, and
// the wider hue travel makes activation strength far more legible than a
// near-monochrome fade.
const lerp = (a, b, t) => a + (b - a) * t;
const STOPS = [
  [0.0, [244, 249, 255]], // near-white
  [0.28, [125, 211, 252]], // sky-300
  [0.6, [14, 165, 233]], // cerulean-500
  [1.0, [30, 58, 138]], // indigo-900
];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (t <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const u = (t - t0) / (t1 - t0 || 1);
      return [lerp(c0[0], c1[0], u), lerp(c0[1], c1[1], u), lerp(c0[2], c1[2], u)];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

/** CSS gradient stops for a legend bar, matching the texture ramp. */
export const RAMP_CSS = STOPS.map(
  ([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${Math.round(t * 100)}%`
).join(", ");

export function gridExtent(grid) {
  let min = Infinity;
  let max = -Infinity;
  for (const row of grid)
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  return { min, max };
}

/** Build a crisp (nearest-filtered) DataTexture from an H×W grid. */
export function makeHeatTexture(grid) {
  const h = grid.length;
  const w = grid[0].length;
  const { min, max } = gridExtent(grid);
  const span = max - min || 1;
  const data = new Uint8Array(w * h * 4);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const t = (grid[r][c] - min) / span;
      const [rr, gg, bb] = ramp(t);
      const i = (r * w + c) * 4;
      data[i] = rr;
      data[i + 1] = gg;
      data[i + 2] = bb;
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.flipY = true;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

export function cssColor(v, min, max) {
  const [r, g, b] = ramp((v - min) / (max - min || 1));
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}
