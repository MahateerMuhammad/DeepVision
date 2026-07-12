// Client-side activation math for the Activation Lab (pure, no backend).

export const ACTIVATIONS = {
  relu: {
    label: "ReLU",
    f: (x) => Math.max(0, x),
    df: (x) => (x > 0 ? 1 : 0),
    domain: [-4, 4],
    range: [-0.5, 4],
    dRange: [-0.25, 1.25],
    note: "Piecewise linear — the kink at x = 0 makes f′ discontinuous. Gradient is exactly 0 for all negative inputs: dead units never recover.",
  },
  leaky_relu: {
    label: "Leaky ReLU",
    f: (x) => (x > 0 ? x : 0.01 * x),
    df: (x) => (x > 0 ? 1 : 0.01),
    domain: [-4, 4],
    range: [-0.5, 4],
    dRange: [-0.25, 1.25],
    note: "The 0.01 slope for x < 0 keeps a trickle of gradient alive, so no unit can die completely.",
  },
  sigmoid: {
    label: "Sigmoid",
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    domain: [-6, 6],
    range: [-0.15, 1.15],
    dRange: [-0.05, 0.3],
    note: "f′ peaks at only 0.25 and collapses toward 0 in the tails — stack a few of these and gradients vanish geometrically.",
  },
  tanh: {
    label: "Tanh",
    f: (x) => Math.tanh(x),
    df: (x) => 1 - Math.tanh(x) ** 2,
    domain: [-4, 4],
    range: [-1.25, 1.25],
    dRange: [-0.15, 1.15],
    note: "Zero-centered with f′(0) = 1 — healthier than sigmoid, but the tails still saturate to zero gradient.",
  },
  linear: {
    label: "Linear",
    f: (x) => x,
    df: () => 1,
    domain: [-4, 4],
    range: [-4.5, 4.5],
    dRange: [-0.25, 1.5],
    note: "Constant gradient of 1, but a stack of linear layers collapses to a single linear map — no depth is gained.",
  },
};

export const ACTIVATION_KEYS = ["relu", "leaky_relu", "sigmoid", "tanh", "linear"];

/** Sample fn over [x0, x1] into n points → [{x, y}] */
export function sampleCurve(fn, x0, x1, n = 240) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const x = x0 + ((x1 - x0) * i) / n;
    pts.push({ x, y: fn(x) });
  }
  return pts;
}

/**
 * Resample freehand points to be monotone in x on a fixed grid,
 * linearly interpolating between captured samples.
 */
export function monotoneResample(rawPts, x0, x1, n = 160) {
  if (rawPts.length < 2) return null;
  const sorted = [...rawPts].sort((a, b) => a.x - b.x);
  const out = [];
  for (let i = 0; i <= n; i++) {
    const x = x0 + ((x1 - x0) * i) / n;
    if (x <= sorted[0].x) {
      out.push({ x, y: sorted[0].y });
      continue;
    }
    if (x >= sorted[sorted.length - 1].x) {
      out.push({ x, y: sorted[sorted.length - 1].y });
      continue;
    }
    let j = 1;
    while (j < sorted.length && sorted[j].x < x) j++;
    const a = sorted[j - 1];
    const b = sorted[j];
    const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
    out.push({ x, y: a.y + t * (b.y - a.y) });
  }
  return out;
}

/** Central-difference numerical derivative of sampled points. */
export function numericalDerivative(pts) {
  return pts.map((p, i) => {
    const prev = pts[Math.max(0, i - 1)];
    const next = pts[Math.min(pts.length - 1, i + 1)];
    const dx = next.x - prev.x;
    return { x: p.x, y: dx === 0 ? 0 : (next.y - prev.y) / dx };
  });
}
