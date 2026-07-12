// Layered left→right layout for the network graph.
// Column 0 is the input vector, columns 1..L are dense layers.

export const NODE_R = 16;
export const LAYER_GAP = 190;
export const NODE_GAP = 64;

/**
 * @param {number} inputSize
 * @param {Array<{out_features:number, activation:string}>} layers
 * @returns {{nodes, edges, columns, width, height}}
 */
export function layoutNetwork(inputSize, layers) {
  const sizes = [inputSize, ...layers.map((l) => l.out_features)];
  const maxUnits = Math.max(...sizes);
  const height = maxUnits * NODE_GAP + 120;
  const width = (sizes.length - 1) * LAYER_GAP + 160;

  const columns = sizes.map((count, col) => {
    const x = 80 + col * LAYER_GAP;
    const totalH = (count - 1) * NODE_GAP;
    const y0 = height / 2 - totalH / 2;
    return {
      col,
      x,
      count,
      isInput: col === 0,
      layerIndex: col - 1, // backend layer index; -1 for input column
      activation: col === 0 ? null : layers[col - 1].activation,
      ys: Array.from({ length: count }, (_, i) => y0 + i * NODE_GAP),
      top: y0 - NODE_R,
      bottom: y0 + totalH + NODE_R,
    };
  });

  const nodes = [];
  columns.forEach((c) => {
    c.ys.forEach((y, i) => {
      nodes.push({
        id: `${c.col}:${i}`,
        col: c.col,
        layerIndex: c.layerIndex,
        neuron: i,
        x: c.x,
        y,
        isInput: c.isInput,
      });
    });
  });

  const edges = [];
  for (let col = 1; col < columns.length; col++) {
    const src = columns[col - 1];
    const dst = columns[col];
    for (let r = 0; r < dst.count; r++) {
      for (let cIdx = 0; cIdx < src.count; cIdx++) {
        edges.push({
          id: `e${col - 1}:${r}:${cIdx}`,
          layerIndex: col - 1, // weight lives in layer col-1
          row: r, // output neuron index
          colIdx: cIdx, // input neuron index
          x1: src.x + NODE_R,
          y1: src.ys[cIdx],
          x2: dst.x - NODE_R,
          y2: dst.ys[r],
        });
      }
    }
  }

  return { nodes, edges, columns, width, height };
}

/** Cubic bezier path between two node anchors. */
export function edgePath(e) {
  const mx = (e.x1 + e.x2) / 2;
  return `M ${e.x1} ${e.y1} C ${mx} ${e.y1}, ${mx} ${e.y2}, ${e.x2} ${e.y2}`;
}

/**
 * Point on the edge bezier at parameter t (0..1) — used to spread
 * per-edge labels so they don't pile up at the shared midpoint.
 */
export function edgePoint(e, t) {
  const mx = (e.x1 + e.x2) / 2;
  const u = 1 - t;
  const x = u * u * u * e.x1 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * e.x2;
  const s = 3 * t * t - 2 * t * t * t; // y eases from y1 to y2
  const y = e.y1 + s * (e.y2 - e.y1);
  return { x, y };
}

/** Map |weight| → stroke width, clamped so the web stays readable. */
export function weightStroke(w, max) {
  const m = max > 0 ? Math.abs(w) / max : 0;
  return 0.75 + m * 2.75;
}
