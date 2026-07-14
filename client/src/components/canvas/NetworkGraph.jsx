import { useCallback, useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import { AnimatePresence, motion } from "framer-motion";
import Odometer from "../ui/Odometer";
import KatexBlock from "../ui/KatexBlock";
import { edgePath, edgePoint, weightStroke, NODE_R } from "../../lib/graphLayout";

const CERULEAN = "#0EA5E9";
const CRIMSON = "#EF4444";
const DEEP_BLUE = "#1D4ED8";
const LINE = "#E4E4E7";
const INK = "#18181B";

/** Set of edge ids on the loss→weight path for a clicked edge. */
function tracePathEdges(edge, layout) {
  const ids = new Set([edge.id]);
  const numLayers = layout.columns.length - 1;
  let reachable = new Set([edge.row]);
  for (let l = edge.layerIndex + 1; l < numLayers; l++) {
    const next = new Set();
    layout.edges.forEach((e) => {
      if (e.layerIndex === l && reachable.has(e.colIdx)) {
        ids.add(e.id);
        next.add(e.row);
      }
    });
    reachable = next;
  }
  return ids;
}

export default function NetworkGraph({
  layout,
  spec,
  forwardData,
  backwardData,
  mode, // "forward" | "backward"
  revealed, // number of layers revealed in current mode
  pulsing, // layer index currently receiving a pulse, or null
  pulseId, // increments per pulse to retrigger CSS animation
  selection, // {type:'node'|'edge', ...} | null
  traceEdge, // edge object when trace mode active
  deadXray,
  shockwaveKey,
  chip, // {layerIndex, neuron, linear, activation} | null
  onSelectNode,
  onSelectEdge,
  onClearSelection,
}) {
  const wrapRef = useRef(null);
  const svgRef = useRef(null);
  const gRef = useRef(null);
  const zoomRef = useRef(null);
  const transformRef = useRef(d3.zoomIdentity);
  const vignetteRef = useRef(null);
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const layoutRef = useRef(layout);
  layoutRef.current = layout;

  const numLayers = layout.columns.length - 1;

  const updateVignette = useCallback(() => {
    const el = vignetteRef.current;
    const sel = selectionRef.current;
    if (!el || !sel || sel.type !== "node") return;
    const node = layoutRef.current.nodes.find(
      (n) => n.col === sel.col && n.neuron === sel.neuron
    );
    if (!node) return;
    const t = transformRef.current;
    el.style.setProperty("--vx", `${t.applyX(node.x)}px`);
    el.style.setProperty("--vy", `${t.applyY(node.y)}px`);
  }, []);

  // ── d3-zoom (imperative; no React re-render per frame) ──
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const zoom = d3
      .zoom()
      .scaleExtent([0.2, 5])
      .on("zoom", (event) => {
        transformRef.current = event.transform;
        gRef.current.setAttribute("transform", event.transform.toString());
        const k = event.transform.k;
        const band = k < 0.6 ? "zoom-far" : k > 2.2 ? "zoom-near" : "zoom-mid";
        svgRef.current.setAttribute("class", `net-svg block h-full w-full ${band}`);
        updateVignette();
      });
    svg.call(zoom).on("dblclick.zoom", null);
    zoomRef.current = zoom;
    return () => svg.on(".zoom", null);
  }, [updateVignette]);

  // fit graph on layout change / container resize
  useEffect(() => {
    const fit = () => {
      const rect = svgRef.current.getBoundingClientRect();
      if (rect.width === 0) return;
      const k = Math.min(rect.width / layout.width, rect.height / layout.height, 1.4) * 0.92;
      const tx = (rect.width - layout.width * k) / 2;
      const ty = (rect.height - layout.height * k) / 2;
      d3.select(svgRef.current).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(tx, ty).scale(k)
      );
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [layout.width, layout.height]);

  useEffect(updateVignette, [selection, updateVignette]);

  // ── derived styling data ──
  const maxW = useMemo(() => {
    if (!forwardData) return 1;
    let m = 0;
    forwardData.layers.forEach((l) =>
      l.weights.forEach((row) => row.forEach((w) => (m = Math.max(m, Math.abs(w)))))
    );
    return m || 1;
  }, [forwardData]);

  const maxG = useMemo(() => {
    if (!backwardData) return 1;
    let m = 0;
    backwardData.layers.forEach((l) =>
      l.grad_weights.forEach((row) => row.forEach((g) => (m = Math.max(m, Math.abs(g)))))
    );
    return m || 1;
  }, [backwardData]);

  // layers whose gradient signal has effectively died (vanishing)
  const dryLayers = useMemo(() => {
    const s = new Set();
    if (backwardData) {
      backwardData.layers.forEach((l, i) => {
        const mx = Math.max(...l.grad_pre_activation.map(Math.abs));
        if (mx < 1e-4) s.add(i);
      });
    }
    return s;
  }, [backwardData]);

  const bRevealed = useCallback(
    (l) => mode === "backward" && revealed > 0 && l >= numLayers - revealed,
    [mode, revealed, numLayers]
  );
  const fRevealed = useCallback(
    (l) => (mode === "forward" ? l < revealed : forwardData != null),
    [mode, revealed, forwardData]
  );

  const traceIds = useMemo(
    () => (traceEdge ? tracePathEdges(traceEdge, layout) : null),
    [traceEdge, layout]
  );

  const nodeValue = useCallback(
    (n) => {
      if (!forwardData) return null;
      if (n.isInput) return forwardData.input[n.neuron];
      return forwardData.layers[n.layerIndex]?.post_activation[n.neuron] ?? null;
    },
    [forwardData]
  );

  const isDropped = useCallback(
    (n) => {
      if (n.isInput || !forwardData) return false;
      return forwardData.layers[n.layerIndex]?.dropped_mask?.[n.neuron] === 1;
    },
    [forwardData]
  );

  // A token that changes whenever a new forward pass arrives, so the drop
  // animation replays each time the dropout mask is resampled.
  const forwardTokenRef = useRef(0);
  const forwardToken = useMemo(() => {
    void forwardData; // recompute the token on each new forward pass
    return ++forwardTokenRef.current;
  }, [forwardData]);

  const edgeStyle = useCallback(
    (e) => {
      const fl = forwardData?.layers[e.layerIndex];
      if (!fl) return { stroke: "#D4D4D8", width: 1, opacity: 0.4, value: null };
      if (bRevealed(e.layerIndex)) {
        if (dryLayers.has(e.layerIndex)) {
          return { stroke: LINE, width: 1, opacity: 0.9, value: 0 };
        }
        const g = backwardData.layers[e.layerIndex].grad_weights[e.row][e.colIdx];
        return {
          stroke: g >= 0 ? CERULEAN : CRIMSON,
          width: weightStroke(g, maxG),
          opacity: 0.55,
          value: g,
        };
      }
      const w = fl.weights[e.row][e.colIdx];
      return {
        stroke: w >= 0 ? CERULEAN : CRIMSON,
        width: weightStroke(w, maxW),
        opacity: 0.3,
        value: w,
      };
    },
    [forwardData, backwardData, bRevealed, dryLayers, maxW, maxG]
  );

  const inputCol = layout.columns[0];

  return (
    <div ref={wrapRef} className="dot-grid relative h-full w-full overflow-hidden">
      <svg
        ref={svgRef}
        className="net-svg zoom-mid block h-full w-full"
        style={{ cursor: "grab" }}
        onClick={onClearSelection}
      >
        <g ref={gRef}>
          {/* ══ SLAB VIEW (zoomed out) ══ */}
          <g className="slab-view">
            {layout.columns.map((c) => (
              <g key={c.col}>
                <rect
                  x={c.x - 46}
                  y={c.top - 16}
                  width={92}
                  height={c.bottom - c.top + 32}
                  fill="#F4F4F5"
                  stroke="#D4D4D8"
                  strokeWidth="1"
                  rx="4"
                />
                <text
                  x={c.x}
                  y={(c.top + c.bottom) / 2}
                  textAnchor="middle"
                  fontSize="15"
                  fontWeight="700"
                  fill={INK}
                  fontFamily="Inter Variable, sans-serif"
                  style={{ letterSpacing: "0.08em" }}
                >
                  {c.isInput ? `IN ${c.count}` : `${c.count}`}
                </text>
                <text
                  x={c.x}
                  y={(c.top + c.bottom) / 2 + 20}
                  textAnchor="middle"
                  fontSize="9"
                  fontWeight="600"
                  fill="#71717A"
                  fontFamily="Inter Variable, sans-serif"
                  style={{ letterSpacing: "0.1em" }}
                >
                  {c.isInput ? "INPUT" : `DENSE · ${c.activation?.toUpperCase()}`}
                </text>
              </g>
            ))}
            {layout.columns.slice(1).map((c, i) => {
              const prev = layout.columns[i];
              return (
                <line
                  key={c.col}
                  x1={prev.x + 46}
                  x2={c.x - 46}
                  y1={(prev.top + prev.bottom) / 2}
                  y2={(c.top + c.bottom) / 2}
                  stroke="#D4D4D8"
                  strokeWidth="10"
                />
              );
            })}
          </g>

          {/* ══ DETAIL VIEW (node-and-edge web) ══ */}
          <g className="detail-view">
            {/* base edges (fade to 10% in trace mode) */}
            <g style={{ opacity: traceIds ? 0.1 : 1, transition: "opacity 150ms" }}>
              {layout.edges.map((e) => {
                const s = edgeStyle(e);
                const dstCount = layout.columns[e.layerIndex + 1].count;
                const lt = 0.22 + 0.56 * (dstCount > 1 ? e.row / (dstCount - 1) : 0.5);
                const lp = edgePoint(e, lt);
                return (
                  <g key={e.id}>
                    <path
                      d={edgePath(e)}
                      fill="none"
                      stroke={s.stroke}
                      strokeWidth={s.width}
                      strokeOpacity={s.opacity}
                      style={{ transition: "stroke 300ms, stroke-width 300ms, stroke-opacity 300ms" }}
                    />
                    {/* invisible hit area */}
                    <path
                      d={edgePath(e)}
                      data-edge={e.id}
                      fill="none"
                      stroke="transparent"
                      strokeWidth="9"
                      style={{ cursor: "pointer" }}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        onSelectEdge(e);
                      }}
                    />
                    {/* weight/grad label at deep zoom */}
                    {s.value != null && (
                      <text
                        className="near-view"
                        x={lp.x}
                        y={lp.y - 2.5}
                        textAnchor="middle"
                        fontSize="5"
                        fill={s.stroke === LINE ? "#A1A1AA" : s.stroke}
                        fontFamily="IBM Plex Mono, monospace"
                        style={{ pointerEvents: "none" }}
                      >
                        {s.value.toFixed(4)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>

            {/* trace highlight path */}
            {traceIds && (
              <g>
                {layout.edges
                  .filter((e) => traceIds.has(e.id))
                  .map((e) => (
                    <path
                      key={e.id}
                      d={edgePath(e)}
                      fill="none"
                      stroke={CERULEAN}
                      strokeWidth={e.id === traceEdge.id ? 2.5 : 2}
                      strokeOpacity={e.id === traceEdge.id ? 1 : 0.7}
                    />
                  ))}
              </g>
            )}

            {/* pulse overlays */}
            {pulsing != null &&
              layout.edges
                .filter((e) => e.layerIndex === pulsing)
                .map((e) => (
                  <path
                    key={`p${pulseId}-${e.id}`}
                    d={edgePath(e)}
                    pathLength="100"
                    fill="none"
                    stroke={mode === "forward" ? CERULEAN : DEEP_BLUE}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    className={mode === "forward" ? "edge-pulse" : "edge-pulse-rev"}
                    style={{ pointerEvents: "none" }}
                  />
                ))}

            {/* what-if shockwave */}
            {shockwaveKey > 0 && (
              <circle
                key={shockwaveKey}
                className="shockwave"
                cx={inputCol.x}
                cy={layout.height / 2}
                fill="none"
                stroke={CERULEAN}
                strokeWidth="1.5"
                style={{ pointerEvents: "none" }}
              />
            )}

            {/* column micro-labels */}
            <g style={{ opacity: traceIds ? 0.1 : 1, transition: "opacity 150ms" }}>
              {layout.columns.map((c) => (
                <text
                  key={c.col}
                  x={c.x}
                  y={c.top - 26}
                  textAnchor="middle"
                  fontSize="8.5"
                  fontWeight="600"
                  fill="#71717A"
                  fontFamily="Inter Variable, sans-serif"
                  style={{ letterSpacing: "0.1em" }}
                >
                  {c.isInput
                    ? "INPUT"
                    : `LAYER ${String(c.layerIndex + 1).padStart(2, "0")} · ${c.activation?.toUpperCase()}`}
                </text>
              ))}
              {/* dead-neuron x-ray counts */}
              {deadXray &&
                forwardData &&
                layout.columns
                  .filter((c) => !c.isInput && spec.layers[c.layerIndex]?.activation === "relu")
                  .map((c) => {
                    const acts = forwardData.layers[c.layerIndex]?.post_activation ?? [];
                    const dead = acts.filter((a) => a === 0).length;
                    if (dead === 0) return null;
                    return (
                      <text
                        key={`x${c.col}`}
                        x={c.x}
                        y={c.bottom + 30}
                        textAnchor="middle"
                        fontSize="8"
                        fontWeight="600"
                        fill={CRIMSON}
                        fontFamily="IBM Plex Mono, monospace"
                        style={{ letterSpacing: "0.06em" }}
                      >
                        {dead}/{acts.length} UNITS INACTIVE
                      </text>
                    );
                  })}
            </g>

            {/* nodes */}
            <g style={{ opacity: traceIds ? 0.1 : 1, transition: "opacity 150ms" }}>
              {layout.nodes.map((n) => {
                const v = nodeValue(n);
                const shown = n.isInput ? forwardData != null : fRevealed(n.layerIndex);
                const dropped = shown && isDropped(n);
                const isDead =
                  !dropped &&
                  deadXray &&
                  !n.isInput &&
                  spec.layers[n.layerIndex]?.activation === "relu" &&
                  v === 0;
                const isSel =
                  selection?.type === "node" &&
                  selection.col === n.col &&
                  selection.neuron === n.neuron;
                const mag = v != null ? Math.min(1, Math.abs(v)) : 0;
                return (
                  <g
                    key={n.id}
                    data-node={n.id}
                    transform={`translate(${n.x}, ${n.y})${isSel ? " scale(1.05)" : ""}`}
                    style={{ cursor: "pointer", transition: "transform 150ms cubic-bezier(0.16,1,0.3,1)" }}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      onSelectNode(n);
                    }}
                  >
                    <circle
                      r={NODE_R}
                      fill={
                        dropped
                          ? "rgba(161,161,170,0.10)"
                          : isDead
                            ? "rgba(239,68,68,0.15)"
                            : shown && v != null
                              ? v >= 0
                                ? `rgba(14,165,233,${0.08 + mag * 0.3})`
                                : `rgba(239,68,68,${0.08 + mag * 0.25})`
                              : "#FFFFFF"
                      }
                      stroke={dropped ? "#A1A1AA" : isDead ? CRIMSON : isSel ? CERULEAN : INK}
                      strokeWidth="1.5"
                      strokeDasharray={dropped ? "2.5 2.5" : undefined}
                      opacity={dropped ? 0.6 : 1}
                      style={{ transition: "fill 300ms, stroke 150ms" }}
                    />
                    {/* dropout: a droplet falls away each time the mask resamples */}
                    {dropped && (
                      <motion.circle
                        key={`drop-${n.id}-${forwardToken}`}
                        r={2.4}
                        cx={0}
                        fill="#A1A1AA"
                        initial={{ cy: 0, opacity: 0.9 }}
                        animate={{ cy: NODE_R * 3.4, opacity: 0 }}
                        transition={{ duration: 0.7, ease: "easeIn" }}
                        style={{ pointerEvents: "none" }}
                      />
                    )}
                    {/* activation value under node (odometer) — hidden for dropped units */}
                    {shown && v != null && !dropped && (
                      <foreignObject x={-30} y={NODE_R + 4} width="60" height="14" style={{ pointerEvents: "none", overflow: "visible" }}>
                        <div className="flex justify-center">
                          <Odometer
                            value={v}
                            decimals={3}
                            className="text-[8px] leading-none"
                            duration={0.5}
                          />
                        </div>
                      </foreignObject>
                    )}
                    {dropped && (
                      <text y={NODE_R + 11} textAnchor="middle" className="mono-num" style={{ fontSize: 7, fill: "#A1A1AA" }}>
                        drop
                      </text>
                    )}
                  </g>
                );
              })}
            </g>

            {/* floating equation chip */}
            <AnimatePresence>
              {chip && !traceIds && (
                <foreignObject
                  key={`chip-${chip.layerIndex}`}
                  x={layout.columns[chip.layerIndex + 1].x - 180}
                  y={layout.columns[chip.layerIndex + 1].top - 118}
                  width="380"
                  height="96"
                  style={{ pointerEvents: "none", overflow: "visible" }}
                >
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 8 }}
                    transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                    className="glass inline-block max-w-full px-3 py-2"
                    style={{ borderRadius: 4 }}
                  >
                    <p className="micro-label mb-1">
                      Layer {String(chip.layerIndex + 1).padStart(2, "0")} · neuron {chip.neuron} · most active
                    </p>
                    <div className="katex-chip thin-scroll overflow-x-auto whitespace-nowrap">
                      <KatexBlock chip latex={chip.linear} />
                    </div>
                    <div className="katex-chip">
                      <KatexBlock chip latex={chip.activation} />
                    </div>
                  </motion.div>
                </foreignObject>
              )}
            </AnimatePresence>
          </g>
        </g>
      </svg>

      {/* frosted vignette when a node is selected */}
      {selection?.type === "node" && (
        <div
          ref={vignetteRef}
          onClick={onClearSelection}
          className="absolute inset-0"
          style={{
            background: "rgba(255,255,255,0.6)",
            backdropFilter: "blur(3px)",
            WebkitBackdropFilter: "blur(3px)",
            maskImage:
              "radial-gradient(circle at var(--vx, 50%) var(--vy, 50%), transparent 0 70px, black 220px)",
            WebkitMaskImage:
              "radial-gradient(circle at var(--vx, 50%) var(--vy, 50%), transparent 0 70px, black 220px)",
          }}
        />
      )}
    </div>
  );
}
