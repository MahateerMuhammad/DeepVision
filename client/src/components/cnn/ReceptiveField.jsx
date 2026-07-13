import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Workbench from "../layout/Workbench";
import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Stepper from "../ui/Stepper";
import ErrorBoundary from "../ui/ErrorBoundary";
import { cssColor, gridExtent } from "./featureTexture";
import { api } from "../../lib/api";
import { useToast } from "../ui/Toast";
import { useEngine } from "../../lib/useEngine";

const DEFAULT_SPEC = {
  input_channels: 1,
  input_height: 12,
  input_width: 12,
  seed: 7,
  num_classes: 3,
  layers: [
    { kind: "conv", out_channels: 3, kernel_size: 3, stride: 1, padding: 1, activation: "relu" },
    { kind: "pool", pool_type: "max", kernel_size: 2 },
    { kind: "conv", out_channels: 4, kernel_size: 3, stride: 1, padding: 0, activation: "relu" },
  ],
};

const SAMPLES = {
  Diagonal: (r, c) => (c >= r ? 1 : 0),
  Center: (r, c, H, W) => (Math.hypot(r - (H - 1) / 2, c - (W - 1) / 2) < Math.min(H, W) / 3 ? 1 : 0),
  Rings: (r, c, H, W) => (Math.round(Math.hypot(r - (H - 1) / 2, c - (W - 1) / 2)) % 2 === 0 ? 1 : 0),
  Corners: (r, c, H, W) => ((r < H / 3 || r >= (2 * H) / 3) && (c < W / 3 || c >= (2 * W) / 3) ? 1 : 0),
};
const buildInput = (H, W, fn) =>
  Array.from({ length: H }, (_, r) => Array.from({ length: W }, (_, c) => fn(r, c, H, W)));

/**
 * Trace one output pixel backward through every stage, purely geometrically.
 * Returns a map keyed by "coordinate space": space `-1` is the input image,
 * space `j` is the output of layer `j`. Each entry is a raw (unclipped) range
 * {r0,r1,c0,c1} in that space's own pixel grid — exactly the recurrence the
 * backend uses, so the final input range matches /cnn/receptive-field.
 */
function traceRegions(geom, target) {
  const regions = { [target.layerIndex]: { r0: target.r, r1: target.r, c0: target.c, c1: target.c } };
  let { r0, r1, c0, c1 } = regions[target.layerIndex];
  for (let idx = target.layerIndex; idx >= 0; idx--) {
    const g = geom[idx];
    r0 = r0 * g.s - g.p;
    r1 = r1 * g.s - g.p + g.k - 1;
    c0 = c0 * g.s - g.p;
    c1 = c1 * g.s - g.p + g.k - 1;
    regions[idx - 1] = { r0, r1, c0, c1 };
  }
  return regions;
}

export default function ReceptiveField({ tabBar }) {
  const toast = useToast();
  const { online } = useEngine();
  const [spec, setSpec] = useState(DEFAULT_SPEC);
  const [image, setImage] = useState(() =>
    buildInput(DEFAULT_SPEC.input_height, DEFAULT_SPEC.input_width, SAMPLES.Diagonal)
  );
  const [sample, setSample] = useState("Diagonal");
  const [networkId, setNetworkId] = useState(null);
  const [forward, setForward] = useState(null);
  const [target, setTarget] = useState(null); // {layerIndex, r, c}
  const [rf, setRf] = useState(null); // backend receptive-field response
  const [busy, setBusy] = useState(false);
  const netRef = useRef(null);

  // ── forge network ──
  const forge = useCallback(
    async (theSpec) => {
      setBusy(true);
      setTarget(null);
      setRf(null);
      try {
        const created = await api.createCnn(theSpec);
        const prev = netRef.current;
        netRef.current = created.network_id;
        setNetworkId(created.network_id);
        if (prev) api.deleteCnn(prev).catch(() => {});
      } catch (e) {
        toast(`CNN — ${e.message}`);
      } finally {
        setBusy(false);
      }
    },
    [toast]
  );
  useEffect(() => {
    forge(DEFAULT_SPEC);
  }, [forge]);

  // ── forward on network / input change ──
  useEffect(() => {
    if (!networkId) return;
    let alive = true;
    const t = setTimeout(() => {
      api
        .cnnForward({ networkId, image: [image] })
        .then((data) => alive && setForward(data))
        .catch((e) => alive && toast(`FORWARD — ${e.message}`));
    }, 100);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [networkId, image, toast]);

  // ── geometry of each stage (kernel/stride/pad), for the client-side trace ──
  const geom = useMemo(
    () =>
      (forward?.layers ?? []).map((L) =>
        L.kind === "conv"
          ? { k: L.kernel_size, s: L.stride, p: L.padding }
          : { k: L.kernel_size, s: L.stride || L.kernel_size, p: 0 }
      ),
    [forward]
  );

  // ── stages: input + each layer's channel-0 map (RF is channel-independent) ──
  const stages = useMemo(() => {
    if (!forward) return [];
    const out = [
      { layerIndex: -1, label: "Input", sub: "source", grid: forward.input[0] },
    ];
    for (const L of forward.layers) {
      const isConv = L.kind === "conv";
      out.push({
        layerIndex: L.layer_index,
        label: isConv ? `Conv ${L.kernel_size}×${L.kernel_size}` : `${L.pool_type}pool ${L.kernel_size}×${L.kernel_size}`,
        sub: isConv ? `s${L.stride} p${L.padding}` : `s${L.stride || L.kernel_size}`,
        grid: isConv ? L.post_activation[0] : L.output[0],
      });
    }
    return out;
  }, [forward]);

  // ── default target: deepest layer, centre pixel ──
  useEffect(() => {
    if (!forward || target) return;
    const last = forward.layers[forward.layers.length - 1];
    if (!last) return;
    const H = last.out_shape[1];
    const W = last.out_shape[2];
    setTarget({ layerIndex: last.layer_index, r: Math.floor(H / 2), c: Math.floor(W / 2) });
  }, [forward, target]);

  // ── validate target against current stages (guard stale pixel post-forge) ──
  const validTarget = useMemo(() => {
    if (!target || !forward) return null;
    // the input (layerIndex -1) is a destination, never a target — its RF is itself
    if (target.layerIndex < 0) return null;
    const st = stages.find((s) => s.layerIndex === target.layerIndex);
    if (!st) return null;
    if (target.r >= st.grid.length || target.c >= st.grid[0].length) return null;
    return target;
  }, [target, forward, stages]);

  // ── regions per coordinate space (client-side trace) ──
  const regions = useMemo(
    () => (validTarget && geom.length ? traceRegions(geom, validTarget) : null),
    [validTarget, geom]
  );

  // ── authoritative receptive field from the backend ──
  useEffect(() => {
    if (!networkId || !validTarget) return;
    let alive = true;
    api
      .receptiveField({
        networkId,
        layerIndex: validTarget.layerIndex,
        channel: 0,
        row: validTarget.r,
        col: validTarget.c,
      })
      .then((data) => alive && setRf(data))
      .catch((e) => alive && toast(`RF — ${e.message}`));
    return () => {
      alive = false;
    };
  }, [networkId, validTarget, toast]);

  // ── spec editing ──
  const patchSpec = (patch) => setSpec((s) => ({ ...s, ...patch }));
  const patchLayer = (i, patch) =>
    setSpec((s) => ({ ...s, layers: s.layers.map((l, j) => (j === i ? { ...l, ...patch } : l)) }));
  const addLayer = (kind) =>
    setSpec((s) => ({
      ...s,
      layers: [
        ...s.layers,
        kind === "conv"
          ? { kind: "conv", out_channels: 4, kernel_size: 3, stride: 1, padding: 1, activation: "relu" }
          : { kind: "pool", pool_type: "max", kernel_size: 2 },
      ],
    }));
  const removeLayer = (i) => setSpec((s) => ({ ...s, layers: s.layers.filter((_, j) => j !== i) }));

  const applySample = (name) => {
    setSample(name);
    setImage(buildInput(spec.input_height, spec.input_width, SAMPLES[name]));
  };
  const onForge = () => {
    setImage(buildInput(spec.input_height, spec.input_width, SAMPLES[sample]));
    forge(spec);
  };

  // ── canvas: the stage strip ──
  const canvas = (
    <div className="relative h-full w-full overflow-auto">
      {!online ? (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
            <p className="micro-label !text-crimson mb-1">Engine offline</p>
            <p className="text-xs text-ink-soft">Start the backend on :8000 to trace receptive fields.</p>
          </div>
        </div>
      ) : !forward ? (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="h-40 w-64 animate-pulse border border-line bg-panel" style={{ borderRadius: 4 }} />
            <p className="micro-label">{busy ? "Forging network…" : "Running forward pass…"}</p>
          </div>
        </div>
      ) : (
        <div className="dot-grid flex min-h-full items-center p-8">
          <div className="flex items-center gap-2">
            {stages.map((st, i) => (
              <div key={st.layerIndex} className="flex items-center gap-2">
                {i > 0 && <span className="text-ink-soft select-none">→</span>}
                <StageGrid
                  stage={st}
                  region={regions?.[st.layerIndex] ?? null}
                  isTargetStage={validTarget?.layerIndex === st.layerIndex}
                  active={regions != null && (st.layerIndex <= (validTarget?.layerIndex ?? -99))}
                  pickable={st.layerIndex >= 0}
                  onPick={(r, c) => setTarget({ layerIndex: st.layerIndex, r, c })}
                />
              </div>
            ))}
          </div>
          <div className="pointer-events-none absolute bottom-4 left-4">
            <p className="micro-label">Click any conv/pool neuron · the box back-traces to its input footprint</p>
          </div>
        </div>
      )}
    </div>
  );

  // ── inspector ──
  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">{tabBar}</div>

      <div className="border-b border-line p-4">
        <ErrorBoundary resetKey={`${validTarget?.layerIndex}:${validTarget?.r}:${validTarget?.c}`}>
          <RfReadout rf={rf} target={validTarget} stages={stages} regions={regions} geom={geom} forward={forward} />
        </ErrorBoundary>
      </div>

      {/* input sample */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">
          Input · <span className="mono-num text-ink-soft">{spec.input_height}×{spec.input_width}</span>
        </p>
        <div className="flex flex-wrap gap-1">
          {Object.keys(SAMPLES).map((n) => (
            <InstrumentButton key={n} size="sm" active={sample === n} onClick={() => applySample(n)}>
              {n}
            </InstrumentButton>
          ))}
        </div>
      </div>

      {/* architecture */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Architecture</p>
        <div className="mb-3 grid grid-cols-2 gap-2">
          {[
            ["H", "input_height", 6, 20],
            ["W", "input_width", 6, 20],
          ].map(([lab, key, mn, mx]) => (
            <label key={key} className="flex items-center justify-between gap-1">
              <span className="micro-label">{lab}</span>
              <Stepper value={spec[key]} min={mn} max={mx} onChange={(v) => patchSpec({ [key]: v })} />
            </label>
          ))}
        </div>
        <div className="flex flex-col gap-2">
          {spec.layers.map((l, i) => (
            <LayerRow
              key={i}
              layer={l}
              onChange={(patch) => patchLayer(i, patch)}
              onRemove={spec.layers.length > 1 ? () => removeLayer(i) : null}
            />
          ))}
        </div>
        <div className="mt-2 flex gap-1.5">
          <InstrumentButton size="sm" onClick={() => addLayer("conv")}>+ Conv</InstrumentButton>
          <InstrumentButton size="sm" onClick={() => addLayer("pool")}>+ Pool</InstrumentButton>
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="micro-label">Seed</span>
          <input
            type="number"
            value={spec.seed ?? 0}
            onChange={(e) => patchSpec({ seed: Number(e.target.value) })}
            className="mono-num h-7 w-20 border border-line bg-panel px-1.5 text-[11px] focus:border-ink focus:outline-none"
            style={{ borderRadius: 3 }}
          />
        </div>
      </div>

      <div className="p-4">
        <InstrumentButton variant="primary" className="w-full" disabled={busy} onClick={onForge}>
          {busy ? "Forging…" : "Forge CNN"}
        </InstrumentButton>
        <p className="mt-3 text-[11px] leading-relaxed text-ink-soft">
          Receptive field is pure geometry — kernel size, stride and padding — so it's identical for
          every channel and every weight. Add depth or stride to watch one neuron's window over the
          input grow.
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

// ── stage grid with region overlay ──
function StageGrid({ stage, region, isTargetStage, active, pickable = true, onPick }) {
  const grid = stage.grid;
  const H = grid.length;
  const W = grid[0].length;
  const CELL = Math.max(10, Math.min(22, Math.floor(210 / Math.max(H, W))));
  const { min, max } = gridExtent(grid);

  // clip the region to the grid for drawing; raw range may exceed bounds
  let box = null;
  if (region) {
    const r0 = Math.max(0, region.r0);
    const r1 = Math.min(H - 1, region.r1);
    const c0 = Math.max(0, region.c0);
    const c1 = Math.min(W - 1, region.c1);
    if (r1 >= r0 && c1 >= c0) {
      box = { x: c0 * CELL, y: r0 * CELL, w: (c1 - c0 + 1) * CELL, h: (r1 - r0 + 1) * CELL };
    }
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={W * CELL}
        height={H * CELL}
        className="block touch-none select-none border border-line bg-panel"
        style={{ borderRadius: 3 }}
      >
        {grid.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * CELL}
              y={r * CELL}
              width={CELL}
              height={CELL}
              fill={cssColor(v, min, max)}
              stroke="#E4E4E7"
              strokeWidth="0.4"
              style={{ cursor: pickable ? "pointer" : "default" }}
              onClick={pickable ? () => onPick(r, c) : undefined}
            />
          ))
        )}
        {box && !isTargetStage && (
          <rect
            x={box.x}
            y={box.y}
            width={box.w}
            height={box.h}
            fill="#0EA5E9"
            fillOpacity={0.14}
            stroke="#0EA5E9"
            strokeWidth={2}
            style={{ transition: "all 220ms cubic-bezier(0.16,1,0.3,1)" }}
          />
        )}
        {box && isTargetStage && (
          <rect
            x={box.x}
            y={box.y}
            width={box.w}
            height={box.h}
            fill="#0EA5E9"
            fillOpacity={0.5}
            stroke="#0EA5E9"
            strokeWidth={2.5}
          />
        )}
      </svg>
      <div className="flex flex-col items-center leading-tight">
        <span className={`micro-label ${active ? "!text-cerulean" : ""}`}>{stage.label}</span>
        <span className="mono-num text-[9px] text-ink-soft">
          {H}×{W} · {stage.sub}
        </span>
      </div>
    </div>
  );
}

// ── readout ──
function RfReadout({ rf, target, stages, regions, geom, forward }) {
  if (!forward) return <p className="micro-label">Awaiting forward pass…</p>;
  if (!target || !rf) {
    return <p className="text-[12px] leading-relaxed text-ink-soft">Click a cell in any stage to trace its receptive field.</p>;
  }
  const [rh, rw] = rf.receptive_field_size;
  const inH = forward.input[0].length;
  const inW = forward.input[0][0].length;
  const coverage = ((rh * rw) / (inH * inW)) * 100;
  const clipped =
    rf.raw_row_range[0] < 0 ||
    rf.raw_col_range[0] < 0 ||
    rf.raw_row_range[1] > inH - 1 ||
    rf.raw_col_range[1] > inW - 1;

  const targetStage = stages.find((s) => s.layerIndex === target.layerIndex);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <p className="micro-label">Receptive field</p>
        <span className="mono-num text-[10px] text-ink-soft">
          {targetStage?.label} · ({target.r}, {target.c})
        </span>
      </div>

      <div className="mb-3 flex items-baseline gap-2">
        <span className="mono-num text-2xl font-semibold text-cerulean">{rh}×{rw}</span>
        <span className="micro-label">input pixels</span>
      </div>

      <div className="mono-num flex flex-col gap-1 text-[11px] text-ink-soft">
        <div className="flex justify-between">
          <span>rows (raw)</span>
          <span className="text-ink">[{rf.raw_row_range[0]}, {rf.raw_row_range[1]}]</span>
        </div>
        <div className="flex justify-between">
          <span>cols (raw)</span>
          <span className="text-ink">[{rf.raw_col_range[0]}, {rf.raw_col_range[1]}]</span>
        </div>
        <div className="flex justify-between">
          <span>clipped rows</span>
          <span className="text-ink">[{rf.clipped_row_range[0]}, {rf.clipped_row_range[1]}]</span>
        </div>
        <div className="flex justify-between">
          <span>clipped cols</span>
          <span className="text-ink">[{rf.clipped_col_range[0]}, {rf.clipped_col_range[1]}]</span>
        </div>
        <div className="flex justify-between border-t border-line pt-1">
          <span>coverage</span>
          <span className="text-ink">{coverage.toFixed(0)}% of input</span>
        </div>
      </div>

      {clipped && (
        <p className="mt-2 text-[11px] leading-relaxed text-ink-soft">
          The raw window spills past the image edge — near the border the effective field is{" "}
          <span className="text-crimson">clipped</span>, so edge neurons see less than interior ones.
        </p>
      )}

      {/* stage-by-stage growth */}
      {regions && (
        <div className="mt-3 border-t border-line pt-2">
          <p className="micro-label mb-1.5">Window growth (back to front)</p>
          <div className="flex flex-col gap-0.5">
            {stages
              .filter((s) => s.layerIndex <= target.layerIndex)
              .slice()
              .reverse()
              .map((s) => {
                const rg = regions[s.layerIndex];
                const h = rg ? rg.r1 - rg.r0 + 1 : 1;
                const w = rg ? rg.c1 - rg.c0 + 1 : 1;
                const g = s.layerIndex >= 0 ? geom[s.layerIndex] : null;
                return (
                  <div key={s.layerIndex} className="mono-num flex items-center justify-between text-[10px]">
                    <span className="text-ink-soft">
                      {s.layerIndex === -1 ? "input" : s.label}
                      {g && <span className="ml-1 text-ink-soft/70">k{g.k} s{g.s} p{g.p}</span>}
                    </span>
                    <span className={s.layerIndex === -1 ? "font-semibold text-cerulean" : "text-ink"}>
                      {h}×{w}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── compact layer editor (mirrors Conv Explorer) ──
function LayerRow({ layer, onChange, onRemove }) {
  const isConv = layer.kind === "conv";
  return (
    <div className="border border-line bg-panel p-2" style={{ borderRadius: 3 }}>
      <div className="mb-1.5 flex items-center gap-2">
        <span className={`micro-label ${isConv ? "!text-cerulean" : "!text-emerald-sig"}`}>
          {isConv ? "Conv" : "Pool"}
        </span>
        <span className="flex-1" />
        {onRemove && (
          <button type="button" aria-label="remove layer" onClick={onRemove} className="text-ink-soft hover:text-crimson">
            ✕
          </button>
        )}
      </div>
      {isConv ? (
        <div className="flex items-center gap-1.5">
          <MiniSeg label="k" options={[3, 5]} value={layer.kernel_size} onChange={(v) => onChange({ kernel_size: v })} />
          <MiniSeg label="s" options={[1, 2]} value={layer.stride} onChange={(v) => onChange({ stride: v })} />
          <MiniSeg label="p" options={[0, 1, 2]} value={layer.padding} onChange={(v) => onChange({ padding: v })} />
        </div>
      ) : (
        <div className="flex items-center gap-1.5">
          <SegmentedControl
            size="sm"
            options={[{ value: "max", label: "Max" }, { value: "avg", label: "Avg" }]}
            value={layer.pool_type}
            onChange={(v) => onChange({ pool_type: v })}
          />
          <MiniSeg label="k" options={[2, 3]} value={layer.kernel_size} onChange={(v) => onChange({ kernel_size: v })} />
        </div>
      )}
    </div>
  );
}

function MiniSeg({ label, options, value, onChange }) {
  return (
    <div className="flex items-center gap-1">
      <span className="micro-label">{label}</span>
      <SegmentedControl size="sm" options={options.map((o) => ({ value: o, label: String(o) }))} value={value} onChange={onChange} />
    </div>
  );
}
