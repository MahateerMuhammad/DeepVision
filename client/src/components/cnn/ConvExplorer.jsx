import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Workbench from "../layout/Workbench";
import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Stepper from "../ui/Stepper";
import Odometer from "../ui/Odometer";
import ErrorBoundary from "../ui/ErrorBoundary";
import FeatureScene from "./FeatureScene";
import { cssColor, gridExtent, RAMP_CSS } from "./featureTexture";
import { api } from "../../lib/api";
import { useToast } from "../ui/Toast";
import { useEngine } from "../../lib/useEngine";

const ACTS = ["relu", "sigmoid", "tanh", "linear", "leaky_relu"];
const DEFAULT_SPEC = {
  input_channels: 1,
  input_height: 8,
  input_width: 8,
  seed: 7,
  num_classes: 3,
  layers: [
    { kind: "conv", out_channels: 3, kernel_size: 3, stride: 1, padding: 1, activation: "relu" },
    { kind: "pool", pool_type: "max", kernel_size: 2 },
    { kind: "conv", out_channels: 4, kernel_size: 3, stride: 1, padding: 0, activation: "relu" },
  ],
};

const SAMPLES = {
  Diagonal: (ch, r, c) => (c >= r + ch ? 1 : 0),
  Bars: (ch, r, c) => ((c + ch) % 3 === 0 ? 1 : 0),
  Center: (ch, r, c, H, W) => (Math.hypot(r - (H - 1) / 2, c - (W - 1) / 2) < Math.min(H, W) / 3 - ch ? 1 : 0),
  Rings: (ch, r, c, H, W) => (Math.round(Math.hypot(r - (H - 1) / 2, c - (W - 1) / 2)) % 2 === 0 ? 1 : 0),
};
const buildInput = (C, H, W, fn) =>
  Array.from({ length: C }, (_, ch) =>
    Array.from({ length: H }, (_, r) => Array.from({ length: W }, (_, c) => fn(ch, r, c, H, W)))
  );

const softmax = (a) => {
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
};

export default function ConvExplorer({ tabBar }) {
  const toast = useToast();
  const { online } = useEngine();
  const [spec, setSpec] = useState(DEFAULT_SPEC);
  const [image, setImage] = useState(() =>
    buildInput(DEFAULT_SPEC.input_channels, DEFAULT_SPEC.input_height, DEFAULT_SPEC.input_width, SAMPLES.Diagonal)
  );
  const [sample, setSample] = useState("Diagonal");
  const [networkId, setNetworkId] = useState(null);
  const [forward, setForward] = useState(null);
  const [outputShape, setOutputShape] = useState(null);
  const [paramCount, setParamCount] = useState(null);
  const [selected, setSelected] = useState(null); // {layerIndex, channel}
  const [busy, setBusy] = useState(false);
  const [brush, setBrush] = useState(1);
  const paintingRef = useRef(false);
  const netRef = useRef(null);

  // ── forge (create network) ──
  const forge = useCallback(
    async (theSpec) => {
      setBusy(true);
      setSelected(null);
      try {
        const created = await api.createCnn(theSpec);
        const prev = netRef.current;
        netRef.current = created.network_id;
        setNetworkId(created.network_id);
        setOutputShape(created.output_shape);
        setParamCount(created.param_count);
        if (prev) api.deleteCnn(prev).catch(() => {});
      } catch (e) {
        toast(`CNN — ${e.message}`);
      } finally {
        setBusy(false);
      }
    },
    [toast]
  );

  // initial forge
  useEffect(() => {
    forge(DEFAULT_SPEC);
  }, [forge]);

  // ── forward whenever network or input changes (debounced) ──
  useEffect(() => {
    if (!networkId) return;
    let alive = true;
    const t = setTimeout(() => {
      api
        .cnnForward({ networkId, image })
        .then((data) => alive && setForward(data))
        .catch((e) => alive && toast(`FORWARD — ${e.message}`));
    }, 100);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [networkId, image, toast]);

  // ── input painting (channel 0) ──
  const paint = useCallback(
    (r, c) =>
      setImage((img) =>
        img[0][r][c] === brush
          ? img
          : img.map((ch, k) => (k === 0 ? ch.map((row, i) => (i === r ? row.map((v, j) => (j === c ? brush : v)) : row)) : ch))
      ),
    [brush]
  );
  useEffect(() => {
    const up = () => (paintingRef.current = false);
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  }, []);

  const applySample = (name) => {
    setSample(name);
    setImage(buildInput(spec.input_channels, spec.input_height, spec.input_width, SAMPLES[name]));
  };

  // ── spec editing helpers ──
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

  const onForge = () => {
    // resize input to match new dims using the current sample
    setImage(buildInput(spec.input_channels, spec.input_height, spec.input_width, SAMPLES[sample]));
    forge(spec);
  };

  // ── scene data ──
  const sceneLayers = useMemo(() => {
    if (!forward) return [];
    const out = [
      {
        layerIndex: -1,
        label: "Input",
        sublabel: "source image",
        kind: "input",
        channels: forward.input,
        spatial: [forward.input[0].length, forward.input[0][0].length],
      },
    ];
    for (const L of forward.layers) {
      const isConv = L.kind === "conv";
      out.push({
        layerIndex: L.layer_index,
        label: isConv ? `Conv ${L.kernel_size}×${L.kernel_size}` : `${L.pool_type}pool ${L.kernel_size}×${L.kernel_size}`,
        sublabel: isConv ? `s${L.stride} p${L.padding} · ${L.activation}` : `stride ${L.stride}`,
        kind: L.kind,
        channels: isConv ? L.post_activation : L.output,
        spatial: [L.out_shape[1], L.out_shape[2]],
        raw: L,
      });
    }
    return out;
  }, [forward]);

  const selectedLayer = useMemo(() => {
    if (!selected || !forward) return null;
    if (selected.layerIndex === -1)
      return { kind: "input", channels: forward.input, spatial: [forward.input[0].length, forward.input[0][0].length] };
    return forward.layers.find((l) => l.layer_index === selected.layerIndex) ?? null;
  }, [selected, forward]);

  // ── canvas ──
  const canvas = (
    <div className="relative h-full w-full">
      {!online ? (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
            <p className="micro-label !text-crimson mb-1">Engine offline</p>
            <p className="text-xs text-ink-soft">Start the backend on :8000 to build a CNN.</p>
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
        <>
          <ErrorBoundary resetKey={networkId}>
            <FeatureScene
              layers={sceneLayers}
              selected={selected}
              onSelect={(layerIndex, channel) =>
                setSelected(layerIndex == null ? null : { layerIndex, channel })
              }
            />
          </ErrorBoundary>
          <div className="pointer-events-none absolute bottom-4 left-4">
            <p className="micro-label">Drag to orbit · scroll to zoom · click a pane to inspect</p>
          </div>
          {/* heat-map legend */}
          <div
            className="pointer-events-none absolute right-4 bottom-4 flex items-center gap-2 border border-line bg-panel/85 px-2.5 py-1.5 shadow-instrument"
            style={{ borderRadius: 4, backdropFilter: "blur(12px)" }}
          >
            <span className="micro-label">Low</span>
            <div className="h-2 w-24" style={{ borderRadius: 2, background: `linear-gradient(90deg, ${RAMP_CSS})` }} />
            <span className="micro-label">High</span>
            <span className="micro-label ml-1 text-ink-soft">activation</span>
          </div>
          {outputShape && (
            <div className="pointer-events-none absolute top-4 left-4 border border-line bg-panel/85 px-3 py-1.5 shadow-instrument" style={{ borderRadius: 4, backdropFilter: "blur(12px)" }}>
              <span className="micro-label">Feature volume </span>
              <span className="mono-num text-xs text-ink">{outputShape.join("×")}</span>
              {paramCount != null && (
                <>
                  <span className="micro-label"> · params </span>
                  <span className="mono-num text-xs text-ink">{paramCount}</span>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );

  // ── inspector ──
  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">{tabBar}</div>

      {/* selection detail / classifier output */}
      <div className="border-b border-line p-4">
        <ErrorBoundary resetKey={`${selected?.layerIndex}:${selected?.channel}`}>
          {selectedLayer ? (
            <SelectionDetail layer={selectedLayer} channel={selected.channel} onClear={() => setSelected(null)} />
          ) : (
            <ClassifierReadout forward={forward} numClasses={spec.num_classes} />
          )}
        </ErrorBoundary>
      </div>

      {/* input */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">
          Input · <span className="mono-num text-ink-soft">{spec.input_channels}×{spec.input_height}×{spec.input_width}</span>
          {spec.input_channels > 1 && <span className="text-ink-soft"> · painting ch 0</span>}
        </p>
        <PaintGrid
          grid={image[0]}
          brush={brush}
          onPaintStart={(r, c) => {
            paintingRef.current = true;
            paint(r, c);
          }}
          onPaintOver={(r, c) => paintingRef.current && paint(r, c)}
        />
        <div className="mt-2 flex items-center gap-2">
          <span className="micro-label">Brush</span>
          {[0, 1].map((b) => (
            <button
              key={b}
              type="button"
              aria-label={`brush ${b}`}
              onClick={() => setBrush(b)}
              className={`h-5 w-5 border ${brush === b ? "border-cerulean" : "border-line"}`}
              style={{ borderRadius: 3, background: cssColor(b, 0, 1) }}
            />
          ))}
          <div className="ml-auto flex flex-wrap gap-1">
            {Object.keys(SAMPLES).map((n) => (
              <InstrumentButton key={n} size="sm" active={sample === n} onClick={() => applySample(n)}>
                {n}
              </InstrumentButton>
            ))}
          </div>
        </div>
      </div>

      {/* architecture */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Architecture</p>
        <div className="mb-3 grid grid-cols-3 gap-2">
          {[
            ["Ch", "input_channels", 1, 3],
            ["H", "input_height", 4, 16],
            ["W", "input_width", 4, 16],
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
          <span className="micro-label">Classes</span>
          <Stepper value={spec.num_classes ?? 2} min={2} max={6} onChange={(v) => patchSpec({ num_classes: v })} />
        </div>
        <div className="mt-2 flex items-center justify-between">
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
          Editing the input repaints the feature towers live. Architecture changes need a re-forge
          (fresh random weights from the seed).
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

// ── sub-components ──

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
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <span className="micro-label">Filters</span>
            <Stepper value={layer.out_channels} min={1} max={8} onChange={(v) => onChange({ out_channels: v })} />
          </div>
          <div className="flex items-center gap-1.5">
            <MiniSeg label="k" options={[3, 5]} value={layer.kernel_size} onChange={(v) => onChange({ kernel_size: v })} />
            <MiniSeg label="s" options={[1, 2]} value={layer.stride} onChange={(v) => onChange({ stride: v })} />
            <MiniSeg label="p" options={[0, 1, 2]} value={layer.padding} onChange={(v) => onChange({ padding: v })} />
          </div>
          <select
            value={layer.activation}
            onChange={(e) => onChange({ activation: e.target.value })}
            className="mono-num h-7 border border-line bg-panel px-1.5 text-[11px] focus:border-ink focus:outline-none"
            style={{ borderRadius: 3 }}
          >
            {ACTS.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
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

function PaintGrid({ grid, onPaintStart, onPaintOver }) {
  const H = grid.length;
  const W = grid[0].length;
  const CELL = Math.max(12, Math.min(22, Math.floor(200 / Math.max(H, W))));
  return (
    <svg width={W * CELL} height={H * CELL} className="block touch-none select-none border border-line" style={{ borderRadius: 3 }}>
      {grid.map((row, r) =>
        row.map((v, c) => (
          <rect
            key={`${r}-${c}`}
            x={c * CELL}
            y={r * CELL}
            width={CELL}
            height={CELL}
            fill={cssColor(v, 0, 1)}
            stroke="#E4E4E7"
            strokeWidth="0.5"
            style={{ cursor: "crosshair" }}
            onPointerDown={(e) => {
              e.preventDefault();
              onPaintStart(r, c);
            }}
            onPointerEnter={() => onPaintOver(r, c)}
          />
        ))
      )}
    </svg>
  );
}

function MiniHeat({ grid, cell = 16, onCellClick, selected }) {
  const { min, max } = gridExtent(grid);
  const H = grid.length;
  const W = grid[0].length;
  const c = Math.max(8, Math.min(cell, Math.floor(180 / Math.max(H, W))));
  return (
    <svg width={W * c} height={H * c} className="block border border-line" style={{ borderRadius: 3 }}>
      {grid.map((row, r) =>
        row.map((v, cc) => {
          const isSel = selected && selected.r === r && selected.c === cc;
          return (
            <rect
              key={`${r}-${cc}`}
              x={cc * c}
              y={r * c}
              width={c}
              height={c}
              fill={cssColor(v, min, max)}
              stroke={isSel ? "#0EA5E9" : "#E4E4E7"}
              strokeWidth={isSel ? 2 : 0.4}
              style={onCellClick ? { cursor: "pointer" } : undefined}
              onClick={onCellClick ? () => onCellClick(r, cc) : undefined}
            />
          );
        })
      )}
    </svg>
  );
}

function SelectionDetail({ layer, channel, onClear }) {
  if (layer.kind === "pool") return <PoolingDetail layer={layer} channel={channel} onClear={onClear} />;
  if (layer.kind === "input") return <InputDetail layer={layer} channel={channel} onClear={onClear} />;
  return <ConvDetail layer={layer} channel={channel} onClear={onClear} />;
}

function DetailHeader({ title, channel, onClear }) {
  return (
    <div className="mb-2 flex items-center justify-between">
      <p className="micro-label">
        {title} · <span className="text-ink">ch {channel}</span>
      </p>
      <button type="button" onClick={onClear} className="micro-label hover:text-ink">
        Clear
      </button>
    </div>
  );
}

function StatRow({ grid }) {
  const { min, max } = gridExtent(grid);
  const flat = grid.flat();
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
  return (
    <div className="mono-num mt-2 flex gap-3 text-[10px] text-ink-soft">
      <span>min {min.toFixed(2)}</span>
      <span>max {max.toFixed(2)}</span>
      <span>μ {mean.toFixed(2)}</span>
    </div>
  );
}

function InputDetail({ layer, channel, onClear }) {
  const grid = layer.channels[channel];
  return (
    <div>
      <DetailHeader title="Input" channel={channel} onClear={onClear} />
      <MiniHeat grid={grid} />
      <StatRow grid={grid} />
    </div>
  );
}

function ConvDetail({ layer, channel, onClear }) {
  const grid = layer.post_activation[channel];
  const [pixel, setPixel] = useState(null);
  useEffect(() => setPixel(null), [layer, channel]);
  // Guard against a stale pixel from a previously-selected (larger) channel:
  // the reset effect runs after render, so bounds-check synchronously here.
  const valid = pixel && pixel.r < grid.length && pixel.c < grid[0].length ? pixel : null;
  return (
    <div>
      <DetailHeader title={`Conv · ${layer.activation}`} channel={channel} onClear={onClear} />
      <MiniHeat grid={grid} onCellClick={(r, c) => setPixel({ r, c })} selected={valid} />
      <StatRow grid={grid} />
      {valid ? (
        <ConvPixelCalc layer={layer} channel={channel} pixel={valid} />
      ) : (
        <p className="mt-2 text-[11px] leading-relaxed text-ink-soft">
          Click a cell above to see the exact convolution that produced it.
        </p>
      )}
    </div>
  );
}

function convPixelBreakdown(layer, oc, or, ocol) {
  const k = layer.kernel_size;
  const s = layer.stride;
  const p = layer.padding;
  const W = layer.weights[oc]; // Cin × k × k
  const channels = [];
  let total = 0;
  for (let ic = 0; ic < layer.input.length; ic++) {
    const src = layer.input[ic];
    const H = src.length;
    const Wd = src[0].length;
    const patch = [];
    const products = [];
    let partial = 0;
    for (let ki = 0; ki < k; ki++) {
      const prow = [];
      const erow = [];
      for (let kj = 0; kj < k; kj++) {
        const ir = or * s + ki - p;
        const jc = ocol * s + kj - p;
        const x = ir >= 0 && ir < H && jc >= 0 && jc < Wd ? src[ir][jc] : 0;
        const prod = W[ic][ki][kj] * x;
        prow.push(x);
        erow.push(prod);
        partial += prod;
      }
      patch.push(prow);
      products.push(erow);
    }
    total += partial;
    channels.push({ ic, patch, kernel: W[ic], products, partial });
  }
  return { channels, bias: layer.biases[oc], preTotal: total };
}

const actName = { relu: "relu", sigmoid: "σ", tanh: "tanh", linear: "id", leaky_relu: "lrelu" };

function ConvPixelCalc({ layer, channel, pixel }) {
  const { channels, bias } = convPixelBreakdown(layer, channel, pixel.r, pixel.c);
  const z = layer.pre_activation[channel][pixel.r][pixel.c];
  const a = layer.post_activation[channel][pixel.r][pixel.c];
  const multi = channels.length > 1;
  return (
    <div className="mt-3 border-t border-line pt-3">
      <p className="micro-label mb-2">
        Output pixel <span className="mono-num text-ink">({pixel.r}, {pixel.c})</span>
        {multi && <span className="text-ink-soft"> · sums {channels.length} input channels</span>}
      </p>
      <div className="flex flex-col gap-2.5">
        {channels.map((ch) => (
          <div key={ch.ic} className="flex flex-wrap items-center gap-2">
            {multi && <span className="mono-num w-8 text-[10px] text-ink-soft">in {ch.ic}</span>}
            <CalcGrid values={ch.patch} tint />
            <span className="text-ink-soft">⊙</span>
            <CalcGrid values={ch.kernel} />
            <span className="text-ink-soft">→</span>
            <span className="mono-num text-[11px]" style={{ color: ch.partial >= 0 ? "#0EA5E9" : "#EF4444" }}>
              {ch.partial.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
      <div className="mono-num mt-3 flex flex-col gap-1 border-t border-line pt-2 text-[11px]">
        <div>
          <span className="text-ink-soft">{multi ? "Σ channels" : "Σ"} + bias = </span>
          <span className="text-ink-soft">
            {channels.reduce((s, c) => s + c.partial, 0).toFixed(3)} + ({bias.toFixed(3)})
          </span>
          <span className="ml-1">= </span>
          <span className="font-semibold text-ink">z = {z.toFixed(3)}</span>
        </div>
        <div>
          <span className="text-ink-soft">a = {actName[layer.activation] ?? layer.activation}(z) = </span>
          <span className="font-semibold text-cerulean">{a.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

function CalcGrid({ values, tint }) {
  const { min, max } = gridExtent(values);
  const cols = values[0]?.length ?? 0;
  return (
    <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
      {values.flat().map((v, i) => (
        <span
          key={i}
          className="mono-num flex h-6 w-6 items-center justify-center text-[9px]"
          style={{
            background: tint ? cssColor(v, min, max) : "transparent",
            color: tint && (v - min) / (max - min || 1) > 0.6 ? "#fff" : "#18181b",
            border: tint ? "none" : "1px solid #E4E4E7",
            borderRadius: 2,
          }}
        >
          {Number.isInteger(v) ? v : v.toFixed(1)}
        </span>
      ))}
    </div>
  );
}

function PoolingDetail({ layer, channel, onClear }) {
  const input = layer.input[channel];
  const output = layer.output[channel];
  const mask = layer.kept_mask?.[channel];
  const { min, max } = gridExtent(input);
  const H = input.length;
  const W = input[0].length;
  const cell = Math.max(9, Math.min(18, Math.floor(150 / Math.max(H, W))));
  const dropped = mask ? mask.flat().filter((v) => v === 0).length : 0;
  const total = H * W;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="micro-label">
          {layer.pool_type}-pool · <span className="text-ink">ch {channel}</span>
        </p>
        <button type="button" onClick={onClear} className="micro-label hover:text-ink">
          Clear
        </button>
      </div>
      <div className="flex items-end gap-3">
        <div>
          <svg width={W * cell} height={H * cell} className="block border border-line" style={{ borderRadius: 3 }}>
            {input.map((row, r) =>
              row.map((v, c) => {
                const kept = !mask || mask[r][c] === 1;
                return (
                  <rect
                    key={`${r}-${c}`}
                    x={c * cell}
                    y={r * cell}
                    width={cell}
                    height={cell}
                    fill={cssColor(v, min, max)}
                    opacity={kept ? 1 : 0.18}
                    stroke={kept && mask ? "#0EA5E9" : "#E4E4E7"}
                    strokeWidth={kept && mask ? 1.2 : 0.4}
                  />
                );
              })
            )}
          </svg>
          <p className="micro-label mt-1">input {H}×{W}</p>
        </div>
        <span className="mb-4 text-ink-soft">→</span>
        <div>
          <MiniHeat grid={output} cell={cell + 4} />
          <p className="micro-label mt-1">
            output {output.length}×{output[0].length}
          </p>
        </div>
      </div>
      {mask && (
        <p className="mt-2 text-[11px] leading-relaxed text-ink-soft">
          Keeps <span className="mono-num text-ink">{total - dropped}</span> of{" "}
          <span className="mono-num text-ink">{total}</span> cells — drops{" "}
          <span className="mono-num text-crimson">{((dropped / total) * 100).toFixed(0)}%</span> of the data.
          Cerulean cells survived the {layer.pool_type} window.
        </p>
      )}
    </div>
  );
}

function ClassifierReadout({ forward, numClasses }) {
  if (!forward) return <p className="micro-label">Awaiting forward pass…</p>;
  const logits = forward.output;
  const probs = softmax(logits);
  const top = probs.indexOf(Math.max(...probs));
  return (
    <div>
      <p className="micro-label mb-2">
        Classifier output · <span className="text-ink-soft">{numClasses} classes</span>
      </p>
      <div className="flex flex-col gap-1.5">
        {probs.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`mono-num w-4 text-[11px] ${i === top ? "text-cerulean" : "text-ink-soft"}`}>{i}</span>
            <div className="h-3 flex-1 overflow-hidden bg-line" style={{ borderRadius: 2 }}>
              <div
                className="h-full"
                style={{
                  width: `${p * 100}%`,
                  background: i === top ? "#0EA5E9" : "#A1A1AA",
                  transition: "width 200ms cubic-bezier(0.16,1,0.3,1)",
                }}
              />
            </div>
            <Odometer value={p} decimals={3} className="w-12 text-right text-[10px]" />
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-ink-soft">
        Predicted class <span className="mono-num text-ink">{top}</span>. Click any pane to inspect a
        feature map; pool panes reveal what pooling discards.
      </p>
    </div>
  );
}
