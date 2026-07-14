import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Workbench from "../layout/Workbench";
import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Stepper from "../ui/Stepper";
import Odometer from "../ui/Odometer";
import ErrorBoundary from "../ui/ErrorBoundary";
import { cssColor } from "./featureTexture";
import { api } from "../../lib/api";
import { useToast } from "../ui/Toast";
import { useEngine } from "../../lib/useEngine";

const DEFAULT_SPEC = {
  input_channels: 1,
  input_height: 10,
  input_width: 10,
  seed: 7,
  num_classes: 3,
  layers: [
    { kind: "conv", out_channels: 4, kernel_size: 3, stride: 1, padding: 1, activation: "relu" },
    { kind: "pool", pool_type: "max", kernel_size: 2 },
    { kind: "conv", out_channels: 6, kernel_size: 3, stride: 1, padding: 1, activation: "relu" },
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

export default function SaliencyLab({ tabBar }) {
  const toast = useToast();
  const { online } = useEngine();
  const [spec, setSpec] = useState(DEFAULT_SPEC);
  const [image, setImage] = useState(() =>
    buildInput(DEFAULT_SPEC.input_height, DEFAULT_SPEC.input_width, SAMPLES.Diagonal)
  );
  const [sample, setSample] = useState("Diagonal");
  const [networkId, setNetworkId] = useState(null);
  const [numClasses, setNumClasses] = useState(DEFAULT_SPEC.num_classes);
  const [targetClass, setTargetClass] = useState(0);
  const [result, setResult] = useState(null); // saliency response
  const [busy, setBusy] = useState(false);
  const [brush, setBrush] = useState(1);
  const [showOverlay, setShowOverlay] = useState(true);
  const paintingRef = useRef(false);
  const netRef = useRef(null);

  // ── forge network ──
  const forge = useCallback(
    async (theSpec) => {
      setBusy(true);
      setResult(null);
      try {
        const created = await api.createCnn(theSpec);
        const prev = netRef.current;
        netRef.current = created.network_id;
        setNetworkId(created.network_id);
        setNumClasses(theSpec.num_classes);
        setTargetClass((t) => Math.min(t, theSpec.num_classes - 1));
        if (prev) api.deleteCnn(prev).catch(() => {});
      } catch (e) {
        toast(`CNN ${e.message}`);
      } finally {
        setBusy(false);
      }
    },
    [toast]
  );
  useEffect(() => {
    forge(DEFAULT_SPEC);
  }, [forge]);

  // ── recompute saliency on network / image / target change (debounced) ──
  useEffect(() => {
    if (!networkId) return;
    let alive = true;
    const t = setTimeout(() => {
      api
        .saliency({ networkId, image: [image], targetClass })
        .then((data) => alive && setResult(data))
        .catch((e) => alive && toast(`SALIENCY ${e.message}`));
    }, 120);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [networkId, image, targetClass, toast]);

  // ── painting ──
  const paint = useCallback(
    (r, c) =>
      setImage((img) => (img[r][c] === brush ? img : img.map((row, i) => (i === r ? row.map((v, j) => (j === c ? brush : v)) : row)))),
    [brush]
  );
  useEffect(() => {
    const up = () => (paintingRef.current = false);
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  }, []);

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

  const saliencyMax = useMemo(
    () => (result ? Math.max(1e-9, ...result.saliency_map.flat()) : 1),
    [result]
  );

  // ── canvas: input → saliency, + overlay ──
  const canvas = (
    <div className="relative h-full w-full overflow-auto">
      {!online ? (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
            <p className="micro-label !text-crimson mb-1">Engine offline</p>
            <p className="text-xs text-ink-soft">Start the backend on :8000 to compute saliency.</p>
          </div>
        </div>
      ) : !result ? (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="h-40 w-64 animate-pulse border border-line bg-panel" style={{ borderRadius: 4 }} />
            <p className="micro-label">{busy ? "Forging network…" : "Computing gradients…"}</p>
          </div>
        </div>
      ) : (
        <div className="dot-grid flex min-h-full flex-col items-center justify-center gap-6 p-8">
          <div className="flex flex-wrap items-start justify-center gap-6">
            <Panel title="Input" sub="paint · drag to draw">
              <PaintGrid
                grid={image}
                onPaintStart={(r, c) => {
                  paintingRef.current = true;
                  paint(r, c);
                }}
                onPaintOver={(r, c) => paintingRef.current && paint(r, c)}
              />
            </Panel>
            <span className="self-center text-ink-soft">→</span>
            <Panel title="Saliency" sub={`|∂ logit${targetClass} / ∂ input|`}>
              <HeatGrid grid={result.saliency_map} min={0} max={saliencyMax} />
            </Panel>
            {showOverlay && (
              <>
                <span className="self-center text-ink-soft">=</span>
                <Panel title="Overlay" sub="hot pixels drive the class">
                  <OverlayGrid input={image} saliency={result.saliency_map} max={saliencyMax} />
                </Panel>
              </>
            )}
          </div>
          <div className="pointer-events-none absolute bottom-4 left-4">
            <p className="micro-label">Brighter pixels change class {targetClass}'s score the most · repaint to watch saliency shift</p>
          </div>
        </div>
      )}
    </div>
  );

  // ── inspector ──
  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">{tabBar}</div>

      {/* class probabilities + target selection */}
      <div className="border-b border-line p-4">
        <ErrorBoundary resetKey={`${networkId}:${targetClass}`}>
          <ClassPanel
            result={result}
            numClasses={numClasses}
            targetClass={targetClass}
            onPick={setTargetClass}
          />
        </ErrorBoundary>
      </div>

      {/* input controls */}
      <div className="border-b border-line p-4">
        <div className="mb-2 flex items-center justify-between">
          <p className="micro-label">Input · <span className="mono-num text-ink-soft">{spec.input_height}×{spec.input_width}</span></p>
          <label className="flex items-center gap-1.5 text-[13px] text-ink-soft">
            <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
            overlay
          </label>
        </div>
        <div className="flex items-center gap-2">
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
        <div className="mb-3 grid grid-cols-2 gap-2">
          {[
            ["H", "input_height", 6, 18],
            ["W", "input_width", 6, 18],
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
            className="mono-num h-7 w-20 border border-line bg-panel px-1.5 text-[13px] focus:border-ink focus:outline-none"
            style={{ borderRadius: 3 }}
          />
        </div>
      </div>

      <div className="p-4">
        <InstrumentButton variant="primary" className="w-full" disabled={busy} onClick={onForge}>
          {busy ? "Forging…" : "Forge CNN"}
        </InstrumentButton>
        <p className="mt-3 text-[13px] leading-relaxed text-ink-soft">
          Vanilla-gradient saliency: the network runs a backward pass of the target class's logit onto
          the input, so each pixel's brightness is how much nudging it would change that score. Untrained
          weights give diffuse maps the mechanism is what matters here.
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

// ── canvas sub-components ──
function Panel({ title, sub, children }) {
  return (
    <div className="flex flex-col items-center gap-1">
      {children}
      <div className="flex flex-col items-center leading-tight">
        <span className="micro-label">{title}</span>
        {sub && <span className="mono-num text-[9px] text-ink-soft">{sub}</span>}
      </div>
    </div>
  );
}

const CANVAS_CELL = (H, W) => Math.max(14, Math.min(30, Math.floor(280 / Math.max(H, W))));

function PaintGrid({ grid, onPaintStart, onPaintOver }) {
  const H = grid.length;
  const W = grid[0].length;
  const CELL = CANVAS_CELL(H, W);
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

function HeatGrid({ grid, min, max }) {
  const H = grid.length;
  const W = grid[0].length;
  const CELL = CANVAS_CELL(H, W);
  return (
    <svg width={W * CELL} height={H * CELL} className="block border border-line" style={{ borderRadius: 3 }}>
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
          />
        ))
      )}
    </svg>
  );
}

/** Input drawn faint; saliency painted on top as a cerulean glow (alpha ∝ magnitude). */
function OverlayGrid({ input, saliency, max }) {
  const H = input.length;
  const W = input[0].length;
  const CELL = CANVAS_CELL(H, W);
  return (
    <svg width={W * CELL} height={H * CELL} className="block border border-line bg-panel" style={{ borderRadius: 3 }}>
      {input.map((row, r) =>
        row.map((v, c) => {
          const a = Math.min(1, saliency[r][c] / max);
          return (
            <g key={`${r}-${c}`}>
              <rect x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill={cssColor(v, 0, 1)} opacity={0.28} stroke="#E4E4E7" strokeWidth="0.4" />
              <rect x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill="#0EA5E9" opacity={a} />
            </g>
          );
        })
      )}
    </svg>
  );
}

// ── inspector sub-components ──
function ClassPanel({ result, numClasses, targetClass, onPick }) {
  if (!result) return <p className="micro-label">Computing…</p>;
  const probs = result.probabilities;
  const pred = result.predicted_class;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="micro-label">Class scores</p>
        <span className="mono-num text-[10px] text-ink-soft">
          predicted <span className="text-cerulean">{pred}</span>
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {Array.from({ length: numClasses }, (_, i) => {
          const p = probs[i] ?? 0;
          const isTarget = i === targetClass;
          return (
            <button
              key={i}
              type="button"
              onClick={() => onPick(i)}
              className={`flex items-center gap-2 border px-1.5 py-1 text-left transition-colors ${
                isTarget ? "border-cerulean bg-cerulean/5" : "border-transparent hover:bg-black/[0.03]"
              }`}
              style={{ borderRadius: 3 }}
            >
              <span className={`mono-num w-4 text-[13px] ${i === pred ? "text-cerulean" : "text-ink-soft"}`}>{i}</span>
              <div className="h-3 flex-1 overflow-hidden bg-line" style={{ borderRadius: 2 }}>
                <div
                  className="h-full"
                  style={{
                    width: `${p * 100}%`,
                    background: i === pred ? "#0EA5E9" : "#A1A1AA",
                    transition: "width 200ms cubic-bezier(0.16,1,0.3,1)",
                  }}
                />
              </div>
              <Odometer value={p} decimals={3} className="w-12 text-right text-[10px]" />
            </button>
          );
        })}
      </div>
      <p className="mt-2 text-[13px] leading-relaxed text-ink-soft">
        Saliency is computed for the <span className="text-cerulean">selected</span> class (click a row to
        change it) not necessarily the predicted one, so you can ask "which pixels support a class the
        net <em>didn't</em> choose?"
      </p>
    </div>
  );
}

function LayerRow({ layer, onChange, onRemove }) {
  const isConv = layer.kind === "conv";
  return (
    <div className="border border-line bg-panel p-2" style={{ borderRadius: 3 }}>
      <div className="mb-1.5 flex items-center gap-2">
        <span className={`micro-label ${isConv ? "!text-cerulean" : "!text-emerald-sig"}`}>{isConv ? "Conv" : "Pool"}</span>
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
