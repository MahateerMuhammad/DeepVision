import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence } from "framer-motion";
import Workbench from "../layout/Workbench";
import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Fader from "../ui/Fader";
import GlassChip from "../ui/GlassChip";
import { api } from "../../lib/api";
import { useToast } from "../ui/Toast";
import { useEngine } from "../../lib/useEngine";

const N = 9; // input grid is N×N
const K = 3; // kernel is K×K

// ── image presets (N×N, values 0–9) ──
const IMAGE_PRESETS = {
  Diagonal: (r, c) => (c >= r ? 9 : 0),
  Gradient: (r, c) => Math.round((c / (N - 1)) * 9),
  Cross: (r, c) => (r === (N >> 1) || c === (N >> 1) ? 9 : 0),
  Ring: (r, c) => {
    const b = 2;
    const inside = r >= b && r < N - b && c >= b && c < N - b;
    const core = r > b && r < N - b - 1 && c > b && c < N - b - 1;
    return inside && !core ? 9 : 0;
  },
  Blank: () => 0,
};
const buildImage = (fn) => Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => fn(r, c)));

// ── kernel presets (K×K) ──
const KERNEL_PRESETS = {
  Identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  Edge: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
  Sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  Blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((row) => row.map((v) => Number((v / 9).toFixed(3)))),
  "Sobel X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  "Sobel Y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
};
const BRUSHES = [0, 3, 6, 9];
const SPEEDS = [
  { value: "0.5", label: "0.5×" },
  { value: "1", label: "1×" },
  { value: "2", label: "2×" },
];

// ── color helpers ──
const lerp = (a, b, t) => Math.round(a + (b - a) * t);
function grayInk(v) {
  const t = Math.max(0, Math.min(1, v / 9));
  return `rgb(${lerp(255, 24, t)},${lerp(255, 24, t)},${lerp(255, 27, t)})`;
}
function diverging(v, maxAbs) {
  if (!maxAbs) return "#ffffff";
  const t = Math.max(-1, Math.min(1, v / maxAbs));
  return t >= 0
    ? `rgb(${lerp(255, 14, t)},${lerp(255, 165, t)},${lerp(255, 233, t)})`
    : `rgb(${lerp(255, 239, -t)},${lerp(255, 68, -t)},${lerp(255, 68, -t)})`;
}
const inkFor = (v, mid) => (Math.abs(v) > mid ? "#ffffff" : "#18181b");

export default function FilterFactory({ tabBar }) {
  const toast = useToast();
  const { online } = useEngine();
  const [image, setImage] = useState(() => buildImage(IMAGE_PRESETS.Diagonal));
  const [kernel, setKernel] = useState(() => KERNEL_PRESETS["Sobel X"].map((r) => [...r]));
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("1");
  const [brush, setBrush] = useState(9);
  const paintingRef = useRef(false);

  // ── fetch convolution (debounced) ──
  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      api
        .slidingKernel({ image, kernel, stride, padding })
        .then((data) => {
          if (!alive) return;
          setResult(data);
          setStep((s) => Math.min(s, data.steps.length - 1));
        })
        .catch((e) => alive && toast(`FILTER — ${e.message}`));
    }, 120);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [image, kernel, stride, padding, toast]);

  // ── playback ──
  useEffect(() => {
    if (!playing || !result) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= result.steps.length - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 240 / Number(speed));
    return () => clearInterval(id);
  }, [playing, speed, result]);

  const cur = result?.steps[Math.min(step, result.steps.length - 1)] ?? null;
  const outMaxAbs = useMemo(() => {
    if (!result) return 1;
    let m = 0;
    for (const row of result.output) for (const v of row) m = Math.max(m, Math.abs(v));
    return m || 1;
  }, [result]);

  // ── input painting ──
  const paint = useCallback(
    (r, c) => setImage((img) => (img[r][c] === brush ? img : img.map((row, i) => (i === r ? row.map((v, j) => (j === c ? brush : v)) : row)))),
    [brush]
  );
  useEffect(() => {
    const up = () => (paintingRef.current = false);
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  }, []);

  const setKernelCell = (r, c, raw) => {
    const v = raw === "" || raw === "-" ? raw : Number(raw);
    setKernel((k) => k.map((row, i) => (i === r ? row.map((x, j) => (j === c ? v : x)) : row)));
  };
  const commitKernel = () =>
    setKernel((k) => k.map((row) => row.map((v) => (typeof v === "number" && Number.isFinite(v) ? v : 0))));

  // ── geometry ──
  const CELL = 30;
  const pad = padding;
  const gridPx = (N + 2 * pad) * CELL;
  const inWindow = cur
    ? { r0: cur.input_row_start + pad, c0: cur.input_col_start + pad }
    : null;

  const OUT = result?.output_shape ?? [0, 0];
  const outCell = Math.max(14, Math.min(30, Math.floor(220 / Math.max(1, OUT[0]))));

  // ── canvas ──
  const canvas = (
    <div className="dot-grid relative flex h-full items-center justify-center overflow-auto p-6">
      {!online ? (
        <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
          <p className="micro-label !text-crimson mb-1">Engine offline</p>
          <p className="text-xs text-ink-soft">Start the backend on :8000 to run the filter.</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-6">
          <div className="flex flex-wrap items-start justify-center gap-8">
            {/* INPUT */}
            <div>
              <p className="micro-label mb-2">
                Input · <span className="mono-num">{N}×{N}</span>
                {pad > 0 && <span className="text-ink-soft"> · pad {pad}</span>}
              </p>
              <svg
                width={gridPx}
                height={gridPx}
                className="block touch-none select-none"
                style={{ maxWidth: 380 }}
              >
                {Array.from({ length: N + 2 * pad }).map((_, r) =>
                  Array.from({ length: N + 2 * pad }).map((__, c) => {
                    const or = r - pad;
                    const oc = c - pad;
                    const isPad = or < 0 || or >= N || oc < 0 || oc >= N;
                    const v = isPad ? 0 : image[or][oc];
                    return (
                      <g key={`${r}-${c}`}>
                        <rect
                          x={c * CELL}
                          y={r * CELL}
                          width={CELL}
                          height={CELL}
                          fill={isPad ? "#FAFAFA" : grayInk(v)}
                          stroke="#E4E4E7"
                          strokeWidth="1"
                          strokeDasharray={isPad ? "2 2" : undefined}
                          style={{ cursor: isPad ? "default" : "crosshair" }}
                          onPointerDown={(e) => {
                            if (isPad) return;
                            e.preventDefault();
                            paintingRef.current = true;
                            paint(or, oc);
                          }}
                          onPointerEnter={() => !isPad && paintingRef.current && paint(or, oc)}
                        />
                        {!isPad && (
                          <text
                            x={c * CELL + CELL / 2}
                            y={r * CELL + CELL / 2}
                            dominantBaseline="central"
                            textAnchor="middle"
                            className="mono-num pointer-events-none"
                            fontSize="10"
                            fill={inkFor(v, 5)}
                          >
                            {v}
                          </text>
                        )}
                      </g>
                    );
                  })
                )}
                {/* sliding window */}
                {inWindow && (
                  <rect
                    x={inWindow.c0 * CELL}
                    y={inWindow.r0 * CELL}
                    width={K * CELL}
                    height={K * CELL}
                    fill="none"
                    stroke="#0EA5E9"
                    strokeWidth="2.5"
                    style={{ transition: "x 120ms cubic-bezier(0.16,1,0.3,1), y 120ms cubic-bezier(0.16,1,0.3,1)" }}
                  />
                )}
              </svg>
            </div>

            {/* OUTPUT */}
            <div>
              <p className="micro-label mb-2">
                Feature map · <span className="mono-num">{OUT[0]}×{OUT[1]}</span>
              </p>
              <svg width={OUT[1] * outCell} height={OUT[0] * outCell} className="block" style={{ maxWidth: 300 }}>
                {result?.output.map((row, r) =>
                  row.map((v, c) => {
                    const isCur = cur && cur.output_row === r && cur.output_col === c;
                    return (
                      <g key={`${r}-${c}`}>
                        <rect
                          x={c * outCell}
                          y={r * outCell}
                          width={outCell}
                          height={outCell}
                          fill={diverging(v, outMaxAbs)}
                          stroke={isCur ? "#0EA5E9" : "#E4E4E7"}
                          strokeWidth={isCur ? 2.5 : 1}
                        />
                        {outCell >= 22 && (
                          <text
                            x={c * outCell + outCell / 2}
                            y={r * outCell + outCell / 2}
                            dominantBaseline="central"
                            textAnchor="middle"
                            className="mono-num pointer-events-none"
                            fontSize="8"
                            fill={inkFor(v, outMaxAbs * 0.6)}
                          >
                            {v.toFixed(1)}
                          </text>
                        )}
                      </g>
                    );
                  })
                )}
              </svg>
            </div>
          </div>

          {/* computation readout */}
          <AnimatePresence mode="wait">
            {cur && (
              <GlassChip key={step} className="!px-4 !py-3">
                <div className="flex items-center gap-3">
                  <span className="micro-label">
                    Cell <span className="mono-num text-ink">({cur.output_row},{cur.output_col})</span>
                  </span>
                  <MiniGrid values={cur.patch} label="patch" color={(v) => grayInk(v)} textMid={5} />
                  <span className="text-ink-soft">⊙</span>
                  <MiniGrid values={kernel} label="kernel" />
                  <span className="text-ink-soft">=</span>
                  <MiniGrid values={cur.elementwise_products} label="products" mono />
                  <span className="text-ink-soft">Σ</span>
                  <span className="mono-num text-lg font-semibold text-cerulean">{cur.value.toFixed(3)}</span>
                </div>
              </GlassChip>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );

  // ── inspector ──
  const kernelValid = kernel.every((row) => row.every((v) => typeof v === "number" && Number.isFinite(v)));
  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">{tabBar}</div>

      {/* kernel */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Kernel · {K}×{K}</p>
        <div className="mb-3 inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${K}, minmax(0,1fr))` }}>
          {kernel.map((row, r) =>
            row.map((v, c) => (
              <input
                key={`${r}-${c}`}
                type="text"
                inputMode="numeric"
                value={String(v)}
                onChange={(e) => setKernelCell(r, c, e.target.value.trim())}
                onBlur={commitKernel}
                onKeyDown={(e) => e.key === "Enter" && commitKernel()}
                className="mono-num h-11 w-11 border border-line bg-panel text-center text-[12px] transition-colors duration-150 focus:border-cerulean focus:outline-none"
                style={{ borderRadius: 3 }}
              />
            ))
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {Object.keys(KERNEL_PRESETS).map((name) => (
            <InstrumentButton
              key={name}
              size="sm"
              onClick={() => setKernel(KERNEL_PRESETS[name].map((r) => [...r]))}
            >
              {name}
            </InstrumentButton>
          ))}
        </div>
      </div>

      {/* stride / padding */}
      <div className="border-b border-line p-4">
        <div className="mb-1 flex items-center justify-between">
          <span className="micro-label">Stride</span>
          <span className="mono-num text-[11px]">{stride}</span>
        </div>
        <Fader value={stride} min={1} max={3} step={1} ticks={3} onChange={(v) => setStride(Math.round(v))} />
        <div className="mb-1 mt-4 flex items-center justify-between">
          <span className="micro-label">Padding</span>
          <span className="mono-num text-[11px]">{padding}</span>
        </div>
        <Fader value={padding} min={0} max={2} step={1} ticks={3} onChange={(v) => setPadding(Math.round(v))} />
        <p className="mt-3 text-[11px] leading-relaxed text-ink-soft">
          Stride skips input positions; padding rings the image in zeros (dashed cells) so the
          kernel can reach the border.
        </p>
      </div>

      {/* image */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Input image</p>
        <div className="mb-3 flex flex-wrap gap-1.5">
          {Object.keys(IMAGE_PRESETS).map((name) => (
            <InstrumentButton key={name} size="sm" onClick={() => setImage(buildImage(IMAGE_PRESETS[name]))}>
              {name}
            </InstrumentButton>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="micro-label">Brush</span>
          {BRUSHES.map((b) => (
            <button
              key={b}
              type="button"
              aria-label={`brush ${b}`}
              onClick={() => setBrush(b)}
              className={`h-6 w-6 border ${brush === b ? "border-cerulean" : "border-line"}`}
              style={{ borderRadius: 3, background: grayInk(b) }}
            />
          ))}
          <span className="mono-num ml-auto text-[11px] text-ink-soft">paint 0–9 · drag to fill</span>
        </div>
      </div>

      {/* VCR */}
      <div className="p-4">
        <div className="mb-3 flex items-center gap-2">
          <InstrumentButton size="icon" title="Reset" onClick={() => { setPlaying(false); setStep(0); }}>
            ⏮
          </InstrumentButton>
          <InstrumentButton
            size="icon"
            title={playing ? "Pause" : "Play"}
            active={playing}
            disabled={!result || !kernelValid}
            onClick={() => {
              if (result && step >= result.steps.length - 1) setStep(0);
              setPlaying((p) => !p);
            }}
          >
            {playing ? "⏸" : "▶"}
          </InstrumentButton>
          <InstrumentButton
            size="icon"
            title="Step"
            disabled={!result || !kernelValid}
            onClick={() => {
              setPlaying(false);
              setStep((s) => Math.min(s + 1, (result?.steps.length ?? 1) - 1));
            }}
          >
            ⏭
          </InstrumentButton>
          <SegmentedControl size="sm" options={SPEEDS} value={speed} onChange={setSpeed} className="ml-auto" />
        </div>
        <div className="flex items-center justify-between">
          <span className="micro-label">Position</span>
          <span className="mono-num text-[11px]">
            {result ? `${Math.min(step + 1, result.steps.length)} / ${result.steps.length}` : "—"}
          </span>
        </div>
        {result && (
          <div className="mt-2 h-1 w-full bg-line" style={{ borderRadius: 2 }}>
            <div
              className="h-full bg-cerulean"
              style={{
                width: `${((step + 1) / result.steps.length) * 100}%`,
                borderRadius: 2,
                transition: "width 120ms cubic-bezier(0.16,1,0.3,1)",
              }}
            />
          </div>
        )}
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

function MiniGrid({ values, label, color, mono, textMid = 0 }) {
  const cols = values[0]?.length ?? 0;
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
        {values.flat().map((v, i) => (
          <span
            key={i}
            className="mono-num flex h-6 w-6 items-center justify-center text-[9px]"
            style={{
              background: color ? color(v) : "transparent",
              color: color ? inkFor(v, textMid) : "#18181b",
              border: color ? "none" : "1px solid #E4E4E7",
              borderRadius: 2,
            }}
          >
            {mono ? Number(v).toFixed(1) : v}
          </span>
        ))}
      </div>
      <span className="micro-label !text-[8px]">{label}</span>
    </div>
  );
}
