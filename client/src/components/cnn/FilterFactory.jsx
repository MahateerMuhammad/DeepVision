import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Workbench from "../layout/Workbench";
import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Fader from "../ui/Fader";
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
  { value: "0.25", label: "0.25×" },
  { value: "0.5", label: "0.5×" },
  { value: "1", label: "1×" },
  { value: "2", label: "2×" },
];
const BASE_INTERVAL = 340; // ms per step at 1× (slow enough to read the math)

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
const signColor = (v) => (v > 0 ? "#0EA5E9" : v < 0 ? "#EF4444" : "#A1A1AA");
const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

export default function FilterFactory({ tabBar }) {
  const toast = useToast();
  const { online } = useEngine();
  const [image, setImage] = useState(() => buildImage(IMAGE_PRESETS.Diagonal));
  const [kernel, setKernel] = useState(() => KERNEL_PRESETS["Sobel X"].map((r) => [...r]));
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(0);
  const [hoverCell, setHoverCell] = useState(null); // {r,c} previewed without committing
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("1");
  const [brush, setBrush] = useState(9);
  const [reveal, setReveal] = useState(true);
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
    }, BASE_INTERVAL / Number(speed));
    return () => clearInterval(id);
  }, [playing, speed, result]);

  const OUT = result?.output_shape ?? [0, 0];
  const outW = OUT[1] || 1;
  const stepIndexOf = useCallback((r, c) => r * outW + c, [outW]);

  // active cell = hovered preview (if any) else the stepped cell
  const activeStep = hoverCell ? stepIndexOf(hoverCell.r, hoverCell.c) : step;
  const active = result?.steps[Math.min(Math.max(activeStep, 0), (result?.steps.length ?? 1) - 1)] ?? null;

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

  const jumpTo = (r, c) => {
    setPlaying(false);
    setHoverCell(null);
    setStep(Math.min(stepIndexOf(r, c), (result?.steps.length ?? 1) - 1));
  };

  // ── geometry ──
  const CELL = 30;
  const pad = padding;
  const gridPx = (N + 2 * pad) * CELL;
  const inWindow = active ? { r0: active.input_row_start + pad, c0: active.input_col_start + pad } : null;
  const outCell = Math.max(16, Math.min(34, Math.floor(230 / Math.max(1, OUT[0]))));

  // ── canvas ──
  const canvas = (
    <div className="dot-grid relative flex h-full items-center justify-center overflow-auto p-6">
      {!online ? (
        <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
          <p className="micro-label !text-crimson mb-1">Engine offline</p>
          <p className="text-xs text-ink-soft">Start the backend on :8000 to run the filter.</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-5">
          <div className="flex flex-wrap items-start justify-center gap-8">
            {/* INPUT */}
            <div>
              <p className="micro-label mb-2">
                Input · <span className="mono-num">{N}×{N}</span>
                {pad > 0 && <span className="text-ink-soft"> · pad {pad}</span>}
              </p>
              <svg width={gridPx} height={gridPx} className="block touch-none select-none" style={{ maxWidth: 380 }}>
                {Array.from({ length: N + 2 * pad }).map((_, r) =>
                  Array.from({ length: N + 2 * pad }).map((__, c) => {
                    const or = r - pad;
                    const oc = c - pad;
                    const isPad = or < 0 || or >= N || oc < 0 || oc >= N;
                    const v = isPad ? 0 : image[or][oc];
                    const inWin =
                      inWindow && r >= inWindow.r0 && r < inWindow.r0 + K && c >= inWindow.c0 && c < inWindow.c0 + K;
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
                          opacity={inWindow && !inWin ? 0.5 : 1}
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
                            opacity={inWindow && !inWin ? 0.5 : 1}
                          >
                            {v}
                          </text>
                        )}
                      </g>
                    );
                  })
                )}
                {inWindow && (
                  <rect
                    x={inWindow.c0 * CELL}
                    y={inWindow.r0 * CELL}
                    width={K * CELL}
                    height={K * CELL}
                    fill="none"
                    stroke="#0EA5E9"
                    strokeWidth="2.5"
                    style={{ transition: "x 140ms cubic-bezier(0.16,1,0.3,1), y 140ms cubic-bezier(0.16,1,0.3,1)" }}
                  />
                )}
              </svg>
            </div>

            {/* OUTPUT — clickable / hoverable */}
            <div>
              <p className="micro-label mb-2">
                Feature map · <span className="mono-num">{OUT[0]}×{OUT[1]}</span>
                <span className="ml-1 text-ink-soft">· click a cell</span>
              </p>
              <svg
                width={OUT[1] * outCell}
                height={OUT[0] * outCell}
                className="block"
                style={{ maxWidth: 320 }}
                onPointerLeave={() => setHoverCell(null)}
              >
                {result?.output.map((row, r) =>
                  row.map((v, c) => {
                    const idx = stepIndexOf(r, c);
                    const revealed = !reveal || idx <= step;
                    const isActive = active && active.output_row === r && active.output_col === c;
                    return (
                      <g key={`${r}-${c}`}>
                        <rect
                          x={c * outCell}
                          y={r * outCell}
                          width={outCell}
                          height={outCell}
                          fill={revealed ? diverging(v, outMaxAbs) : "#FAFAFA"}
                          stroke={isActive ? "#0EA5E9" : "#E4E4E7"}
                          strokeWidth={isActive ? 2.5 : 1}
                          strokeDasharray={revealed ? undefined : "2 2"}
                          style={{ cursor: "pointer" }}
                          onPointerEnter={() => setHoverCell({ r, c })}
                          onClick={() => jumpTo(r, c)}
                        />
                        {revealed && outCell >= 22 && (
                          <text
                            x={c * outCell + outCell / 2}
                            y={r * outCell + outCell / 2}
                            dominantBaseline="central"
                            textAnchor="middle"
                            className="mono-num pointer-events-none"
                            fontSize="9"
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

          {/* detailed calculation card */}
          {active && (
            <CalculationCard active={active} kernel={kernel} hovering={!!hoverCell} />
          )}
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
            <InstrumentButton key={name} size="sm" onClick={() => setKernel(KERNEL_PRESETS[name].map((r) => [...r]))}>
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
        <div className="mb-3 flex items-center justify-between">
          <span className="micro-label">Reveal as it scans</span>
          <button
            type="button"
            role="switch"
            aria-checked={reveal}
            onClick={() => setReveal((v) => !v)}
            className={`relative h-5 w-9 border transition-colors duration-150 ${reveal ? "border-cerulean bg-cerulean/15" : "border-line bg-panel"}`}
            style={{ borderRadius: 999 }}
          >
            <span
              className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-all duration-150"
              style={{ left: reveal ? 18 : 2, borderRadius: 999, background: reveal ? "#0EA5E9" : "#A1A1AA" }}
            />
          </button>
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
        <p className="mt-3 text-[11px] leading-relaxed text-ink-soft">
          Hover a feature-map cell to preview its window; click to jump there. Slow the speed to
          watch each multiply-and-sum.
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

// ── detailed calculation card ──
function CalculationCard({ active, kernel, hovering }) {
  const terms = [];
  for (let i = 0; i < active.patch.length; i++)
    for (let j = 0; j < active.patch[i].length; j++)
      terms.push({ w: kernel[i][j], x: active.patch[i][j], p: active.elementwise_products[i][j] });
  const nonzero = terms.filter((t) => t.p !== 0);

  return (
    <div className="w-full max-w-2xl border border-line bg-panel px-5 py-4 shadow-instrument" style={{ borderRadius: 6 }}>
      <div className="mb-3 flex items-center gap-2">
        <span className="micro-label">
          Output cell <span className="mono-num text-ink">({active.output_row}, {active.output_col})</span>
        </span>
        <span className="micro-label text-ink-soft">
          · window origin <span className="mono-num">({active.input_row_start}, {active.input_col_start})</span>
        </span>
        {hovering && <span className="micro-label !text-cerulean">· preview</span>}
      </div>

      {/* grids */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <MiniGrid values={active.patch} label="patch" color={(v) => grayInk(v)} textMid={5} />
        <span className="text-lg text-ink-soft">⊙</span>
        <MiniGrid values={kernel} label="kernel" />
        <span className="text-lg text-ink-soft">=</span>
        <MiniGrid values={active.elementwise_products} label="products" colorProduct />
      </div>

      {/* term-by-term sum */}
      <div className="border-t border-line pt-3">
        <p className="micro-label mb-1.5">Weighted sum</p>
        <div className="flex flex-wrap items-baseline gap-x-1 gap-y-1 leading-relaxed">
          {nonzero.length === 0 ? (
            <span className="mono-num text-[12px] text-ink-soft">all products are zero</span>
          ) : (
            nonzero.map((t, i) => (
              <span key={i} className="mono-num text-[12px]">
                {i > 0 && <span className="text-ink-soft">+ </span>}
                <span className="text-ink-soft">({fmt(t.w)}×{fmt(t.x)}) </span>
                <span style={{ color: signColor(t.p) }}>= {fmt(t.p)}</span>
                {i < nonzero.length - 1 && "  "}
              </span>
            ))
          )}
          <span className="mono-num ml-2 text-[13px]">
            <span className="text-ink-soft">Σ = </span>
            <span className="font-semibold text-cerulean">{active.value.toFixed(3)}</span>
          </span>
        </div>
      </div>
    </div>
  );
}

function MiniGrid({ values, label, color, colorProduct, textMid = 0 }) {
  const cols = values[0]?.length ?? 0;
  const maxAbs = colorProduct ? Math.max(1, ...values.flat().map((v) => Math.abs(v))) : 0;
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
        {values.flat().map((v, i) => {
          const bg = colorProduct ? diverging(v, maxAbs) : color ? color(v) : "transparent";
          const fg = colorProduct ? inkFor(v, maxAbs * 0.6) : color ? inkFor(v, textMid) : "#18181b";
          return (
            <span
              key={i}
              className="mono-num flex h-7 w-7 items-center justify-center text-[10px]"
              style={{
                background: bg,
                color: fg,
                border: colorProduct || color ? "none" : "1px solid #E4E4E7",
                borderRadius: 2,
              }}
            >
              {colorProduct ? fmt(v) : fmt(v)}
            </span>
          );
        })}
      </div>
      <span className="micro-label !text-[8px]">{label}</span>
    </div>
  );
}
