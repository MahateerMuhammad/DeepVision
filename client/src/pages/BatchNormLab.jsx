import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import Workbench from "../components/layout/Workbench";
import InstrumentButton from "../components/ui/InstrumentButton";
import Fader from "../components/ui/Fader";
import { api } from "../lib/api";
import { useToast } from "../components/ui/Toast";
import { useEngine } from "../lib/useEngine";

// Feature 0 lives on a ~10 scale, feature 1 on a ~0.3 scale the whole point:
// BatchNorm brings wildly-different feature scales onto a common μ=0, σ=1 footing.
const PRESETS = {
  "Two scales": [
    [10, 0.1],
    [12, 0.3],
    [8, 0.2],
    [14, 0.5],
    [11, 0.25],
    [9, 0.4],
  ],
  Outlier: [
    [4, 1],
    [5, 1.1],
    [4.5, 0.9],
    [30, 1.05],
    [5.2, 0.95],
  ],
  Tight: [
    [2, -1],
    [2.1, -1.1],
    [1.9, -0.9],
    [2.05, -1.05],
    [1.95, -0.95],
  ],
};

const OUT_DOMAIN = [-4, 4]; // fixed axis for the normalized/output view
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export default function BatchNormLab() {
  const toast = useToast();
  const { online } = useEngine();

  const [batch, setBatch] = useState(PRESETS["Two scales"]);
  const [preset, setPreset] = useState("Two scales");
  const numFeatures = batch[0]?.length ?? 0;
  const [gamma, setGamma] = useState(() => Array(numFeatures).fill(1));
  const [beta, setBeta] = useState(() => Array(numFeatures).fill(0));
  const [result, setResult] = useState(null);
  const [squeezed, setSqueezed] = useState(false);

  // keep gamma/beta length in sync with the feature count
  useEffect(() => {
    setGamma((g) => (g.length === numFeatures ? g : Array.from({ length: numFeatures }, (_, i) => g[i] ?? 1)));
    setBeta((b) => (b.length === numFeatures ? b : Array.from({ length: numFeatures }, (_, i) => b[i] ?? 0)));
  }, [numFeatures]);

  // recompute on any change (debounced)
  useEffect(() => {
    if (!online) return;
    const valid =
      batch.length >= 2 && numFeatures >= 1 && batch.every((r) => r.length === numFeatures);
    if (!valid) return;
    let alive = true;
    const t = setTimeout(() => {
      api
        .batchnorm({ batch: batch.map((r) => r.map(Number)), gamma, beta })
        .then((d) => alive && setResult(d))
        .catch((e) => alive && toast(`BATCHNORM ${e.message}`));
    }, 120);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [batch, gamma, beta, numFeatures, online, toast]);

  // ── batch editing ──
  const setCell = useCallback((r, c, v) => {
    setBatch((b) => b.map((row, i) => (i === r ? row.map((x, j) => (j === c ? v : x)) : row)));
  }, []);
  const addSample = () => setBatch((b) => [...b, Array(numFeatures).fill(0)]);
  const removeSample = (r) => setBatch((b) => (b.length > 2 ? b.filter((_, i) => i !== r) : b));
  const addFeature = () => setBatch((b) => b.map((row) => [...row, 0]));
  const removeFeature = (c) =>
    setBatch((b) => (numFeatures > 1 ? b.map((row) => row.filter((_, j) => j !== c)) : b));
  const applyPreset = (name) => {
    setPreset(name);
    setBatch(PRESETS[name].map((r) => [...r]));
    setGamma(Array(PRESETS[name][0].length).fill(1));
    setBeta(Array(PRESETS[name][0].length).fill(0));
  };

  const canvas = (
    <div className="dot-grid relative h-full w-full overflow-auto">
      {!online ? (
        <div className="flex h-full items-center justify-center">
          <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 8 }}>
            <p className="micro-label !text-crimson mb-1">Engine offline</p>
            <p className="text-xs text-ink-soft">Start the backend on :8000 to normalize a batch.</p>
          </div>
        </div>
      ) : !result ? (
        <div className="flex h-full items-center justify-center">
          <p className="micro-label">Computing statistics…</p>
        </div>
      ) : (
        <div className="mx-auto flex min-h-full max-w-3xl flex-col gap-5 p-8">
          <div className="flex flex-wrap items-center gap-3">
            <InstrumentButton
              variant={squeezed ? "default" : "primary"}
              onClick={() => setSqueezed((s) => !s)}
            >
              {squeezed ? "◂ Show raw batch" : "Normalize · squeeze ▸"}
            </InstrumentButton>
            <p className="text-[14px] text-ink-soft">
              {squeezed ? (
                <>Every feature is now centered at <span className="mono-num text-ink">μ=0</span> with <span className="mono-num text-ink">σ=1</span> then rescaled by <span className="mono-num text-cerulean">γ</span>, <span className="mono-num text-cerulean">β</span>.</>
              ) : (
                <>Raw features sit on wildly different scales. Hit normalize to watch them collapse onto one footing.</>
              )}
            </p>
          </div>

          {Array.from({ length: result.num_features }, (_, f) => (
            <FeatureLane
              key={f}
              index={f}
              raw={result.input.map((row) => row[f])}
              out={result.output.map((row) => row[f])}
              rawMu={result.mean[f]}
              rawSigma={Math.sqrt(result.variance[f])}
              outMu={result.output_mean[f]}
              outSigma={result.output_std[f]}
              gamma={result.gamma[f]}
              beta={result.beta[f]}
              squeezed={squeezed}
            />
          ))}
        </div>
      )}
    </div>
  );

  const inspector = (
    <div className="flex flex-col">
      {/* readout */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Per-feature statistics</p>
        {result ? (
          <div className="flex flex-col gap-1.5">
            {Array.from({ length: result.num_features }, (_, f) => (
              <div key={f} className="border border-line bg-canvas px-2.5 py-1.5" style={{ borderRadius: 8 }}>
                <div className="mb-0.5 flex items-center justify-between">
                  <span className="micro-label">feat {f}</span>
                  <span className="mono-num text-[10px] text-ink-soft">raw → out</span>
                </div>
                <div className="mono-num flex justify-between text-[13px]">
                  <span className="text-ink-soft">μ</span>
                  <span className="text-ink">{result.mean[f].toFixed(2)}</span>
                  <span className="text-ink-soft">→</span>
                  <span className="text-cerulean">{result.output_mean[f].toFixed(2)}</span>
                </div>
                <div className="mono-num flex justify-between text-[13px]">
                  <span className="text-ink-soft">σ</span>
                  <span className="text-ink">{Math.sqrt(result.variance[f]).toFixed(2)}</span>
                  <span className="text-ink-soft">→</span>
                  <span className="text-cerulean">{result.output_std[f].toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="micro-label">—</p>
        )}
      </div>

      {/* gamma / beta */}
      <div className="border-b border-line p-4">
        <p className="micro-label mb-1">Learnable scale &amp; shift</p>
        <p className="mb-3 text-[13px] leading-relaxed text-ink-soft">
          After normalizing, BatchNorm lets the network re-stretch (<span className="mono-num text-cerulean">γ</span>)
          and re-center (<span className="mono-num text-cerulean">β</span>) each feature so it can undo the
          normalization if that helps.
        </p>
        <div className="flex flex-col gap-3">
          {Array.from({ length: numFeatures }, (_, f) => (
            <div key={f}>
              <div className="mb-1 flex items-center justify-between">
                <span className="micro-label">feat {f}</span>
                <span className="mono-num text-[10px] text-ink-soft">
                  γ {(gamma[f] ?? 1).toFixed(2)} · β {(beta[f] ?? 0).toFixed(2)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="mono-num w-3 text-[10px] text-ink-soft">γ</span>
                <Fader value={gamma[f] ?? 1} min={0} max={2.5} step={0.05} accent="#0EA5E9"
                  onChange={(v) => setGamma((g) => g.map((x, i) => (i === f ? v : x)))} className="flex-1" />
              </div>
              <div className="mt-1 flex items-center gap-2">
                <span className="mono-num w-3 text-[10px] text-ink-soft">β</span>
                <Fader value={beta[f] ?? 0} min={-2} max={2} step={0.05} accent="#0EA5E9"
                  onChange={(v) => setBeta((b) => b.map((x, i) => (i === f ? v : x)))} className="flex-1" />
              </div>
            </div>
          ))}
        </div>
        <InstrumentButton
          size="sm"
          className="mt-3 w-full"
          onClick={() => {
            setGamma(Array(numFeatures).fill(1));
            setBeta(Array(numFeatures).fill(0));
          }}
        >
          Reset γ=1 · β=0
        </InstrumentButton>
      </div>

      {/* batch editor */}
      <div className="p-4">
        <div className="mb-2 flex items-center justify-between">
          <p className="micro-label">Batch · {batch.length}×{numFeatures}</p>
          <div className="flex flex-wrap gap-1">
            {Object.keys(PRESETS).map((n) => (
              <InstrumentButton key={n} size="sm" active={preset === n} onClick={() => applyPreset(n)}>
                {n}
              </InstrumentButton>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="border-collapse">
            <thead>
              <tr>
                <th className="w-6" />
                {Array.from({ length: numFeatures }, (_, c) => (
                  <th key={c} className="px-1 pb-1">
                    <button
                      type="button"
                      onClick={() => removeFeature(c)}
                      disabled={numFeatures <= 1}
                      title="remove feature"
                      className="micro-label hover:text-crimson disabled:opacity-30"
                    >
                      f{c} ✕
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {batch.map((row, r) => (
                <tr key={r}>
                  <td className="pr-1">
                    <button
                      type="button"
                      onClick={() => removeSample(r)}
                      disabled={batch.length <= 2}
                      title="remove sample"
                      className="text-ink-soft hover:text-crimson disabled:opacity-30"
                    >
                      ✕
                    </button>
                  </td>
                  {row.map((v, c) => (
                    <td key={c} className="p-0.5">
                      <input
                        type="number"
                        step="0.1"
                        value={v}
                        onChange={(e) => setCell(r, c, e.target.value === "" ? "" : Number(e.target.value))}
                        onBlur={(e) => e.target.value === "" && setCell(r, c, 0)}
                        className="mono-num h-8 w-16 border border-line bg-panel px-1.5 text-[14px] focus:border-ink focus:outline-none"
                        style={{ borderRadius: 8 }}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-2 flex gap-1.5">
          <InstrumentButton size="sm" onClick={addSample}>+ Sample</InstrumentButton>
          <InstrumentButton size="sm" onClick={addFeature}>+ Feature</InstrumentButton>
        </div>
        <p className="mt-3 text-[13px] leading-relaxed text-ink-soft">
          Statistics are per-feature across the batch which is why BatchNorm needs ≥2 samples and behaves
          differently at inference, where running averages replace the batch's own μ/σ.
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

// ── one feature's number line: dots animate from raw positions to normalized ──
function FeatureLane({ index, raw, out, rawMu, rawSigma, outMu, outSigma, gamma, beta, squeezed }) {
  const W = 560;
  const H = 96;
  const pad = 28;
  const midY = 44;
  const laneRef = useRef({ min: 0, max: 1 });

  // raw domain padded a touch so points don't touch the edges
  const rmin = Math.min(...raw);
  const rmax = Math.max(...raw);
  const rspan = rmax - rmin || 1;
  laneRef.current = { min: rmin - rspan * 0.15, max: rmax + rspan * 0.15 };

  const rawX = (v) => pad + ((v - laneRef.current.min) / (laneRef.current.max - laneRef.current.min)) * (W - 2 * pad);
  const outX = (v) => pad + ((clamp(v, OUT_DOMAIN[0], OUT_DOMAIN[1]) - OUT_DOMAIN[0]) / (OUT_DOMAIN[1] - OUT_DOMAIN[0])) * (W - 2 * pad);

  const muX = squeezed ? outX(outMu) : rawX(rawMu);
  const sigLo = squeezed ? outX(outMu - outSigma) : rawX(rawMu - rawSigma);
  const sigHi = squeezed ? outX(outMu + outSigma) : rawX(rawMu + rawSigma);

  const domain = squeezed ? OUT_DOMAIN : [laneRef.current.min, laneRef.current.max];

  return (
    <div className="border border-line bg-panel p-3" style={{ borderRadius: 8 }}>
      <div className="mb-1 flex items-center justify-between">
        <span className="micro-label">feature {index}</span>
        <span className="mono-num text-[10px] text-ink-soft">
          {squeezed ? <>out μ {outMu.toFixed(2)} · σ {outSigma.toFixed(2)} · γ {gamma.toFixed(2)} β {beta.toFixed(2)}</> : <>raw μ {rawMu.toFixed(2)} · σ {rawSigma.toFixed(2)}</>}
        </span>
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        {/* baseline */}
        <line x1={pad} y1={midY} x2={W - pad} y2={midY} stroke="#E4E4E7" strokeWidth="1" />
        {/* ±σ band */}
        <rect x={Math.min(sigLo, sigHi)} y={midY - 16} width={Math.abs(sigHi - sigLo)} height="32" fill="#0EA5E9" opacity="0.07" style={{ transition: "all 700ms cubic-bezier(0.16,1,0.3,1)" }} />
        {/* μ marker */}
        <line x1={muX} y1={midY - 20} x2={muX} y2={midY + 20} stroke="#0EA5E9" strokeWidth="1.5" strokeDasharray="3 3" style={{ transition: "all 700ms cubic-bezier(0.16,1,0.3,1)" }} />
        {/* dots */}
        {raw.map((v, i) => (
          <motion.circle
            key={i}
            cy={midY}
            r={4.5}
            fill="#0EA5E9"
            fillOpacity={0.55}
            stroke="#0EA5E9"
            initial={false}
            animate={{ cx: squeezed ? outX(out[i]) : rawX(v) }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          />
        ))}
        {/* axis endpoints */}
        <text x={pad} y={H - 6} textAnchor="middle" className="mono-num" style={{ fontSize: 9, fill: "#A1A1AA" }}>
          {domain[0].toFixed(squeezed ? 0 : 1)}
        </text>
        <text x={W - pad} y={H - 6} textAnchor="middle" className="mono-num" style={{ fontSize: 9, fill: "#A1A1AA" }}>
          {domain[1].toFixed(squeezed ? 0 : 1)}
        </text>
        {squeezed && (
          <text x={outX(0)} y={H - 6} textAnchor="middle" className="mono-num" style={{ fontSize: 9, fill: "#0EA5E9" }}>
            0
          </text>
        )}
      </svg>
    </div>
  );
}
