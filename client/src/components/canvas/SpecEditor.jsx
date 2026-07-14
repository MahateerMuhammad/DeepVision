import InstrumentButton from "../ui/InstrumentButton";
import SegmentedControl from "../ui/SegmentedControl";
import Stepper from "../ui/Stepper";

const ACTIVATION_OPTS = ["relu", "sigmoid", "tanh", "linear", "leaky_relu"];
const LOSS_OPTS = [
  { value: "mse", label: "MSE" },
  { value: "bce", label: "BCE" },
  { value: "cross_entropy", label: "CE" },
];

function resize(vec, n, fill = 0) {
  return Array.from({ length: n }, (_, i) => vec[i] ?? fill);
}

function VectorFields({ label, values, onChange, hot = false, onCommit }) {
  return (
    <div>
      <p className="micro-label mb-1.5">
        {label}
        {hot && <span className="ml-1.5 text-cerulean">· live</span>}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {values.map((v, i) => (
          <input
            key={i}
            type="number"
            step="0.1"
            value={v}
            onChange={(e) => {
              const next = [...values];
              next[i] = e.target.value === "" ? "" : Number(e.target.value);
              onChange(next);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") onCommit?.();
            }}
            onBlur={() => {
              if (values[i] === "") {
                const next = [...values];
                next[i] = 0;
                onChange(next);
              }
            }}
            className={`mono-num h-7 w-16 border bg-panel px-1.5 text-[11px] transition-colors duration-150 focus:outline-none ${
              hot ? "border-cerulean/60 focus:border-cerulean" : "border-line focus:border-ink"
            }`}
            style={{ borderRadius: 3 }}
          />
        ))}
      </div>
    </div>
  );
}

export default function SpecEditor({
  spec,
  setSpec,
  inputVec,
  setInputVec,
  targetVec,
  setTargetVec,
  loss,
  setLoss,
  onForge,
  onWhatIf,
  busy,
  hasNetwork,
}) {
  const setInputSize = (n) => {
    setSpec({ ...spec, inputSize: n });
    setInputVec(resize(inputVec, n));
  };
  const setLayer = (i, patch) => {
    const layers = spec.layers.map((l, j) => (j === i ? { ...l, ...patch } : l));
    setSpec({ ...spec, layers });
    if (i === spec.layers.length - 1 && patch.out_features) {
      setTargetVec(resize(targetVec, patch.out_features));
    }
  };
  const removeLayer = (i) => {
    if (spec.layers.length <= 1) return;
    const layers = spec.layers.filter((_, j) => j !== i);
    setSpec({ ...spec, layers });
    setTargetVec(resize(targetVec, layers[layers.length - 1].out_features));
  };
  const addLayer = () => {
    const layers = [...spec.layers, { out_features: 3, activation: "relu", dropout_prob: 0 }];
    setSpec({ ...spec, layers });
    setTargetVec(resize(targetVec, 3));
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex items-center justify-between">
        <p className="micro-label">Input size</p>
        <Stepper value={spec.inputSize} onChange={setInputSize} min={1} max={8} />
      </div>

      <div>
        <p className="micro-label mb-1.5">Layers</p>
        <div className="flex flex-col gap-1.5">
          {spec.layers.map((l, i) => (
            <div
              key={i}
              className="flex items-center gap-1.5 border border-line bg-canvas p-1.5"
              style={{ borderRadius: 3 }}
            >
              <Stepper
                value={l.out_features}
                onChange={(v) => setLayer(i, { out_features: v })}
                min={1}
                max={12}
              />
              <select
                value={l.activation}
                onChange={(e) => setLayer(i, { activation: e.target.value })}
                className="mono-num h-7 min-w-0 flex-1 border border-line bg-panel px-1 text-[11px] focus:border-ink focus:outline-none"
                style={{ borderRadius: 3 }}
              >
                {ACTIVATION_OPTS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
              <label className="flex shrink-0 items-center gap-1" title="dropout probability (0–0.9)">
                <span className="micro-label">p</span>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="0.9"
                  value={l.dropout_prob ?? 0}
                  onChange={(e) =>
                    setLayer(i, {
                      dropout_prob: Math.min(0.9, Math.max(0, Number(e.target.value) || 0)),
                    })
                  }
                  className={`mono-num h-7 w-12 border bg-panel px-1 text-[11px] focus:outline-none ${
                    (l.dropout_prob ?? 0) > 0 ? "border-cerulean/60 focus:border-cerulean" : "border-line focus:border-ink"
                  }`}
                  style={{ borderRadius: 3 }}
                />
              </label>
              <button
                type="button"
                aria-label={`remove layer ${i + 1}`}
                onClick={() => removeLayer(i)}
                disabled={spec.layers.length <= 1}
                className="flex h-7 w-7 shrink-0 items-center justify-center text-ink-soft transition-colors duration-150 hover:text-crimson disabled:opacity-30"
              >
                <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 2 L9 9 M9 2 L2 9" />
                </svg>
              </button>
            </div>
          ))}
        </div>
        <InstrumentButton size="sm" className="mt-1.5 w-full" onClick={addLayer}>
          + Add layer
        </InstrumentButton>
      </div>

      <div className="flex items-center justify-between">
        <p className="micro-label">Seed</p>
        <input
          type="number"
          value={spec.seed}
          onChange={(e) => setSpec({ ...spec, seed: Number(e.target.value) })}
          className="mono-num h-7 w-20 border border-line bg-panel px-2 text-right text-[11px] focus:border-ink focus:outline-none"
          style={{ borderRadius: 3 }}
        />
      </div>

      <VectorFields
        label="Input vector"
        values={inputVec}
        onChange={setInputVec}
        hot={hasNetwork}
        onCommit={onWhatIf}
      />
      <VectorFields label="Target vector" values={targetVec} onChange={setTargetVec} />

      <div className="flex items-center justify-between">
        <p className="micro-label">Loss</p>
        <SegmentedControl size="sm" options={LOSS_OPTS} value={loss} onChange={setLoss} />
      </div>

      <InstrumentButton variant="primary" className="w-full" onClick={onForge} disabled={busy}>
        {busy ? "Forging…" : "Forge network"}
      </InstrumentButton>
      {hasNetwork && (
        <p className="text-[10px] leading-relaxed text-ink-soft">
          Input fields are live — edit a value and press Enter to re-run the pass through the
          existing weights (what-if). Forge rebuilds with fresh weights.
        </p>
      )}
    </div>
  );
}
