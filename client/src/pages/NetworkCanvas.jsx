import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Workbench from "../components/layout/Workbench";
import NetworkGraph from "../components/canvas/NetworkGraph";
import VCRControls from "../components/canvas/VCRControls";
import SpecEditor from "../components/canvas/SpecEditor";
import TraceView from "../components/canvas/TraceView";
import NodeCard from "../components/canvas/NodeCard";
import KatexBlock from "../components/ui/KatexBlock";
import Odometer from "../components/ui/Odometer";
import InstrumentButton from "../components/ui/InstrumentButton";
import SegmentedControl from "../components/ui/SegmentedControl";
import { useToast } from "../components/ui/Toast";
import { useNetwork, DEFAULT_SPEC } from "../lib/useNetwork";
import { layoutNetwork } from "../lib/graphLayout";
import { api } from "../lib/api";

const nums = (arr) => arr.map((v) => (typeof v === "number" ? v : Number(v) || 0));

const LOSS_LABELS = { mse: "MSE", bce: "BCE", cross_entropy: "CROSS-ENTROPY" };

export default function NetworkCanvas() {
  const toast = useToast();
  const net = useNetwork(toast);

  const [activeSpec, setActiveSpec] = useState(DEFAULT_SPEC);
  const layout = useMemo(
    () => layoutNetwork(activeSpec.inputSize, activeSpec.layers),
    [activeSpec]
  );
  const L = activeSpec.layers.length;

  // ── playback ──
  const [mode, setMode] = useState("forward");
  const [revealed, setRevealedState] = useState(0);
  const [pulsing, setPulsing] = useState(null);
  const [pulseId, setPulseId] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const animRef = useRef(false);
  const timerRef = useRef(null);
  const revealedRef = useRef(0);
  const setRevealed = useCallback((v) => {
    revealedRef.current = v;
    setRevealedState(v);
  }, []);

  // ── selection / trace / extras ──
  const [selection, setSelection] = useState(null); // {type:'node'|'edge', ...}
  const [trace, setTrace] = useState(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [deadXray, setDeadXray] = useState(false);
  const [shockwaveKey, setShockwaveKey] = useState(0);

  // ── training controls ──
  const [lr, setLr] = useState(0.1);
  const [steps, setSteps] = useState(10);
  const [trainingMode, setTrainingMode] = useState("train"); // "train" => dropout active, "eval" => off

  const resetPlayback = useCallback(() => {
    setRevealed(0);
    setPulsing(null);
    setPlaying(false);
    animRef.current = false;
  }, [setRevealed]);

  // reset when a new network arrives
  const netId = net.network?.id;
  useEffect(() => {
    resetPlayback();
    setSelection(null);
    setTrace(null);
    setMode("forward");
  }, [netId, resetPlayback]);

  const dataReady = mode === "forward" ? net.forward != null : net.backward != null;

  const advance = useCallback(() => {
    if (animRef.current || !dataReady) return;
    const r = revealedRef.current;
    if (r >= L) {
      setPlaying(false);
      return;
    }
    const k = mode === "forward" ? r : L - 1 - r;
    animRef.current = true;
    setPulsing(k);
    setPulseId((p) => p + 1);
    setTimeout(() => {
      setPulsing(null);
      setRevealed(r + 1);
      animRef.current = false;
      if (r + 1 >= L) setPlaying(false);
    }, 560);
  }, [L, mode, dataReady, setRevealed]);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(advance, 950 / speed);
      return () => clearInterval(timerRef.current);
    }
  }, [playing, speed, advance]);

  const handleMode = async (m) => {
    if (m === mode) return;
    setMode(m);
    resetPlayback();
    if (m === "backward" && !net.backward && net.network) {
      await net.runBackward(nums(net.inputVec), nums(net.targetVec), net.loss);
    }
  };

  // ── forge / what-if ──
  const handleForge = async () => {
    setSelection(null);
    setTrace(null);
    const snapshot = net.spec;
    const created = await net.forge(snapshot, nums(net.inputVec));
    if (created) {
      setActiveSpec(snapshot);
      resetPlayback();
      setMode("forward");
      setTrainingMode("train"); // fresh forward runs with dropout active
    }
  };

  const handleWhatIf = async () => {
    if (!net.network) return;
    const input = nums(net.inputVec);
    setShockwaveKey((k) => k + 1);
    net.resetHistory(); // new sample/target => a new descent curve
    if (mode === "backward") {
      await net.runBackward(input, nums(net.targetVec), net.loss);
    } else {
      await net.runForward(input, trainingMode === "train");
    }
  };

  // toggle dropout on (train) / off (eval) and re-run the forward pass
  const handleTrainingMode = async (value) => {
    if (value === trainingMode || !net.network) return;
    setTrainingMode(value);
    setSelection(null);
    setTrace(null);
    setMode("forward");
    resetPlayback();
    const fwd = await net.runForward(nums(net.inputVec), value === "train");
    if (fwd) setRevealed(L); // show the whole updated pass at once
  };

  // apply N SGD steps to the live weights, then show the updated gradient field
  const handleTrain = async () => {
    if (!net.network) return;
    const res = await net.runStep(lr, steps, nums(net.inputVec), nums(net.targetVec), net.loss);
    if (res) {
      setSelection(null);
      setTrace(null);
      setMode("backward");
      setRevealed(L); // reveal the whole updated network at once
      setPulsing(null);
      setPlaying(false);
      setShockwaveKey((k) => k + 1);
    }
  };

  // ── selection handlers ──
  const clearSelection = useCallback(() => {
    setSelection(null);
    setTrace(null);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") clearSelection();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearSelection]);

  const handleSelectNode = (n) => {
    setTrace(null);
    setSelection({ type: "node", col: n.col, neuron: n.neuron, layerIndex: n.layerIndex });
  };

  const handleSelectEdge = async (e) => {
    if (!net.network) return;
    setSelection({ type: "edge", edge: e });
    setTrace(null);
    setTraceLoading(true);
    try {
      const t = await api.trace({
        networkId: net.network.id,
        input: nums(net.inputVec),
        target: nums(net.targetVec),
        loss: net.loss,
        layerIndex: e.layerIndex,
        weightRow: e.row,
        weightCol: e.colIdx,
      });
      setTrace(t);
    } catch (err) {
      toast(`TRACE — ${err.message}`);
      setSelection(null);
    } finally {
      setTraceLoading(false);
    }
  };

  // ── equation chip for last revealed forward layer ──
  const chip = useMemo(() => {
    if (mode !== "forward" || revealed === 0 || !net.forward) return null;
    const k = revealed - 1;
    const layer = net.forward.layers[k];
    if (!layer) return null;
    let best = 0;
    layer.post_activation.forEach((a, i) => {
      if (Math.abs(a) > Math.abs(layer.post_activation[best])) best = i;
    });
    const eq = layer.equations?.find((e) => e.neuron === best);
    if (!eq) return null;
    return {
      layerIndex: k,
      neuron: best,
      linear: eq.linear_equation,
      activation: eq.activation_equation,
    };
  }, [mode, revealed, net.forward]);

  const traceEdge = selection?.type === "edge" ? selection.edge : null;

  // ── canvas region ──
  const canvas = (
    <div className="relative h-full w-full">
      {net.online === false && !net.network ? (
        <div className="dot-grid flex h-full items-center justify-center p-6">
          <div className="max-w-md border border-crimson bg-panel shadow-instrument" style={{ borderRadius: 4 }}>
            <div className="flex items-center gap-2 border-b border-line px-5 py-3">
              <span className="h-1.5 w-1.5 rounded-full bg-crimson" />
              <p className="micro-label !text-crimson">Engine offline</p>
            </div>
            <div className="px-5 py-4">
              <p className="mb-3 text-[13px] leading-relaxed text-ink-soft">
                The canvas reads every number from a live PyTorch engine — nothing here is faked.
                Start the backend, and the instrument will connect on its own:
              </p>
              <pre className="mono-num thin-scroll overflow-x-auto border border-line bg-canvas px-3 py-2 text-[11px] leading-relaxed" style={{ borderRadius: 3 }}>
{`cd backend
source venv/bin/activate
uvicorn main:app --port 8000`}
              </pre>
            </div>
          </div>
        </div>
      ) : !net.forward ? (
        <div className="dot-grid flex h-full flex-col items-center justify-center gap-3">
          <div className="flex gap-3">
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-28 w-14 animate-pulse border border-line bg-panel"
                style={{ borderRadius: 4, animationDelay: `${i * 120}ms` }}
              />
            ))}
          </div>
          <p className="micro-label">{net.busy ? "Forging network…" : "Awaiting engine…"}</p>
        </div>
      ) : (
        <NetworkGraph
          layout={layout}
          spec={activeSpec}
          forwardData={net.forward}
          backwardData={net.backward}
          mode={mode}
          revealed={revealed}
          pulsing={pulsing}
          pulseId={pulseId}
          selection={selection}
          traceEdge={traceEdge}
          deadXray={deadXray}
          shockwaveKey={shockwaveKey}
          chip={chip}
          onSelectNode={handleSelectNode}
          onSelectEdge={handleSelectEdge}
          onClearSelection={clearSelection}
        />
      )}

      <VCRControls
        mode={mode}
        onMode={handleMode}
        playing={playing}
        onPlayPause={() => {
          if (!playing && revealed >= L) setRevealed(0);
          setPlaying((p) => !p);
        }}
        onStep={advance}
        onReset={resetPlayback}
        speed={speed}
        onSpeed={setSpeed}
        disabled={!net.forward}
      />

      {/* loss readout — backward mode */}
      <AnimatePresence>
        {mode === "backward" && net.backward && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
            className="glass absolute top-4 right-4 z-20 px-4 py-2.5 max-lg:top-12"
            style={{ borderRadius: 4 }}
          >
            <p className="micro-label mb-0.5">{LOSS_LABELS[net.backward.loss_name] ?? net.backward.loss_name} loss</p>
            <Odometer value={net.backward.loss} decimals={6} className="text-xl font-medium" />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  // ── inspector region ──
  const inspector =
    traceEdge != null ? (
      <TraceView trace={trace} loading={traceLoading} edge={traceEdge} onClose={clearSelection} />
    ) : selection?.type === "node" ? (
      <NodeCard
        node={selection}
        forwardData={net.forward}
        backwardData={net.backward}
        onClose={clearSelection}
      />
    ) : (
      <div className="flex flex-col">
        <SpecEditor
          spec={net.spec}
          setSpec={net.setSpec}
          inputVec={net.inputVec}
          setInputVec={net.setInputVec}
          targetVec={net.targetVec}
          setTargetVec={net.setTargetVec}
          loss={net.loss}
          setLoss={net.setLoss}
          onForge={handleForge}
          onWhatIf={handleWhatIf}
          busy={net.busy}
          hasNetwork={net.network != null}
        />

        {net.network && (
          <div className="border-t border-line p-4">
            <div className="mb-2 flex items-center justify-between">
              <p className="micro-label">Train · gradient descent</p>
              {net.lossHistory.length > 1 && (
                <span className="mono-num text-[10px] text-ink-soft">{net.lossHistory.length - 1} steps</span>
              )}
            </div>
            <div className="mb-2 grid grid-cols-2 gap-2">
              <label className="flex items-center justify-between gap-1">
                <span className="micro-label">lr</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.001"
                  value={lr}
                  onChange={(e) => setLr(Math.max(0.001, Number(e.target.value) || 0.001))}
                  className="mono-num h-7 w-16 border border-line bg-panel px-1.5 text-[11px] focus:border-ink focus:outline-none"
                  style={{ borderRadius: 3 }}
                />
              </label>
              <label className="flex items-center justify-between gap-1">
                <span className="micro-label">steps</span>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={steps}
                  onChange={(e) => setSteps(Math.min(500, Math.max(1, Math.round(Number(e.target.value) || 1))))}
                  className="mono-num h-7 w-16 border border-line bg-panel px-1.5 text-[11px] focus:border-ink focus:outline-none"
                  style={{ borderRadius: 3 }}
                />
              </label>
            </div>
            <InstrumentButton variant="primary" className="w-full" disabled={net.busy} onClick={handleTrain}>
              {net.busy ? "Training…" : `Apply gradient · ${steps} step${steps === 1 ? "" : "s"}`}
            </InstrumentButton>

            {net.lossHistory.length > 1 && (
              <div className="mt-3">
                <div className="mb-0.5 flex items-center justify-between">
                  <span className="micro-label">loss</span>
                  <span className="mono-num text-[11px] text-ink">
                    {net.lossHistory[net.lossHistory.length - 1].toFixed(6)}
                  </span>
                </div>
                <LossSparkline history={net.lossHistory} />
                <div className="flex items-center justify-between">
                  <span className="mono-num text-[9px] text-ink-soft">start {net.lossHistory[0].toFixed(4)}</span>
                  <span className="mono-num text-[9px] text-emerald-sig">
                    −{(((net.lossHistory[0] - net.lossHistory[net.lossHistory.length - 1]) / (net.lossHistory[0] || 1)) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
            <p className="mt-2 text-[10px] leading-relaxed text-ink-soft">
              Applies plain SGD to the live weights, then re-runs backprop — watch the gradient field
              and loss shrink over repeated presses. This mutates the network; Forge to reset.
            </p>
          </div>
        )}

        {net.network && (
          <div className="border-t border-line p-4">
            <div className="mb-2 flex items-center justify-between">
              <p className="micro-label">Dropout · train vs eval</p>
              <SegmentedControl
                size="sm"
                options={[
                  { value: "train", label: "Train" },
                  { value: "eval", label: "Eval" },
                ]}
                value={trainingMode}
                onChange={handleTrainingMode}
              />
            </div>
            {(() => {
              const anyDropout = activeSpec.layers.some((l) => (l.dropout_prob ?? 0) > 0);
              const dropped =
                net.forward?.layers.reduce(
                  (s, l) => s + (l.dropped_mask?.reduce((a, b) => a + b, 0) || 0),
                  0
                ) ?? 0;
              if (!anyDropout) {
                return (
                  <p className="text-[10px] leading-relaxed text-ink-soft">
                    Set a layer's <span className="mono-num text-ink">p</span> above 0 and Forge, then
                    watch units rain out in Train mode.
                  </p>
                );
              }
              return (
                <p className="text-[10px] leading-relaxed text-ink-soft">
                  {trainingMode === "train" ? (
                    <>
                      <span className="mono-num text-crimson">{dropped}</span> unit
                      {dropped === 1 ? "" : "s"} dropped this pass — survivors are scaled up by{" "}
                      <span className="mono-num text-ink">1/(1−p)</span> so the expected signal is
                      preserved. Re-run (edit input · Enter) to resample the mask.
                    </>
                  ) : (
                    <>Eval mode: dropout is off — every unit passes through, the standard behaviour at inference.</>
                  )}
                </p>
              );
            })()}
          </div>
        )}

        {net.forward && (
          <div className="border-t border-line p-4">
            <div className="flex items-center justify-between">
              <p className="micro-label">Dead neuron x-ray</p>
              <button
                type="button"
                role="switch"
                aria-checked={deadXray}
                onClick={() => setDeadXray((d) => !d)}
                className={`relative h-4 w-8 border transition-colors duration-150 ${
                  deadXray ? "border-crimson bg-crimson" : "border-line bg-canvas"
                }`}
                style={{ borderRadius: 2 }}
              >
                <span
                  className="absolute top-0.5 h-2.5 w-2.5 bg-white transition-all duration-150"
                  style={{
                    left: deadXray ? "calc(100% - 12px)" : "2px",
                    borderRadius: 1,
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.15)",
                  }}
                />
              </button>
            </div>
            <p className="mt-1.5 text-[10px] leading-relaxed text-ink-soft">
              Tints ReLU units whose activation is exactly zero for this input — they pass no
              gradient and learn nothing this step.
            </p>
          </div>
        )}

        {mode === "forward" && revealed > 0 && net.forward?.layers[revealed - 1] && (
          <div className="border-t border-line p-4">
            <p className="micro-label mb-2">
              Layer {String(revealed).padStart(2, "0")} equations ·{" "}
              {net.forward.layers[revealed - 1].activation}
            </p>
            <div className="thin-scroll flex max-h-80 flex-col gap-2 overflow-y-auto">
              {net.forward.layers[revealed - 1].equations.map((eq) => (
                <div
                  key={eq.neuron}
                  className="border border-line bg-canvas px-3 py-2"
                  style={{ borderRadius: 3 }}
                >
                  <p className="micro-label mb-1">neuron {eq.neuron}</p>
                  <div className="thin-scroll overflow-x-auto">
                    <KatexBlock latex={eq.linear_equation} />
                  </div>
                  <div className="thin-scroll mt-1 overflow-x-auto">
                    <KatexBlock latex={eq.activation_equation} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {mode === "backward" && net.backward && revealed > 0 && (
          <div className="border-t border-line p-4">
            <p className="micro-label mb-2">Gradient flow · {revealed}/{L} layers back</p>
            <div className="flex flex-col gap-1.5">
              {net.backward.layers
                .slice(L - revealed)
                .reverse()
                .map((l) => {
                  const mx = Math.max(...l.grad_pre_activation.map(Math.abs));
                  const dry = mx < 1e-4;
                  return (
                    <div
                      key={l.layer_index}
                      className="flex items-center justify-between border border-line bg-canvas px-3 py-1.5"
                      style={{ borderRadius: 3 }}
                    >
                      <span className="micro-label">
                        Layer {String(l.layer_index + 1).padStart(2, "0")} · max |∂L/∂z|
                      </span>
                      <span className={`mono-num text-[11px] font-medium ${dry ? "text-crimson" : ""}`}>
                        {dry ? `${mx.toExponential(2)} · DRY` : mx.toFixed(6)}
                      </span>
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        <div className="mt-auto border-t border-line px-4 py-2.5">
          <p className="micro-label">
            {net.network ? (
              <>
                Net <span className="mono-num !text-ink">{net.network.id.slice(0, 8)}</span> ·{" "}
                <span className="mono-num !text-ink">{net.network.paramCount}</span> params · seed{" "}
                <span className="mono-num !text-ink">{activeSpec.seed}</span>
              </>
            ) : (
              "No network forged"
            )}
          </p>
        </div>
      </div>
    );

  return <Workbench canvas={canvas} inspector={inspector} />;
}

/** Stretch-to-fit loss curve. preserveAspectRatio="none" lets it fill the panel
 *  width regardless of how many points the training history holds. */
function LossSparkline({ history }) {
  const W = 300;
  const H = 46;
  const pad = 3;
  if (!history || history.length < 2) return null;
  const min = Math.min(...history);
  const max = Math.max(...history);
  const span = max - min || 1;
  const n = history.length;
  const points = history
    .map((v, i) => {
      const x = pad + (i / (n - 1)) * (W - 2 * pad);
      const y = pad + (1 - (v - min) / span) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className="block w-full border border-line bg-canvas"
      style={{ height: H, borderRadius: 3 }}
    >
      <polyline points={points} fill="none" stroke="#0EA5E9" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
