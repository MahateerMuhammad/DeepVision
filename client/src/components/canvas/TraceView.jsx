import { useState } from "react";
import { motion } from "framer-motion";
import KatexBlock from "../ui/KatexBlock";
import Odometer from "../ui/Odometer";
import InstrumentButton from "../ui/InstrumentButton";

const STEP_LABELS = {
  loss_to_postactivation: "∂ loss / ∂ activation",
  postactivation_to_preactivation: "∂ activation / ∂ pre-activation",
  preactivation_to_param: "∂ pre-activation / ∂ parameter",
};

export default function TraceView({ trace, loading, edge, onClose }) {
  const [showDownstream, setShowDownstream] = useState(false);

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <p className="micro-label !text-cerulean">Chain-rule trace</p>
          <p className="mono-num mt-0.5 text-[11px] text-ink-soft">
            {trace?.param_kind === "bias" ? "b" : "w"}
            <sub>
              {edge.row},{edge.colIdx}
            </sub>{" "}
            · layer {String(edge.layerIndex + 1).padStart(2, "0")}
          </p>
        </div>
        <InstrumentButton size="sm" onClick={onClose}>
          Esc
        </InstrumentButton>
      </div>

      {loading && (
        <div className="flex flex-col gap-2 p-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-10 animate-pulse border border-line bg-canvas" style={{ borderRadius: 3 }} />
          ))}
          <p className="micro-label">Differentiating…</p>
        </div>
      )}

      {trace && !loading && (
        <div className="flex flex-col gap-3 p-4">
          {trace.chain_steps.map((s, i) => (
            <motion.div
              key={s.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.12, duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className="border border-line bg-canvas px-3 py-2"
              style={{ borderRadius: 3 }}
            >
              <p className="micro-label mb-1">
                {String(i + 1).padStart(2, "0")} · {STEP_LABELS[s.name] ?? s.name}
              </p>
              <div className="thin-scroll overflow-x-auto">
                <KatexBlock latex={s.latex} />
              </div>
            </motion.div>
          ))}

          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: trace.chain_steps.length * 0.12 + 0.1, duration: 0.15 }}
            className="border border-ink bg-panel px-3 py-2.5"
            style={{ borderRadius: 3 }}
          >
            <div className="flex items-center justify-between">
              <span className="micro-label">Chain product</span>
              <Odometer value={trace.chain_product} decimals={6} className="text-xs font-medium" />
            </div>
            <div className="mt-1.5 flex items-center justify-between">
              <span className="micro-label">Analytic gradient</span>
              <Odometer value={trace.analytic_gradient} decimals={6} className="text-xs font-medium" />
            </div>
            <div className="mt-2 border-t border-line pt-2">
              {trace.matches_analytic ? (
                <span
                  className="inline-flex items-center gap-1 border border-emerald-sig px-1.5 py-0.5 text-[9px] font-semibold tracking-[0.08em] text-emerald-sig uppercase"
                  style={{ borderRadius: 2 }}
                >
                  <svg width="8" height="8" viewBox="0 0 8 8" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M1 4.5 L3 6.5 L7 1.5" />
                  </svg>
                  Matches autograd
                </span>
              ) : (
                <span
                  className="inline-flex items-center border border-crimson px-1.5 py-0.5 text-[9px] font-semibold tracking-[0.08em] text-crimson uppercase"
                  style={{ borderRadius: 2 }}
                >
                  Mismatch vs autograd
                </span>
              )}
            </div>
          </motion.div>

          {trace.downstream_breakdown && (
            <div className="border border-line" style={{ borderRadius: 3 }}>
              <button
                type="button"
                onClick={() => setShowDownstream((s) => !s)}
                className="micro-label flex w-full items-center justify-between px-3 py-2 hover:bg-black/[0.04]"
              >
                Downstream breakdown · {trace.downstream_breakdown.terms.length} terms
                <span className="mono-num">{showDownstream ? "−" : "+"}</span>
              </button>
              {showDownstream && (
                <div className="flex flex-col gap-2 border-t border-line p-3">
                  <div className="thin-scroll overflow-x-auto">
                    <KatexBlock latex={trace.downstream_breakdown.explanation} />
                  </div>
                  {trace.downstream_breakdown.terms.map((t) => (
                    <div
                      key={t.downstream_neuron}
                      className="thin-scroll overflow-x-auto border-t border-line pt-2"
                    >
                      <p className="micro-label mb-1">via neuron {t.downstream_neuron}</p>
                      <KatexBlock latex={t.latex} />
                    </div>
                  ))}
                  <div className="flex items-center justify-between border-t border-line pt-2">
                    <span className="micro-label">Sum</span>
                    <span className="mono-num text-xs font-medium">
                      {trace.downstream_breakdown.sum.toFixed(6)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          <p className="text-[10px] leading-relaxed text-ink-soft">
            Each factor is one link of the chain rule from the loss down to this parameter. Their
            product must equal what autograd computes and it does, to float precision.
          </p>
        </div>
      )}
    </div>
  );
}
