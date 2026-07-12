import { motion } from "framer-motion";
import Panel from "../components/ui/Panel";
import { useEngine } from "../lib/useEngine";

const ENDPOINTS = [
  { path: "POST /cnn/sliding-kernel", desc: "Kernel convolution, position by position" },
  { path: "POST /cnn/forward", desc: "Full feature-map forward pass" },
  { path: "POST /cnn/receptive-field", desc: "Pixel provenance for any activation" },
  { path: "POST /cnn/saliency", desc: "Input-gradient saliency maps" },
];

export default function CnnLab() {
  const { online } = useEngine();
  return (
    <main className="dot-grid relative flex min-h-0 flex-1 items-center justify-center overflow-y-auto p-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-lg"
      >
        <Panel
          title="Module C — Convolution Instrument"
          right={<span className="micro-label !text-crimson">Phase 2</span>}
        >
          <div className="border-b border-line px-5 py-6">
            <div className="mb-3 flex items-center gap-2">
              <span className="inline-block h-2 w-2 border border-ink-soft" aria-hidden />
              <h1 className="text-sm font-bold tracking-[0.08em] uppercase">
                Instrument Offline
              </h1>
            </div>
            <p className="text-[13px] leading-relaxed text-ink-soft">
              The convolution optics for this bay are scheduled for Phase 2 — sliding-kernel
              scanning, feature-map towers, receptive-field tracebacks and saliency imaging.
              The engine already answers on every required channel below.
            </p>
          </div>
          <ul>
            {ENDPOINTS.map((e, i) => (
              <li
                key={e.path}
                className={`flex items-center gap-3 px-5 py-3 ${i > 0 ? "border-t border-line" : ""}`}
              >
                <span
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                    online ? "led-online bg-emerald-sig" : "bg-crimson"
                  }`}
                />
                <div className="min-w-0 flex-1">
                  <p className="mono-num truncate text-xs font-medium">{e.path}</p>
                  <p className="text-[11px] text-ink-soft">{e.desc}</p>
                </div>
                <span className={`micro-label ${online ? "!text-emerald-sig" : "!text-crimson"}`}>
                  {online ? "Endpoint live" : "Unreachable"}
                </span>
              </li>
            ))}
          </ul>
          <footer className="border-t border-line px-5 py-2.5">
            <p className="micro-label">DeepVision · Bay C · awaiting optics</p>
          </footer>
        </Panel>
      </motion.div>
    </main>
  );
}
