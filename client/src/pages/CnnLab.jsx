import { useState } from "react";
import { motion } from "framer-motion";
import Workbench from "../components/layout/Workbench";
import FilterFactory from "../components/cnn/FilterFactory";
import ConvExplorer from "../components/cnn/ConvExplorer";

const MODES = [
  { value: "filter", label: "Filter Factory" },
  { value: "conv", label: "Conv Explorer" },
  { value: "receptive", label: "Receptive Field" },
  { value: "saliency", label: "Saliency" },
];

const COMING = {
  receptive: {
    title: "Receptive Field",
    body: "Cast a cone of light backward from any deep activation to the exact input patch that produced it.",
    endpoints: ["POST /cnn/receptive-field"],
  },
  saliency: {
    title: "Saliency",
    body: "Ask the network why it predicted a class — overlay an input-gradient heat map on the source image.",
    endpoints: ["POST /cnn/saliency"],
  },
};

export default function CnnLab() {
  const [mode, setMode] = useState("filter");
  const tabBar = (
    <div>
      <p className="micro-label mb-2">Instrument</p>
      <div className="grid grid-cols-2 gap-1.5" role="tablist">
        {MODES.map((m) => {
          const selected = m.value === mode;
          return (
            <button
              key={m.value}
              type="button"
              role="tab"
              aria-selected={selected}
              onClick={() => setMode(m.value)}
              className={`flex h-8 items-center justify-center whitespace-nowrap border px-2 text-[10px] font-semibold uppercase tracking-[0.05em] transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-cerulean ${
                selected
                  ? "border-ink bg-ink text-white"
                  : "border-line bg-panel text-ink-soft hover:bg-black/[0.04] hover:text-ink"
              }`}
              style={{ borderRadius: 3 }}
            >
              {m.label}
            </button>
          );
        })}
      </div>
    </div>
  );

  if (mode === "filter") return <FilterFactory tabBar={tabBar} />;
  if (mode === "conv") return <ConvExplorer tabBar={tabBar} />;

  const info = COMING[mode];
  const canvas = (
    <div className="dot-grid flex h-full items-center justify-center p-6">
      <motion.div
        key={mode}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="max-w-sm border border-line bg-panel px-6 py-5 text-center shadow-instrument"
        style={{ borderRadius: 4 }}
      >
        <div className="mb-2 flex items-center justify-center gap-2">
          <span className="inline-block h-2 w-2 border border-ink-soft" aria-hidden />
          <h2 className="text-sm font-bold uppercase tracking-[0.08em]">{info.title}</h2>
        </div>
        <p className="mb-4 text-[13px] leading-relaxed text-ink-soft">{info.body}</p>
        <div className="flex flex-col gap-1.5">
          {info.endpoints.map((e) => (
            <div key={e} className="flex items-center justify-center gap-2">
              <span className="led-online h-1.5 w-1.5 rounded-full bg-emerald-sig" />
              <span className="mono-num text-[11px]">{e}</span>
              <span className="micro-label !text-emerald-sig">live</span>
            </div>
          ))}
        </div>
        <p className="micro-label mt-4">Next instrument · optics inbound</p>
      </motion.div>
    </div>
  );
  const inspector = <div className="p-4">{tabBar}</div>;
  return <Workbench canvas={canvas} inspector={inspector} />;
}
