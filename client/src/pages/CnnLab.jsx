import { useState } from "react";
import FilterFactory from "../components/cnn/FilterFactory";
import ConvExplorer from "../components/cnn/ConvExplorer";
import ReceptiveField from "../components/cnn/ReceptiveField";
import SaliencyLab from "../components/cnn/SaliencyLab";

const MODES = [
  { value: "filter", label: "Filter Factory" },
  { value: "conv", label: "Conv Explorer" },
  { value: "receptive", label: "Receptive Field" },
  { value: "saliency", label: "Saliency" },
];

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

  if (mode === "conv") return <ConvExplorer tabBar={tabBar} />;
  if (mode === "receptive") return <ReceptiveField tabBar={tabBar} />;
  if (mode === "saliency") return <SaliencyLab tabBar={tabBar} />;
  return <FilterFactory tabBar={tabBar} />;
}
