import { useLocation } from "react-router-dom";
import StatusLED from "../ui/StatusLED";
import { useEngine } from "../../lib/useEngine";

const TITLES = {
  "/": "MODULE 0·B·D NETWORK CANVAS",
  "/activations": "MODULE A ACTIVATION LAB",
  "/optimizers": "MODULE E OPTIMIZER ARENA",
  "/cnn": "MODULE C CNN LAB",
  "/batchnorm": "MODULE F BATCHNORM TRACKER",
};

export default function TopBar() {
  const { pathname } = useLocation();
  const { online, paramCount } = useEngine();
  return (
    <header className="z-40 flex h-12 shrink-0 items-center gap-4 border-b border-line bg-panel px-4">
      <span className="text-[13px] font-bold tracking-[0.14em] select-none">
        DEEP<span className="text-cerulean">VISION</span>
      </span>
      <span className="h-4 w-px bg-line" aria-hidden />
      <span className="micro-label max-sm:hidden">{TITLES[pathname] ?? "INSTRUMENT"}</span>
      <span className="flex-1" />
      {paramCount != null && (
        <span className="micro-label max-sm:hidden">
          PARAMS <span className="mono-num text-[11px] font-medium !text-ink">{paramCount}</span>
        </span>
      )}
      <StatusLED online={online === true} />
    </header>
  );
}
