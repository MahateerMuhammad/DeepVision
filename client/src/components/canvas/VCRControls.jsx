import SegmentedControl from "../ui/SegmentedControl";
import InstrumentButton from "../ui/InstrumentButton";

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.5 };

const Icons = {
  reset: (
    <svg width="14" height="14" viewBox="0 0 14 14" {...S}>
      <path d="M3 2 V12 M11.5 2.5 L5 7 L11.5 11.5 Z" />
    </svg>
  ),
  play: (
    <svg width="14" height="14" viewBox="0 0 14 14" {...S}>
      <path d="M3.5 2.5 L11.5 7 L3.5 11.5 Z" />
    </svg>
  ),
  pause: (
    <svg width="14" height="14" viewBox="0 0 14 14" {...S}>
      <path d="M4.5 2.5 V11.5 M9.5 2.5 V11.5" />
    </svg>
  ),
  step: (
    <svg width="14" height="14" viewBox="0 0 14 14" {...S}>
      <path d="M2.5 2.5 L9 7 L2.5 11.5 Z M11.5 2 V12" />
    </svg>
  ),
};

export default function VCRControls({
  mode,
  onMode,
  playing,
  onPlayPause,
  onStep,
  onReset,
  speed,
  onSpeed,
  disabled,
}) {
  return (
    <div className="glass absolute bottom-5 left-5 z-20 flex items-center gap-2 p-2" style={{ borderRadius: 5 }}>
      <InstrumentButton size="icon" title="Reset" onClick={onReset} disabled={disabled}>
        {Icons.reset}
      </InstrumentButton>
      <InstrumentButton
        size="icon"
        title={playing ? "Pause" : "Play"}
        active={playing}
        onClick={onPlayPause}
        disabled={disabled}
      >
        {playing ? Icons.pause : Icons.play}
      </InstrumentButton>
      <InstrumentButton size="icon" title="Step" onClick={onStep} disabled={disabled}>
        {Icons.step}
      </InstrumentButton>
      <span className="mx-0.5 h-5 w-px bg-line" aria-hidden />
      <SegmentedControl
        size="sm"
        options={[
          { value: 0.5, label: "0.5×" },
          { value: 1, label: "1×" },
          { value: 2, label: "2×" },
        ]}
        value={speed}
        onChange={onSpeed}
      />
      <span className="mx-0.5 h-5 w-px bg-line" aria-hidden />
      <SegmentedControl
        size="sm"
        options={[
          { value: "forward", label: "Forward" },
          { value: "backward", label: "Backward" },
        ]}
        value={mode}
        onChange={onMode}
      />
    </div>
  );
}
