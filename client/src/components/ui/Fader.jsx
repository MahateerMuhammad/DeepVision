import { useCallback, useRef } from "react";

/**
 * Horizontal fader with tick marks and an optional red zone.
 * `redZoneFrom` is a value (same units as min/max); the track past it turns crimson.
 */
export default function Fader({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  redZoneFrom = null,
  disabled = false,
  ticks = 9,
  className = "",
  accent = "#18181b",
}) {
  const trackRef = useRef(null);
  const frac = (value - min) / (max - min);
  const redFrac = redZoneFrom != null ? (redZoneFrom - min) / (max - min) : null;
  const inRed = redZoneFrom != null && value >= redZoneFrom;

  const setFromClientX = useCallback(
    (clientX) => {
      const rect = trackRef.current.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      const raw = min + f * (max - min);
      const snapped = Math.round(raw / step) * step;
      onChange(Number(Math.min(max, Math.max(min, snapped)).toFixed(6)));
    },
    [min, max, step, onChange]
  );

  const onPointerDown = (e) => {
    if (disabled) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers can't be captured */
    }
    setFromClientX(e.clientX);
  };
  const onPointerMove = (e) => {
    if (disabled || e.buttons !== 1) return;
    setFromClientX(e.clientX);
  };
  const onKeyDown = (e) => {
    if (disabled) return;
    const delta = e.key === "ArrowRight" || e.key === "ArrowUp" ? step : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -step : 0;
    if (delta !== 0) {
      e.preventDefault();
      onChange(Number(Math.min(max, Math.max(min, value + delta)).toFixed(6)));
    }
  };

  return (
    <div
      ref={trackRef}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onKeyDown={onKeyDown}
      className={`relative h-6 cursor-ew-resize touch-none select-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cerulean ${
        disabled ? "pointer-events-none opacity-40" : ""
      } ${className}`}
    >
      {/* ticks */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-between">
        {Array.from({ length: ticks }, (_, i) => (
          <span key={i} className="h-1.5 w-px bg-line" />
        ))}
      </div>
      {/* track */}
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line" />
      {redFrac != null && (
        <div
          className="pointer-events-none absolute top-1/2 h-[3px] -translate-y-1/2 bg-crimson/25"
          style={{ left: `${redFrac * 100}%`, right: 0 }}
        />
      )}
      {/* fill */}
      <div
        className="pointer-events-none absolute top-1/2 h-[2px] -translate-y-1/2"
        style={{ left: 0, width: `${frac * 100}%`, background: inRed ? "#ef4444" : accent }}
      />
      {/* thumb */}
      <div
        className="pointer-events-none absolute top-1/2 h-4 w-2 -translate-x-1/2 -translate-y-1/2 border bg-panel shadow-instrument"
        style={{
          left: `${frac * 100}%`,
          borderColor: inRed ? "#ef4444" : "#18181b",
          borderRadius: 2,
        }}
      />
    </div>
  );
}
