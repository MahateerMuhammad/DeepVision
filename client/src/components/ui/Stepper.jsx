export default function Stepper({ value, onChange, min = 1, max = 12, className = "" }) {
  const btn =
    "flex h-8 w-8 items-center justify-center text-base text-ink-soft transition-colors duration-150 " +
    "hover:bg-black/[0.04] hover:text-ink active:translate-y-px disabled:pointer-events-none disabled:opacity-30 " +
    "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-cerulean";
  return (
    <div
      className={`inline-flex items-stretch overflow-hidden border border-line bg-panel ${className}`}
      style={{ borderRadius: 8 }}
    >
      <button
        type="button"
        aria-label="decrement"
        className={btn}
        disabled={value <= min}
        onClick={() => onChange(Math.max(min, value - 1))}
      >
        −
      </button>
      <span className="mono-num flex w-9 items-center justify-center border-x border-line text-sm font-medium">
        {value}
      </span>
      <button
        type="button"
        aria-label="increment"
        className={btn}
        disabled={value >= max}
        onClick={() => onChange(Math.min(max, value + 1))}
      >
        +
      </button>
    </div>
  );
}
