export default function SegmentedControl({ options, value, onChange, size = "md", className = "" }) {
  const h = size === "sm" ? "h-7 text-[11px]" : "h-9 text-[12px]";
  return (
    <div
      className={`inline-flex overflow-hidden border border-line bg-panel ${className}`}
      role="tablist"
      style={{ borderRadius: 8 }}
    >
      {options.map((opt, i) => {
        const selected = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(opt.value)}
            className={`${h} px-2.5 font-semibold uppercase tracking-[0.08em] transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-cerulean ${
              i > 0 ? "border-l border-line" : ""
            } ${selected ? "bg-ink text-white" : "text-ink-soft hover:bg-black/[0.04] hover:text-ink"}`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
