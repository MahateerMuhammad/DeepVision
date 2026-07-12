/**
 * Sharp rectangular instrument button.
 * variants: default (white), primary (black), danger (crimson outline)
 */
export default function InstrumentButton({
  children,
  variant = "default",
  active = false,
  size = "md",
  className = "",
  title,
  disabled,
  ...props
}) {
  const base =
    "inline-flex items-center justify-center gap-1.5 border font-semibold uppercase tracking-[0.08em] select-none " +
    "transition-[background-color,color,border-color,transform] duration-150 " +
    "active:translate-y-px focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-cerulean " +
    "disabled:pointer-events-none disabled:opacity-40";
  const sizes = {
    sm: "h-6 px-2 text-[10px]",
    md: "h-8 px-3 text-[11px]",
    icon: "h-8 w-8 text-[11px]",
  };
  const looks = {
    default: active
      ? "border-ink bg-ink text-white"
      : "border-line bg-panel text-ink hover:bg-black/[0.04]",
    primary: "border-ink bg-ink text-white hover:bg-zinc-800",
    danger: "border-crimson bg-panel text-crimson hover:bg-crimson/5",
  };
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      className={`${base} ${sizes[size]} ${looks[variant]} ${className}`}
      style={{ borderRadius: 3, transitionTimingFunction: "cubic-bezier(0.16,1,0.3,1)" }}
      {...props}
    >
      {children}
    </button>
  );
}
