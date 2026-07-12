export default function StatusLED({ online, className = "" }) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${
          online ? "led-online bg-emerald-sig" : "bg-crimson"
        }`}
      />
      <span className={`micro-label ${online ? "" : "!text-crimson"}`}>
        {online ? "Engine Online" : "Engine Offline"}
      </span>
    </span>
  );
}
