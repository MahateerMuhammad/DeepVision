import { motion } from "framer-motion";

const DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const EASE = [0.16, 1, 0.3, 1];

function DigitColumn({ digit, duration }) {
  return (
    <span
      className="inline-block overflow-hidden align-baseline"
      style={{ height: "1em", lineHeight: 1 }}
    >
      <motion.span
        className="block"
        animate={{ y: `-${digit}em` }}
        transition={{ duration, ease: EASE }}
      >
        {DIGITS.map((d) => (
          <span key={d} className="block" style={{ height: "1em", lineHeight: 1 }}>
            {d}
          </span>
        ))}
      </motion.span>
    </span>
  );
}

/**
 * Rolling-digit numeric readout. Digits roll vertically on change;
 * signs / separators swap in place. Always IBM Plex Mono tabular.
 */
export default function Odometer({ value, decimals = 4, className = "", duration = 0.45 }) {
  if (value == null || Number.isNaN(value)) {
    return <span className={`mono-num ${className}`}>—</span>;
  }
  const text = value.toFixed(decimals);
  const chars = text.split("");
  return (
    <span
      className={`mono-num inline-flex items-baseline whitespace-pre ${className}`}
      style={{ lineHeight: 1 }}
      aria-label={text}
    >
      {chars.map((ch, i) =>
        /\d/.test(ch) ? (
          <DigitColumn key={`${i}-${chars.length}`} digit={Number(ch)} duration={duration} />
        ) : (
          <span key={`${i}-${chars.length}`} style={{ height: "1em", lineHeight: 1 }}>
            {ch}
          </span>
        )
      )}
    </span>
  );
}
