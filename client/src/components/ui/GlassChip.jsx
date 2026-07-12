import { motion } from "framer-motion";

/** Frosted glass chip that slides up 8px with fade. */
export default function GlassChip({ children, className = "", style }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
      className={`glass px-3 py-2 ${className}`}
      style={{ borderRadius: 4, ...style }}
    >
      {children}
    </motion.div>
  );
}
