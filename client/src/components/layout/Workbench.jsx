import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

/**
 * Standard instrument layout: main canvas region + 360px inspector.
 * Below lg the inspector becomes a slide-over sheet toggled by a tab button.
 */
export default function Workbench({ canvas, inspector }) {
  const [sheetOpen, setSheetOpen] = useState(false);

  return (
    <div className="relative flex min-h-0 flex-1">
      <main className="relative min-w-0 flex-1 overflow-hidden">{canvas}</main>

      {/* desktop inspector */}
      <aside className="hidden w-[360px] shrink-0 flex-col border-l border-line bg-panel lg:flex">
        <div className="thin-scroll min-h-0 flex-1 overflow-y-auto">{inspector}</div>
      </aside>

      {/* mobile: inspector toggle + sheet */}
      <button
        type="button"
        onClick={() => setSheetOpen(true)}
        className="micro-label absolute top-3 right-3 z-30 border border-line bg-panel px-2.5 py-1.5 shadow-instrument active:translate-y-px lg:hidden"
        style={{ borderRadius: 3 }}
      >
        Inspector
      </button>
      <AnimatePresence>
        {sheetOpen && (
          <>
            <motion.div
              key="scrim"
              className="absolute inset-0 z-40 bg-black/20 lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              onClick={() => setSheetOpen(false)}
            />
            <motion.aside
              key="sheet"
              className="absolute inset-y-0 right-0 z-50 flex w-[min(360px,90vw)] flex-col border-l border-line bg-panel lg:hidden"
              initial={{ x: 24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 24, opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            >
              <div className="flex h-8 items-center justify-between border-b border-line px-3">
                <span className="micro-label">Inspector</span>
                <button
                  type="button"
                  onClick={() => setSheetOpen(false)}
                  className="micro-label hover:text-ink"
                >
                  Close
                </button>
              </div>
              <div className="thin-scroll min-h-0 flex-1 overflow-y-auto">{inspector}</div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
