import { createContext, useCallback, useContext, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const ToastContext = createContext(() => {});

export function useToast() {
  return useContext(ToastContext);
}

/** Slim instrument toasts, bottom-right. `push(message, kind)` kind: "error" | "info". */
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const idRef = useRef(0);

  const push = useCallback((message, kind = "error") => {
    const id = ++idRef.current;
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4500);
  }, []);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-[100] flex w-80 flex-col gap-2">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className={`pointer-events-auto border bg-panel px-3 py-2 shadow-instrument ${
                t.kind === "error" ? "border-crimson" : "border-line"
              }`}
              style={{ borderRadius: 3 }}
            >
              <p
                className={`micro-label !normal-case !tracking-normal !text-[11px] ${
                  t.kind === "error" ? "!text-crimson" : "!text-ink"
                }`}
              >
                {t.message}
              </p>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}
