import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api } from "./api";

const EngineContext = createContext({
  online: null,
  paramCount: null,
  setParamCount: () => {},
});

/** Polls /health and shares engine status + current network param count app-wide. */
export function EngineProvider({ children }) {
  const [online, setOnline] = useState(null); // null = probing
  const [paramCount, setParamCount] = useState(null);

  useEffect(() => {
    let alive = true;
    const probe = async () => {
      try {
        await api.health();
        if (alive) setOnline(true);
      } catch {
        if (alive) setOnline(false);
      }
    };
    probe();
    const t = setInterval(probe, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const value = useMemo(
    () => ({ online, paramCount, setParamCount }),
    [online, paramCount]
  );
  return <EngineContext.Provider value={value}>{children}</EngineContext.Provider>;
}

export function useEngine() {
  return useContext(EngineContext);
}
