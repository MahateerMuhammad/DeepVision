import { useMemo } from "react";
import katex from "katex";

/** Render a LaTeX fragment via KaTeX. Never throws — errors render as raw source. */
export default function KatexBlock({ latex, display = false, chip = false, className = "" }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex ?? "", {
        displayMode: display,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return null;
    }
  }, [latex, display]);

  if (html == null) {
    return <code className={`mono-num text-xs ${className}`}>{latex}</code>;
  }
  return (
    <span
      className={`${chip ? "katex-chip" : "katex-block"} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
