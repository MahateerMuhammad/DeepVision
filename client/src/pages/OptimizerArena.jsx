import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { AnimatePresence, motion } from "framer-motion";
import Workbench from "../components/layout/Workbench";
import SegmentedControl from "../components/ui/SegmentedControl";
import InstrumentButton from "../components/ui/InstrumentButton";
import Fader from "../components/ui/Fader";
import Odometer from "../components/ui/Odometer";
import { api } from "../lib/api";
import { useToast } from "../components/ui/Toast";
import { useEngine } from "../lib/useEngine";

const W = 760;
const H = 640;
const M = 24;
const PW = W - 2 * M;
const PH = H - 2 * M;

const SURFACES = [
  { value: "bowl", label: "Bowl" },
  { value: "saddle", label: "Saddle" },
  { value: "rosenbrock", label: "Rosenbrock" },
];

const DEFAULT_RACERS = [
  {
    key: "sgd",
    label: "SGD + Momentum",
    optimizer: "sgd",
    color: "#0EA5E9",
    lr: 0.05,
    enabled: true,
    hyperparams: { momentum: 0.9 },
  },
  {
    key: "rmsprop",
    label: "RMSProp",
    optimizer: "rmsprop",
    color: "#10B981",
    lr: 0.05,
    enabled: true,
    hyperparams: {},
  },
  {
    key: "adam",
    label: "Adam",
    optimizer: "adam",
    color: "#18181B",
    lr: 0.05,
    enabled: true,
    hyperparams: {},
  },
];

function Sparkline({ points, frame, color }) {
  const upTo = points.slice(0, Math.max(2, frame + 1)).filter((p) => p.loss != null);
  if (upTo.length < 2) return <svg width="120" height="26" />;
  const xs = d3.scaleLinear().domain([0, points.length - 1]).range([1, 119]);
  const finite = upTo.map((p) => p.loss).filter((v) => Number.isFinite(v));
  const ys = d3
    .scaleLinear()
    .domain([Math.min(...finite), Math.max(...finite)])
    .range([24, 2]);
  const line = d3
    .line()
    .x((p) => xs(p.step))
    .y((p) => ys(Math.max(ys.domain()[0], Math.min(ys.domain()[1], p.loss))));
  return (
    <svg width="120" height="26" className="shrink-0">
      <path d={line(upTo)} fill="none" stroke={color} strokeWidth="1" />
    </svg>
  );
}

export default function OptimizerArena() {
  const toast = useToast();
  const { online } = useEngine();
  const [surface, setSurface] = useState("bowl");
  const [grid, setGrid] = useState(null); // {x,y,z}
  const [gridLoading, setGridLoading] = useState(false);
  const [start, setStart] = useState([-1.5, 1.4]);
  const [racers, setRacers] = useState(DEFAULT_RACERS);
  const [race, setRace] = useState(null); // API result
  const [racing, setRacing] = useState(false);
  const [frame, setFrame] = useState(0);
  const rafRef = useRef(null);

  // ── surface fetch ──
  useEffect(() => {
    let alive = true;
    setGridLoading(true);
    setGrid(null);
    setRace(null);
    setFrame(0);
    api
      .lossSurface({ surface, resolution: 70 })
      .then((data) => alive && setGrid(data))
      .catch((e) => alive && toast(`LOSS SURFACE — ${e.message}`))
      .finally(() => alive && setGridLoading(false));
    return () => {
      alive = false;
    };
  }, [surface, toast]);

  // ── coordinate mapping (data ↔ plot px) ──
  const extent = useMemo(() => {
    if (!grid) return { x0: -2, x1: 2, y0: -2, y1: 2 };
    return {
      x0: grid.x[0],
      x1: grid.x[grid.x.length - 1],
      y0: grid.y[0],
      y1: grid.y[grid.y.length - 1],
    };
  }, [grid]);
  const xS = useMemo(
    () => d3.scaleLinear().domain([extent.x0, extent.x1]).range([M, M + PW]),
    [extent]
  );
  const yS = useMemo(
    () => d3.scaleLinear().domain([extent.y0, extent.y1]).range([M + PH, M]),
    [extent]
  );

  // ── contours ──
  const contours = useMemo(() => {
    if (!grid) return null;
    const n = grid.x.length;
    const m = grid.y.length;
    const values = new Float64Array(n * m);
    let zmin = Infinity;
    let zmax = -Infinity;
    let minIdx = [0, 0];
    for (let j = 0; j < m; j++) {
      for (let i = 0; i < n; i++) {
        const v = grid.z[j][i];
        values[j * n + i] = v;
        if (v < zmin) {
          zmin = v;
          minIdx = [i, j];
        }
        if (v > zmax) zmax = v;
      }
    }
    const thresholds = d3.range(1, 16).map((t) => zmin + (zmax - zmin) * Math.pow(t / 16, 2.4));
    const polys = d3.contours().size([n, m]).thresholds(thresholds)(values);
    // grid coords → plot px
    const sx = PW / (n - 1);
    const sy = PH / (m - 1);
    const path = d3.geoPath(
      d3.geoTransform({
        point(px, py) {
          this.stream.point(M + px * sx, M + PH - py * sy);
        },
      })
    );
    const shade = d3
      .scaleLinear()
      .domain([0, polys.length - 1])
      .range(["#D4D4D8", "#FAFAFA"]);
    return {
      paths: polys.map((p, i) => ({ d: path(p), fill: shade(i) })),
      min: { x: grid.x[minIdx[0]], y: grid.y[minIdx[1]] },
    };
  }, [grid]);

  // ── race ──
  const runRace = async () => {
    const active = racers.filter((r) => r.enabled);
    if (active.length === 0) {
      toast("SELECT AT LEAST ONE RACER", "info");
      return;
    }
    setRacing(true);
    setRace(null);
    setFrame(0);
    try {
      const data = await api.race({
        surface,
        start,
        numSteps: 150,
        racers: active.map((r) => ({
          label: r.label,
          optimizer: r.optimizer,
          lr: r.lr,
          hyperparams: r.hyperparams,
        })),
      });
      setRace(data);
      // animate
      const maxLen = Math.max(...data.racers.map((r) => r.points.length));
      let f = 0;
      const tick = () => {
        f += 1.25;
        setFrame(Math.min(Math.floor(f), maxLen - 1));
        if (f < maxLen - 1) {
          rafRef.current = requestAnimationFrame(tick);
        } else {
          setRacing(false);
        }
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      toast(`RACE — ${e.message}`);
      setRacing(false);
    }
  };
  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  const svgRef = useRef(null);
  const onCanvasClick = useCallback(
    (e) => {
      if (racing) return;
      const rect = svgRef.current.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const py = ((e.clientY - rect.top) / rect.height) * H;
      const x = xS.invert(px);
      const y = yS.invert(py);
      if (x < extent.x0 || x > extent.x1 || y < extent.y0 || y > extent.y1) return;
      setStart([Number(x.toFixed(3)), Number(y.toFixed(3))]);
      setRace(null);
      setFrame(0);
    },
    [xS, yS, extent, racing]
  );

  // per-racer render state at current frame
  const racerFrames = useMemo(() => {
    if (!race) return [];
    return race.racers.map((r) => {
      const cfg = racers.find((c) => c.label === r.label) ?? {};
      const idx = Math.min(frame, r.points.length - 1);
      const valid = r.points.filter((p) => p.x != null);
      const visible = valid.slice(0, idx + 1);
      const cur = visible[visible.length - 1];
      const hitNull = r.points[idx]?.x == null;
      let streak = null;
      if (hitNull && visible.length >= 2) {
        const a = visible[visible.length - 2];
        const b = cur;
        streak = { dx: (b.x - a.x) * 40, dy: (b.y - a.y) * 40 };
      }
      return { ...r, cfg, visible, cur, diverging: hitNull, streak };
    });
  }, [race, frame, racers]);

  const trailLine = useMemo(
    () => d3.line().x((p) => xS(p.x)).y((p) => yS(p.y)),
    [xS, yS]
  );

  // ── canvas ──
  const canvas = (
    <div className="dot-grid relative flex h-full items-center justify-center overflow-hidden p-4">
      {!online && !grid ? (
        <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
          <p className="micro-label !text-crimson mb-1">Engine offline</p>
          <p className="text-xs text-ink-soft">Start the backend on :8000 to load loss surfaces.</p>
        </div>
      ) : gridLoading || !contours ? (
        <div className="flex flex-col items-center gap-3">
          <div className="h-64 w-64 animate-pulse border border-line bg-panel" style={{ borderRadius: 4 }} />
          <p className="micro-label">Computing loss surface…</p>
        </div>
      ) : (
        <div className="relative h-full w-full">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="mx-auto block h-full w-full cursor-crosshair"
            style={{ maxWidth: 860 }}
            onClick={onCanvasClick}
          >
            <rect x={M} y={M} width={PW} height={PH} fill="#FAFAFA" stroke="#E4E4E7" />
            <g>
              {contours.paths.map((p, i) => (
                <path key={i} d={p.d} fill={p.fill} stroke="#E4E4E7" strokeWidth="1" />
              ))}
            </g>
            {/* global minimum crosshair */}
            <g stroke="#71717A" strokeWidth="1">
              <line x1={xS(contours.min.x) - 9} x2={xS(contours.min.x) + 9} y1={yS(contours.min.y)} y2={yS(contours.min.y)} />
              <line x1={xS(contours.min.x)} x2={xS(contours.min.x)} y1={yS(contours.min.y) - 9} y2={yS(contours.min.y) + 9} />
              <circle cx={xS(contours.min.x)} cy={yS(contours.min.y)} r="4" fill="none" />
            </g>
            {/* start crosshair */}
            <g stroke="#18181B" strokeWidth="1.5">
              <line x1={xS(start[0]) - 7} x2={xS(start[0]) + 7} y1={yS(start[1])} y2={yS(start[1])} />
              <line x1={xS(start[0])} x2={xS(start[0])} y1={yS(start[1]) - 7} y2={yS(start[1]) + 7} />
            </g>
            {/* trajectories */}
            {racerFrames.map((r) => (
              <g key={r.label}>
                {r.visible.length > 1 && (
                  <path
                    d={trailLine(r.visible)}
                    fill="none"
                    stroke={r.cfg.color}
                    strokeWidth="1"
                    opacity="0.75"
                  />
                )}
                {r.cur && r.cur.momentum_vector && !r.diverging && (
                  <line
                    x1={xS(r.cur.x)}
                    y1={yS(r.cur.y)}
                    x2={xS(r.cur.x + r.cur.momentum_vector[0] * 6)}
                    y2={yS(r.cur.y + r.cur.momentum_vector[1] * 6)}
                    stroke={r.cfg.color}
                    strokeWidth="2.5"
                    opacity="0.3"
                    strokeLinecap="round"
                  />
                )}
                {r.cur && (
                  <motion.circle
                    r="5"
                    fill={r.cfg.color}
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                    initial={false}
                    animate={
                      r.diverging && r.streak
                        ? {
                            cx: xS(r.cur.x + r.streak.dx),
                            cy: yS(r.cur.y + r.streak.dy),
                            opacity: 0,
                          }
                        : { cx: xS(r.cur.x), cy: yS(r.cur.y), opacity: 1 }
                    }
                    transition={
                      r.diverging
                        ? { duration: 0.6, ease: "easeIn" }
                        : { duration: 0.05, ease: "linear" }
                    }
                  />
                )}
              </g>
            ))}
          </svg>
          <div className="pointer-events-none absolute bottom-6 left-6">
            <p className="micro-label">
              Click map to set start · <span className="mono-num">({start[0].toFixed(2)}, {start[1].toFixed(2)})</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );

  // ── inspector ──
  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Loss Surface</p>
        <SegmentedControl options={SURFACES} value={surface} onChange={setSurface} className="w-full" />
      </div>

      <div className="border-b border-line p-4">
        <p className="micro-label mb-3">Racer Roster</p>
        <div className="flex flex-col gap-3">
          {racers.map((r, i) => {
            const result = racerFrames.find((x) => x.label === r.label);
            const finalShown = result?.cur;
            const diverged = result && race?.racers.find((x) => x.label === r.label)?.diverged;
            const showDiverged = diverged && result?.diverging;
            return (
              <div
                key={r.key}
                className={`border p-3 transition-colors duration-150 ${r.enabled ? "border-line bg-panel" : "border-line bg-canvas opacity-60"}`}
                style={{ borderRadius: 3 }}
              >
                <div className="mb-2 flex items-center gap-2">
                  <button
                    type="button"
                    aria-label={`toggle ${r.label}`}
                    onClick={() =>
                      setRacers((rs) => rs.map((x, j) => (j === i ? { ...x, enabled: !x.enabled } : x)))
                    }
                    className="flex h-4 w-4 items-center justify-center border border-ink"
                    style={{ borderRadius: 2, background: r.enabled ? r.color : "transparent" }}
                  />
                  <span className="text-[11px] font-semibold tracking-[0.06em] uppercase">{r.label}</span>
                  <span className="flex-1" />
                  <AnimatePresence>
                    {showDiverged && (
                      <motion.span
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="mono-num border border-crimson bg-crimson px-1.5 py-0.5 text-[9px] font-semibold text-white"
                        style={{ borderRadius: 2 }}
                      >
                        NaN — DIVERGED
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
                <div className="mb-1 flex items-center justify-between">
                  <span className="micro-label">Learning rate</span>
                  <span className={`mono-num text-[11px] font-medium ${r.lr >= 0.6 ? "text-crimson" : ""}`}>
                    {r.lr.toFixed(3)}
                  </span>
                </div>
                <Fader
                  value={r.lr}
                  min={0.001}
                  max={1}
                  step={0.001}
                  redZoneFrom={0.6}
                  accent={r.color}
                  disabled={!r.enabled}
                  onChange={(v) => setRacers((rs) => rs.map((x, j) => (j === i ? { ...x, lr: v } : x)))}
                />
                {result && (
                  <div className="mt-2 flex items-center justify-between gap-2 border-t border-line pt-2">
                    <Sparkline
                      points={race.racers.find((x) => x.label === r.label)?.points ?? []}
                      frame={frame}
                      color={r.color}
                    />
                    <div className="text-right">
                      <p className="micro-label">Loss</p>
                      {finalShown && Number.isFinite(finalShown.loss) ? (
                        <Odometer value={finalShown.loss} decimals={4} className="text-xs font-medium" />
                      ) : (
                        <span className="mono-num text-xs text-crimson">NaN</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="micro-label">Start point</span>
          <span className="mono-num text-[11px]">
            ({start[0].toFixed(2)}, {start[1].toFixed(2)})
          </span>
        </div>
        <InstrumentButton
          variant="primary"
          className="w-full"
          disabled={racing || !grid}
          onClick={runRace}
        >
          {racing ? "Racing…" : "Race"}
        </InstrumentButton>
        <p className="mt-3 text-[11px] leading-relaxed text-ink-soft">
          Push a learning-rate fader into the red zone and re-race to watch a trajectory blow up —
          the dot streaks off the map when the backend reports NaN.
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}
