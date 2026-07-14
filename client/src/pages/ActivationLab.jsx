import { useCallback, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import Workbench from "../components/layout/Workbench";
import Panel from "../components/ui/Panel";
import SegmentedControl from "../components/ui/SegmentedControl";
import InstrumentButton from "../components/ui/InstrumentButton";
import Odometer from "../components/ui/Odometer";
import KatexBlock from "../components/ui/KatexBlock";
import {
  ACTIVATIONS,
  ACTIVATION_KEYS,
  sampleCurve,
  monotoneResample,
  numericalDerivative,
} from "../lib/activations";

const W = 720;
const H = 300;
const M = { l: 44, r: 16, t: 14, b: 26 };

function GridPlot({ xScale, yScale, children, onPointer, cursor = "crosshair" }) {
  const svgRef = useRef(null);
  const toXY = useCallback(
    (e) => {
      const rect = svgRef.current.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      const py = ((e.clientY - rect.top) / rect.height) * H;
      return { x: xScale.invert(px), y: yScale.invert(py), px, py };
    },
    [xScale, yScale]
  );

  const xTicks = xScale.ticks(9);
  const yTicks = yScale.ticks(6);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full touch-none select-none"
      style={{ cursor }}
      onPointerDown={(e) => {
        try {
          svgRef.current.setPointerCapture(e.pointerId);
        } catch {
          /* synthetic pointers can't be captured */
        }
        onPointer?.("down", toXY(e), e);
      }}
      onPointerMove={(e) => onPointer?.("move", toXY(e), e)}
      onPointerUp={(e) => onPointer?.("up", toXY(e), e)}
    >
      {/* dotted gridlines */}
      {xTicks.map((t) => (
        <line
          key={`x${t}`}
          x1={xScale(t)}
          x2={xScale(t)}
          y1={M.t}
          y2={H - M.b}
          stroke="#E4E4E7"
          strokeWidth="1"
          strokeDasharray="1 4"
        />
      ))}
      {yTicks.map((t) => (
        <line
          key={`y${t}`}
          x1={M.l}
          x2={W - M.r}
          y1={yScale(t)}
          y2={yScale(t)}
          stroke="#E4E4E7"
          strokeWidth="1"
          strokeDasharray="1 4"
        />
      ))}
      {/* axes */}
      {yScale(0) >= M.t && yScale(0) <= H - M.b && (
        <line x1={M.l} x2={W - M.r} y1={yScale(0)} y2={yScale(0)} stroke="#71717A" strokeWidth="1" />
      )}
      {xScale(0) >= M.l && xScale(0) <= W - M.r && (
        <line x1={xScale(0)} x2={xScale(0)} y1={M.t} y2={H - M.b} stroke="#71717A" strokeWidth="1" />
      )}
      {/* tick labels */}
      {xTicks.map((t) => (
        <text
          key={`xl${t}`}
          x={xScale(t)}
          y={H - M.b + 14}
          textAnchor="middle"
          fontSize="9"
          fill="#71717A"
          fontFamily="IBM Plex Mono, monospace"
        >
          {t}
        </text>
      ))}
      {yTicks.map((t) => (
        <text
          key={`yl${t}`}
          x={M.l - 6}
          y={yScale(t) + 3}
          textAnchor="end"
          fontSize="9"
          fill="#71717A"
          fontFamily="IBM Plex Mono, monospace"
        >
          {t}
        </text>
      ))}
      <clipPath id="plot-clip">
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} />
      </clipPath>
      <g clipPath="url(#plot-clip)">{children}</g>
    </svg>
  );
}

export default function ActivationLab() {
  const [fnKey, setFnKey] = useState("relu");
  const [x0, setX0] = useState(0.8);
  const [drawMode, setDrawMode] = useState(false);
  const [drawn, setDrawn] = useState(null); // resampled [{x,y}]
  const drawingRef = useRef(null); // raw points while dragging

  const def = ACTIVATIONS[fnKey];
  const [dx0, dx1] = def.domain;

  const xScale = useMemo(() => d3.scaleLinear().domain(def.domain).range([M.l, W - M.r]), [def]);
  const yTop = useMemo(() => d3.scaleLinear().domain(def.range).range([H - M.b, M.t]), [def]);
  const yBot = useMemo(() => d3.scaleLinear().domain(def.dRange).range([H - M.b, M.t]), [def]);

  const curve = useMemo(
    () => (drawMode && drawn ? drawn : sampleCurve(def.f, dx0, dx1)),
    [def, dx0, dx1, drawMode, drawn]
  );
  const dCurve = useMemo(
    () => (drawMode && drawn ? numericalDerivative(drawn) : sampleCurve(def.df, dx0, dx1)),
    [def, dx0, dx1, drawMode, drawn]
  );

  const lineTop = useMemo(
    () => d3.line().x((p) => xScale(p.x)).y((p) => yTop(p.y)),
    [xScale, yTop]
  );
  const lineBot = useMemo(() => {
    const [lo, hi] = yBot.domain();
    return d3
      .line()
      .x((p) => xScale(p.x))
      .y((p) => yBot(Math.max(lo, Math.min(hi, p.y))));
  }, [xScale, yBot]);

  // values at slider position
  const interp = useCallback((pts, x) => {
    if (!pts?.length) return 0;
    const i = Math.min(
      pts.length - 1,
      Math.max(0, Math.round(((x - pts[0].x) / (pts[pts.length - 1].x - pts[0].x)) * (pts.length - 1)))
    );
    return pts[i].y;
  }, []);
  const fx = drawMode && drawn ? interp(drawn, x0) : def.f(x0);
  const dfx = drawMode && drawn ? interp(dCurve, x0) : def.df(x0);

  const handleTopPointer = (phase, pt, e) => {
    if (drawMode) {
      if (phase === "down") {
        drawingRef.current = [pt];
        setDrawn(null);
      } else if (phase === "move" && drawingRef.current && e.buttons === 1) {
        drawingRef.current.push(pt);
        // live preview while drawing
        const rs = monotoneResample(drawingRef.current, dx0, dx1);
        if (rs) setDrawn(rs);
      } else if (phase === "up" && drawingRef.current) {
        const rs = monotoneResample(drawingRef.current, dx0, dx1);
        drawingRef.current = null;
        if (rs) setDrawn(rs);
      }
    } else if (phase === "down" || (phase === "move" && e.buttons === 1)) {
      setX0(Math.max(dx0, Math.min(dx1, pt.x)));
    }
  };
  const handleBotPointer = (phase, pt, e) => {
    if (phase === "down" || (phase === "move" && e.buttons === 1)) {
      setX0(Math.max(dx0, Math.min(dx1, pt.x)));
    }
  };

  // tangent segment around x0
  const tangent = useMemo(() => {
    const span = (dx1 - dx0) * 0.22;
    return {
      x1: xScale(x0 - span),
      y1: yTop(fx - dfx * span),
      x2: xScale(x0 + span),
      y2: yTop(fx + dfx * span),
    };
  }, [x0, fx, dfx, xScale, yTop, dx0, dx1]);

  const canvas = (
    <div className="thin-scroll dot-grid h-full overflow-y-auto p-4">
      <div className="mx-auto flex max-w-3xl flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SegmentedControl
            options={ACTIVATION_KEYS.map((k) => ({ value: k, label: ACTIVATIONS[k].label }))}
            value={fnKey}
            onChange={(v) => {
              setFnKey(v);
              setDrawn(null);
              setX0(0.8);
            }}
          />
          <div className="flex items-center gap-2">
            <InstrumentButton
              active={drawMode}
              onClick={() => {
                setDrawMode((d) => !d);
                setDrawn(null);
              }}
            >
              Draw custom f(x)
            </InstrumentButton>
            {drawMode && drawn && (
              <InstrumentButton onClick={() => setDrawn(null)}>Clear</InstrumentButton>
            )}
          </div>
        </div>

        <Panel
          title={drawMode ? "f(x) freehand input" : `f(x) ${def.label}`}
          right={
            <span className="micro-label">
              {drawMode ? "drag to draw the curve" : "drag to move the probe"}
            </span>
          }
        >
          <GridPlot
            xScale={xScale}
            yScale={yTop}
            onPointer={handleTopPointer}
            cursor={drawMode ? "cell" : "crosshair"}
          >
            {curve.length > 0 && (
              <path
                d={lineTop(curve)}
                fill="none"
                stroke={drawMode ? "#0EA5E9" : "#18181B"}
                strokeWidth="1.75"
              />
            )}
            {drawMode && !drawn && (
              <text
                x={W / 2}
                y={H / 2}
                textAnchor="middle"
                fontSize="11"
                fill="#71717A"
                fontFamily="IBM Plex Mono, monospace"
              >
                DRAG LEFT → RIGHT TO SKETCH f(x)
              </text>
            )}
            {/* tangent + probe */}
            <line
              x1={tangent.x1}
              y1={tangent.y1}
              x2={tangent.x2}
              y2={tangent.y2}
              stroke="#18181B"
              strokeWidth="1.5"
              opacity={drawMode && !drawn ? 0 : 1}
            />
            <line
              x1={xScale(x0)}
              x2={xScale(x0)}
              y1={M.t}
              y2={H - M.b}
              stroke="#18181B"
              strokeWidth="1"
              strokeDasharray="2 3"
              opacity="0.4"
            />
            <circle
              cx={xScale(x0)}
              cy={yTop(fx)}
              r="4.5"
              fill="#FFFFFF"
              stroke="#18181B"
              strokeWidth="1.5"
              opacity={drawMode && !drawn ? 0 : 1}
            />
          </GridPlot>
        </Panel>

        <Panel
          title={drawMode ? "f′(x) numerical differentiation" : `f′(x) derivative`}
          right={
            <span className="mono-num text-[10px] text-ink-soft">
              slope @ x = {x0.toFixed(2)}
            </span>
          }
        >
          <GridPlot xScale={xScale} yScale={yBot} onPointer={handleBotPointer}>
            {dCurve.length > 0 && (
              <path d={lineBot(dCurve)} fill="none" stroke="#0EA5E9" strokeWidth="1.75" />
            )}
            <line
              x1={xScale(x0)}
              x2={xScale(x0)}
              y1={M.t}
              y2={H - M.b}
              stroke="#18181B"
              strokeWidth="1"
              strokeDasharray="2 3"
              opacity="0.4"
            />
            <circle
              cx={xScale(x0)}
              cy={yBot(
                Math.max(yBot.domain()[0], Math.min(yBot.domain()[1], dfx))
              )}
              r="4.5"
              fill="#0EA5E9"
              stroke="#FFFFFF"
              strokeWidth="1.5"
            />
          </GridPlot>
        </Panel>
      </div>
    </div>
  );

  const inspector = (
    <div className="flex flex-col">
      <div className="border-b border-line p-4">
        <p className="micro-label mb-3">Probe Readout</p>
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: "x", v: x0 },
            { label: "f(x)", v: fx },
            { label: "f′(x)", v: dfx },
          ].map((r) => (
            <div key={r.label} className="border border-line bg-canvas px-2 py-2" style={{ borderRadius: 3 }}>
              <p className="micro-label mb-1">{r.label}</p>
              <Odometer value={r.v} decimals={3} className="text-sm font-medium" />
            </div>
          ))}
        </div>
      </div>
      <div className="border-b border-line p-4">
        <p className="micro-label mb-2">Definition</p>
        <div className="border border-line bg-canvas px-3 py-2" style={{ borderRadius: 3 }}>
          <KatexBlock
            display
            latex={
              {
                relu: "f(x) = \\max(0, x)",
                leaky_relu: "f(x) = \\begin{cases} x & x > 0 \\\\ 0.01x & x \\le 0 \\end{cases}",
                sigmoid: "f(x) = \\frac{1}{1 + e^{-x}}",
                tanh: "f(x) = \\tanh(x)",
                linear: "f(x) = x",
              }[fnKey]
            }
          />
        </div>
      </div>
      <div className="p-4">
        <p className="micro-label mb-2">{drawMode ? "Why smoothness matters" : "Field notes"}</p>
        <p className="text-[14px] leading-relaxed text-ink-soft">
          {drawMode
            ? "Backpropagation multiplies derivatives layer by layer, so f′ must exist and behave everywhere. Sketch a jagged curve above and watch the derivative panel spike: every kink becomes a discontinuity, and every flat segment kills the gradient outright. Smooth, monotone-ish activations keep the training signal alive."
            : def.note}
        </p>
      </div>
    </div>
  );

  return <Workbench canvas={canvas} inspector={inspector} />;
}
