import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Edges, Bounds, ContactShadows } from "@react-three/drei";
import { makeHeatTexture } from "./featureTexture";

const TARGET = 2.6; // world units for the largest spatial dimension
const SPACING_X = 3.9; // distance between layer towers
const PANE_GAP = 0.17; // depth between channels in a tower
const SLAB = 0.08; // glass sheet thickness

function Pane({ grid, w, h, z, selected, dim, onSelect }) {
  const texture = useMemo(() => makeHeatTexture(grid), [grid]);
  useEffect(() => () => texture.dispose(), [texture]);
  const [hovered, setHovered] = useState(false);
  const pull = selected ? 0.3 : hovered ? 0.12 : 0;
  const edgeColor = selected ? "#0EA5E9" : hovered ? "#38BDF8" : "#CBD5E1";

  return (
    <group position={[0, 0, z + pull]} scale={selected ? 1.05 : 1}>
      {/* frosted glass sheet */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          setHovered(false);
          document.body.style.cursor = "auto";
        }}
      >
        <boxGeometry args={[w + 0.07, h + 0.07, SLAB]} />
        <meshStandardMaterial
          color="#ffffff"
          roughness={0.3}
          metalness={0}
          transparent
          opacity={dim ? 0.08 : 0.32}
          depthWrite={false}
        />
        <Edges color={edgeColor} />
      </mesh>
      {/* heat map on the front face */}
      <mesh position={[0, 0, SLAB / 2 + 0.004]} raycast={() => null}>
        <planeGeometry args={[w, h]} />
        <meshBasicMaterial map={texture} transparent opacity={dim ? 0.32 : 1} toneMapped={false} />
      </mesh>
    </group>
  );
}

function Tower({ layer, x, w, h, selected, onSelect }) {
  const C = layer.channels.length;
  return (
    <group position={[x, 0, 0]}>
      {layer.channels.map((grid, ci) => (
        <Pane
          key={ci}
          grid={grid}
          w={w}
          h={h}
          z={(ci - (C - 1) / 2) * PANE_GAP}
          selected={selected?.layerIndex === layer.layerIndex && selected?.channel === ci}
          dim={selected != null && selected.layerIndex !== layer.layerIndex}
          onSelect={() => onSelect(layer.layerIndex, ci)}
        />
      ))}
      <Html center position={[0, -h / 2 - 0.62, 0]} zIndexRange={[10, 0]}>
        <div
          className="pointer-events-none flex flex-col items-center gap-0.5 whitespace-nowrap border border-line bg-panel/80 px-2 py-1 text-center shadow-instrument"
          style={{ borderRadius: 4, backdropFilter: "blur(6px)" }}
        >
          <span className="text-[10px] font-semibold uppercase tracking-[0.06em] text-ink">{layer.label}</span>
          {layer.sublabel && <span className="mono-num text-[8px] text-ink-soft">{layer.sublabel}</span>}
          <span className="mono-num text-[9px] text-cerulean">
            {C}×{layer.spatial[0]}×{layer.spatial[1]}
          </span>
        </div>
      </Html>
    </group>
  );
}

/** Animated connector: a beam + arrowhead + packets flowing left→right, so the
 *  scene reads as a pipeline (data flowing layer to layer). */
function Flow({ from, to }) {
  const len = to - from;
  const mid = (from + to) / 2;
  const packets = useRef([]);
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    for (let i = 0; i < packets.current.length; i++) {
      const m = packets.current[i];
      if (!m) continue;
      const p = (t * 0.45 + i / 3) % 1;
      m.position.x = from + p * len;
      m.scale.setScalar(0.5 + 0.5 * Math.sin(p * Math.PI));
    }
  });
  if (len <= 0.06) return null;
  return (
    <group>
      <mesh position={[mid, 0, 0]} rotation={[0, 0, Math.PI / 2]} raycast={() => null}>
        <cylinderGeometry args={[0.009, 0.009, len, 8]} />
        <meshBasicMaterial color="#0EA5E9" transparent opacity={0.16} />
      </mesh>
      <mesh position={[to - 0.1, 0, 0]} rotation={[0, 0, -Math.PI / 2]} raycast={() => null}>
        <coneGeometry args={[0.06, 0.16, 12]} />
        <meshBasicMaterial color="#0EA5E9" transparent opacity={0.4} />
      </mesh>
      {[0, 1, 2].map((i) => (
        <mesh key={i} ref={(el) => (packets.current[i] = el)} raycast={() => null}>
          <sphereGeometry args={[0.05, 12, 12]} />
          <meshBasicMaterial color="#0EA5E9" />
        </mesh>
      ))}
    </group>
  );
}

export default function FeatureScene({ layers, selected, onSelect }) {
  const maxDim = useMemo(
    () => Math.max(1, ...layers.map((l) => Math.max(l.spatial[0], l.spatial[1]))),
    [layers]
  );
  const cellUnit = TARGET / maxDim;
  const geo = useMemo(
    () =>
      layers.map((l, i) => ({
        x: i * SPACING_X,
        w: l.spatial[1] * cellUnit,
        h: l.spatial[0] * cellUnit,
      })),
    [layers, cellUnit]
  );
  const cx = ((layers.length - 1) * SPACING_X) / 2;
  const maxH = Math.max(1, ...geo.map((g) => g.h));
  const maxW = Math.max(1, ...geo.map((g) => g.w));
  const groundY = -maxH / 2 - 0.9;
  const sceneW = (layers.length - 1) * SPACING_X + maxW + 3;

  const boundsKey = useMemo(
    () => layers.map((l) => `${l.layerIndex}:${l.spatial.join("x")}:${l.channels.length}`).join("|"),
    [layers]
  );

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ fov: 34, position: [cx + 2, 2.4, 14] }}
      style={{ background: "#FAFAFA" }}
      onPointerMissed={() => onSelect(null, null)}
    >
      <hemisphereLight args={["#ffffff", "#dbe4ee", 0.75]} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[6, 10, 8]} intensity={1.05} />
      <directionalLight position={[-6, 3, -4]} intensity={0.25} />

      <Bounds key={boundsKey} fit clip observe margin={1.08}>
        {layers.map((layer, i) => (
          <Tower
            key={layer.layerIndex}
            layer={layer}
            x={geo[i].x}
            w={geo[i].w}
            h={geo[i].h}
            selected={selected}
            onSelect={onSelect}
          />
        ))}
        {geo.slice(0, -1).map((g, i) => (
          <Flow key={i} from={g.x + g.w / 2} to={geo[i + 1].x - geo[i + 1].w / 2} />
        ))}
      </Bounds>

      <ContactShadows
        position={[cx, groundY, 0]}
        scale={sceneW}
        opacity={0.16}
        blur={2.8}
        far={6}
        resolution={512}
        color="#334155"
      />
      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={3}
        maxDistance={60}
        target={[cx, 0, 0]}
        minPolarAngle={0.2}
        maxPolarAngle={Math.PI / 2 + 0.15}
      />
    </Canvas>
  );
}
