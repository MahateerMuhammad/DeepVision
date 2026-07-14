import KatexBlock from "../ui/KatexBlock";
import Odometer from "../ui/Odometer";
import InstrumentButton from "../ui/InstrumentButton";

export default function NodeCard({ node, forwardData, backwardData, onClose }) {
  const isInput = node.col === 0;
  const layer = isInput ? null : forwardData?.layers[node.layerIndex];
  const z = layer?.pre_activation[node.neuron];
  const a = layer?.post_activation[node.neuron];
  const gz = backwardData?.layers[node.layerIndex]?.grad_pre_activation?.[node.neuron];
  const eq = layer?.equations?.find((e) => e.neuron === node.neuron);

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <p className="micro-label">
            {isInput
              ? `Input x${node.neuron}`
              : `Layer ${String(node.layerIndex + 1).padStart(2, "0")} · neuron ${node.neuron}`}
          </p>
          {!isInput && (
            <p className="mono-num mt-0.5 text-[14px] text-ink-soft">{layer?.activation}</p>
          )}
        </div>
        <InstrumentButton size="sm" onClick={onClose}>
          Esc
        </InstrumentButton>
      </div>

      <div className="grid grid-cols-2 gap-2 p-4">
        {isInput ? (
          <div className="col-span-2 border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
            <p className="micro-label mb-1">value</p>
            <Odometer value={forwardData?.input[node.neuron]} decimals={4} className="text-xl font-medium" />
          </div>
        ) : (
          <>
            <div className="border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
              <p className="micro-label mb-1">z · pre-activation</p>
              <Odometer value={z} decimals={4} className="text-lg font-medium" />
            </div>
            <div className="border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
              <p className="micro-label mb-1">a · post-activation</p>
              <Odometer value={a} decimals={4} className="text-lg font-medium" />
            </div>
            {gz != null && (
              <div className="col-span-2 border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
                <p className="micro-label mb-1">∂L/∂z · gradient</p>
                <Odometer value={gz} decimals={6} className="text-lg font-medium" />
              </div>
            )}
          </>
        )}
      </div>

      {eq && (
        <div className="border-t border-line p-4">
          <p className="micro-label mb-2">Exact arithmetic</p>
          <div className="thin-scroll overflow-x-auto border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
            <KatexBlock latex={eq.linear_equation} />
          </div>
          <div className="thin-scroll mt-2 overflow-x-auto border border-line bg-canvas px-3 py-2" style={{ borderRadius: 8 }}>
            <KatexBlock latex={eq.activation_equation} />
          </div>
        </div>
      )}

      <div className="border-t border-line p-4">
        <p className="text-[14px] leading-relaxed text-ink-soft">
          What-if: the input vector fields above the fold are hot change one and press Enter to
          watch this neuron recompute through the same frozen weights.
        </p>
      </div>
    </div>
  );
}
