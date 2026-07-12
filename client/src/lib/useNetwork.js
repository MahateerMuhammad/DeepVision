import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { useEngine } from "./useEngine";

export const DEFAULT_SPEC = {
  inputSize: 3,
  seed: 42,
  layers: [
    { out_features: 5, activation: "relu", dropout_prob: 0 },
    { out_features: 4, activation: "tanh", dropout_prob: 0 },
    { out_features: 2, activation: "sigmoid", dropout_prob: 0 },
  ],
};
export const DEFAULT_INPUT = [1.0, -2.0, 0.5];
export const DEFAULT_TARGET = [0.2, 0.8];

/**
 * Owns the live network: spec draft, creation, forward/backward results.
 * Playback / selection / trace stay in the page — this is the data spine.
 */
export function useNetwork(toast) {
  const { online, setParamCount } = useEngine();

  const [spec, setSpec] = useState(DEFAULT_SPEC);
  const [inputVec, setInputVec] = useState(DEFAULT_INPUT);
  const [targetVec, setTargetVec] = useState(DEFAULT_TARGET);
  const [loss, setLoss] = useState("mse");

  const [network, setNetwork] = useState(null); // {id, paramCount, spec}
  const [forward, setForward] = useState(null);
  const [backward, setBackward] = useState(null);
  const [busy, setBusy] = useState(false);
  const autoTried = useRef(false);
  const seqRef = useRef(0); // guards stale async writes

  const forge = useCallback(
    async (s = spec, input = inputVec) => {
      const seq = ++seqRef.current;
      setBusy(true);
      setForward(null);
      setBackward(null);
      try {
        const body = {
          input_size: s.inputSize,
          seed: s.seed,
          layers: s.layers.map((l) => ({
            out_features: l.out_features,
            activation: l.activation,
            dropout_prob: l.dropout_prob ?? 0,
          })),
        };
        const created = await api.createNetwork(body);
        if (seq !== seqRef.current) return null;
        const net = { id: created.network_id, paramCount: created.param_count };
        setNetwork(net);
        setParamCount(created.param_count);
        const fwd = await api.forward({ networkId: net.id, input });
        if (seq !== seqRef.current) return null;
        setForward(fwd);
        return net;
      } catch (e) {
        if (seq === seqRef.current) toast?.(`FORGE — ${e.message}`);
        return null;
      } finally {
        if (seq === seqRef.current) setBusy(false);
      }
    },
    [spec, inputVec, toast, setParamCount]
  );

  const runForward = useCallback(
    async (input = inputVec) => {
      if (!network) return null;
      const seq = ++seqRef.current;
      setBusy(true);
      try {
        const fwd = await api.forward({ networkId: network.id, input });
        if (seq !== seqRef.current) return null;
        setForward(fwd);
        setBackward(null); // stale after new input
        return fwd;
      } catch (e) {
        if (seq === seqRef.current) toast?.(`FORWARD — ${e.message}`);
        return null;
      } finally {
        if (seq === seqRef.current) setBusy(false);
      }
    },
    [network, inputVec, toast]
  );

  const runBackward = useCallback(
    async (input = inputVec, target = targetVec, lossName = loss) => {
      if (!network) return null;
      const seq = ++seqRef.current;
      setBusy(true);
      try {
        const bwd = await api.backward({
          networkId: network.id,
          input,
          target,
          loss: lossName,
        });
        if (seq !== seqRef.current) return null;
        setBackward(bwd);
        setForward(bwd); // backward payload includes full forward data
        return bwd;
      } catch (e) {
        if (seq === seqRef.current) toast?.(`BACKWARD — ${e.message}`);
        return null;
      } finally {
        if (seq === seqRef.current) setBusy(false);
      }
    },
    [network, inputVec, targetVec, loss, toast]
  );

  // auto-forge the default network once the engine is confirmed up
  useEffect(() => {
    if (online === true && !autoTried.current) {
      autoTried.current = true;
      forge();
    }
  }, [online, forge]);

  return {
    online,
    spec,
    setSpec,
    inputVec,
    setInputVec,
    targetVec,
    setTargetVec,
    loss,
    setLoss,
    network,
    forward,
    backward,
    busy,
    forge,
    runForward,
    runBackward,
  };
}
