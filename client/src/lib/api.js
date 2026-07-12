// DeepVision API client — single source of truth for backend access.
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body } = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("ENGINE UNREACHABLE — is the backend running on :8000?", 0);
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),

  createNetwork: (spec) => request("/networks", { method: "POST", body: { spec } }),
  deleteNetwork: (networkId) => request(`/networks/${networkId}`, { method: "DELETE" }),

  forward: ({ networkId, input, training = true }) =>
    request("/networks/forward", {
      method: "POST",
      body: { network_id: networkId, input, training },
    }),

  backward: ({ networkId, input, target, loss = "mse", training = true }) =>
    request("/networks/backward", {
      method: "POST",
      body: { network_id: networkId, input, target, loss, training },
    }),

  trace: ({ networkId, input, target, loss, layerIndex, weightRow, weightCol }) =>
    request("/networks/trace", {
      method: "POST",
      body: {
        network_id: networkId,
        input,
        target,
        loss,
        layer_index: layerIndex,
        weight_row: weightRow,
        weight_col: weightCol,
      },
    }),

  lossSurface: ({ surface, xMin = -2, xMax = 2, yMin = -2, yMax = 2, resolution = 70 }) =>
    request("/training/loss-surface", {
      method: "POST",
      body: {
        surface,
        x_min: xMin,
        x_max: xMax,
        y_min: yMin,
        y_max: yMax,
        resolution,
      },
    }),

  race: ({ surface, start, numSteps = 150, racers }) =>
    request("/training/race", {
      method: "POST",
      body: { surface, start, num_steps: numSteps, racers },
    }),
};

export { API_BASE };
