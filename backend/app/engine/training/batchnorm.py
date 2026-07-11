import torch

from app.engine.common.serialize import to_list

DEFAULT_EPS = 1e-5


def compute_batchnorm_trace(
    batch: list[list[float]],
    gamma: list[float] | None = None,
    beta: list[float] | None = None,
    eps: float = DEFAULT_EPS,
) -> dict:
    """Standalone batch-normalization trace: given a batch of feature
    vectors, compute the per-feature batch mean/variance, the normalized
    (zero-mean, unit-variance) values, and the final scale-and-shift output
    gamma*normalized + beta.

    Deliberately independent of DynamicANN/hooks: BatchNorm is a genuinely
    batch-level operation (its statistics are undefined for a single sample),
    which doesn't fit the rest of the ANN engine's single-sample VCR-style
    stepping model. Kept as a pure function, same pattern as sliding_kernel.
    """
    if not batch:
        raise ValueError("batch must be non-empty")

    num_features = len(batch[0])
    if any(len(row) != num_features for row in batch):
        raise ValueError("all rows in batch must have the same number of features")
    if len(batch) < 2:
        raise ValueError("batch must contain at least 2 samples for meaningful statistics")

    if gamma is None:
        gamma = [1.0] * num_features
    if beta is None:
        beta = [0.0] * num_features
    if len(gamma) != num_features:
        raise ValueError(f"gamma length {len(gamma)} does not match num_features {num_features}")
    if len(beta) != num_features:
        raise ValueError(f"beta length {len(beta)} does not match num_features {num_features}")

    x = torch.tensor(batch, dtype=torch.float32)  # (B, F)
    mean = x.mean(dim=0)
    variance = x.var(dim=0, unbiased=False)  # biased variance: matches BatchNorm's forward normalization
    normalized = (x - mean) / torch.sqrt(variance + eps)

    gamma_t = torch.tensor(gamma, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    output = normalized * gamma_t + beta_t

    return {
        "batch_size": len(batch),
        "num_features": num_features,
        "input": batch,
        "mean": to_list(mean),
        "variance": to_list(variance),
        "normalized": to_list(normalized),
        "gamma": gamma,
        "beta": beta,
        "output": to_list(output),
        "output_mean": to_list(output.mean(dim=0)),
        "output_std": to_list(output.std(dim=0, unbiased=False)),
    }
