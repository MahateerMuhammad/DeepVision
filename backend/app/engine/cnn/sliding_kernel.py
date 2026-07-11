import torch


def sliding_kernel_trace(
    image: list[list[float]],
    kernel: list[list[float]],
    stride: int = 1,
    padding: int = 0,
) -> dict:
    """Pure geometric convolution trace for the 'Filter Factory' sandbox and
    the stride/padding explorer: no trained network involved, just a
    hand-authored kernel slid across a 2D image. Returns the output feature
    map plus one step per window position (patch, elementwise products, and
    the summed value) so the frontend can animate the sliding window.
    """
    h = len(image)
    w = len(image[0]) if h else 0
    if any(len(row) != w for row in image):
        raise ValueError("image rows must all have the same width")

    kh = len(kernel)
    kw = len(kernel[0]) if kh else 0
    if any(len(row) != kw for row in kernel):
        raise ValueError("kernel rows must all have the same width")

    if kh > h + 2 * padding or kw > w + 2 * padding:
        raise ValueError(
            f"kernel ({kh}x{kw}) is larger than the padded image ({h + 2 * padding}x{w + 2 * padding})"
        )

    image_t = torch.tensor(image, dtype=torch.float32)
    kernel_t = torch.tensor(kernel, dtype=torch.float32)
    padded = torch.nn.functional.pad(image_t, (padding, padding, padding, padding))

    padded_h, padded_w = padded.shape
    out_h = (padded_h - kh) // stride + 1
    out_w = (padded_w - kw) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("stride/kernel combination collapses output to non-positive size")

    output = torch.zeros(out_h, out_w)
    steps = []
    for i in range(out_h):
        for j in range(out_w):
            r0, c0 = i * stride, j * stride
            patch = padded[r0 : r0 + kh, c0 : c0 + kw]
            products = patch * kernel_t
            value = products.sum()
            output[i, j] = value
            steps.append(
                {
                    "output_row": i,
                    "output_col": j,
                    "input_row_start": r0 - padding,
                    "input_col_start": c0 - padding,
                    "patch": patch.round(decimals=6).tolist(),
                    "elementwise_products": products.round(decimals=6).tolist(),
                    "value": round(value.item(), 6),
                }
            )

    return {
        "output": output.round(decimals=6).tolist(),
        "output_shape": [out_h, out_w],
        "padded_shape": [padded_h, padded_w],
        "steps": steps,
    }
