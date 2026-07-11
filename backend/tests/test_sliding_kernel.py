import json

import pytest

from app.engine.cnn.sliding_kernel import sliding_kernel_trace


def test_identity_kernel_picks_center_pixel():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    assert result["output"] == [[5]]


def test_box_blur_averages_correctly():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1 / 9] * 3 for _ in range(3)]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    expected_mean = sum(sum(row) for row in image) / 9
    assert result["output"][0][0] == pytest.approx(expected_mean, abs=1e-5)


def test_edge_detector_kernel_manual_check():
    image = [[0, 0, 0], [10, 10, 10], [0, 0, 0]]
    kernel = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]  # picks (row-1) minus (row)
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    # single output cell at center: value = image[0][1]*1 + image[1][1]*-1 = 0 - 10 = -10
    assert result["output"] == [[-10]]


def test_output_shape_no_padding():
    image = [[i + j for j in range(5)] for i in range(5)]
    kernel = [[1, 0], [0, 1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    assert result["output_shape"] == [4, 4]


def test_stride_2_shrinks_output():
    image = [[i + j for j in range(6)] for i in range(6)]
    kernel = [[1, 0], [0, 1]]
    result = sliding_kernel_trace(image, kernel, stride=2, padding=0)
    assert result["output_shape"] == [3, 3]


def test_padding_grows_output_and_padded_shape():
    image = [[1, 2], [3, 4]]
    kernel = [[1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=1)
    assert result["padded_shape"] == [4, 4]
    assert result["output_shape"] == [4, 4]
    # kernel is a single 1 -> output equals the padded image itself
    assert result["output"] == [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]


def test_step_count_matches_output_size():
    image = [[i + j for j in range(4)] for i in range(4)]
    kernel = [[1, 0], [0, 1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    assert len(result["steps"]) == result["output_shape"][0] * result["output_shape"][1]


def test_step_patch_and_value_are_internally_consistent():
    image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, -1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=0)
    for step in result["steps"]:
        manual_value = sum(
            p * k
            for prow, krow in zip(step["patch"], kernel)
            for p, k in zip(prow, krow)
        )
        assert step["value"] == pytest.approx(manual_value, abs=1e-5)
        assert result["output"][step["output_row"]][step["output_col"]] == pytest.approx(
            step["value"], abs=1e-6
        )


def test_step_input_coordinates_account_for_padding():
    image = [[1, 2], [3, 4]]
    kernel = [[1, 0], [0, 1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=1)
    first_step = result["steps"][0]
    # top-left window starts one row/col into the zero-padding, so original coords are negative
    assert first_step["input_row_start"] == -1
    assert first_step["input_col_start"] == -1


def test_kernel_larger_than_padded_image_raises():
    image = [[1, 2], [3, 4]]
    kernel = [[0] * 5 for _ in range(5)]
    with pytest.raises(ValueError):
        sliding_kernel_trace(image, kernel, stride=1, padding=0)


def test_ragged_image_rows_raise():
    with pytest.raises(ValueError):
        sliding_kernel_trace([[1, 2], [3]], [[1]], stride=1, padding=0)


def test_ragged_kernel_rows_raise():
    with pytest.raises(ValueError):
        sliding_kernel_trace([[1, 2], [3, 4]], [[1, 2], [3]], stride=1, padding=0)


def test_result_is_json_serializable():
    image = [[i + j for j in range(4)] for i in range(4)]
    kernel = [[1, -1], [-1, 1]]
    result = sliding_kernel_trace(image, kernel, stride=1, padding=1)
    json.dumps(result)
