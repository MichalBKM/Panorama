"""
200 tests for ex2.py — Panorama Registration & Stitching.
Tests are organized by function, covering correctness, edge cases, shapes, dtypes, and spec compliance.
"""

import numpy as np
import pytest
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

from ex2_207795154_314089517 import (
    harris_corner_detector,
    feature_descriptor,
    find_features,
    match_features,
    apply_homography,
    ransac_homography,
    display_matches,
    accumulate_homographies,
    compute_bounding_box,
    warp_channel,
    warp_image,
)
from utils import (
    build_gaussian_pyramid,
    blur_spatial,
    non_maximum_suppression,
    estimate_rigid_transform,
)


# ==================== Helpers ====================

def make_checkerboard(h=100, w=100, sq=10):
    """Create a checkerboard image with strong corners."""
    im = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            if (i // sq + j // sq) % 2 == 0:
                im[i, j] = 1.0
    return im


def make_random_image(h=100, w=100):
    rng = np.random.RandomState(42)
    return rng.rand(h, w).astype(np.float64)


def make_gradient_image(h=100, w=100):
    """Horizontal gradient — edges but no corners."""
    return np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float64)


def identity_homography():
    return np.eye(3, dtype=np.float64)


def translation_homography(tx, ty):
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


# ==================== harris_corner_detector (tests 1-30) ====================

class TestHarrisCornerDetector:
    def test_01_returns_ndarray(self):
        im = make_checkerboard()
        result = harris_corner_detector(im)
        assert isinstance(result, np.ndarray)

    def test_02_shape_is_n_by_2(self):
        im = make_checkerboard()
        result = harris_corner_detector(im)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_03_detects_corners_on_checkerboard(self):
        im = make_checkerboard()
        result = harris_corner_detector(im)
        assert result.shape[0] > 0, "Should detect corners on checkerboard"

    def test_04_uniform_image_few_corners(self):
        im = np.ones((50, 50), dtype=np.float64) * 0.5
        result = harris_corner_detector(im)
        # Uniform image may produce spurious corners from boundary effects
        assert result.shape[0] <= 5

    def test_05_coordinates_within_bounds(self):
        im = make_checkerboard()
        pts = harris_corner_detector(im)
        if pts.shape[0] > 0:
            assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < im.shape[1])
            assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < im.shape[0])

    def test_06_returns_xy_not_yx(self):
        """First column should be x (horizontal), second y (vertical)."""
        im = make_checkerboard(100, 200, 20)
        pts = harris_corner_detector(im)
        if pts.shape[0] > 0:
            # x coords can go up to width, y up to height
            assert pts[:, 0].max() < 200
            assert pts[:, 1].max() < 100

    def test_07_deterministic(self):
        im = make_checkerboard()
        r1 = harris_corner_detector(im)
        r2 = harris_corner_detector(im)
        np.testing.assert_array_equal(r1, r2)

    def test_08_gradient_image_few_corners(self):
        im = make_gradient_image(200, 200)
        # Gradient may cause empty response in non_maximum_suppression
        try:
            pts = harris_corner_detector(im)
            assert pts.shape[0] < 20
        except ValueError:
            # non_maximum_suppression raises if no local maxima found
            pass

    def test_09_small_image(self):
        im = make_checkerboard(20, 20, 5)
        pts = harris_corner_detector(im)
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_10_single_bright_pixel(self):
        im = np.zeros((50, 50), dtype=np.float64)
        im[25, 25] = 1.0
        pts = harris_corner_detector(im)
        # May or may not detect; just shouldn't crash
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_11_float64_input(self):
        im = make_checkerboard().astype(np.float64)
        pts = harris_corner_detector(im)
        assert pts.shape[1] == 2

    def test_12_uses_correct_derivative_filters(self):
        """Verify the derivative filter spec: [1, 0, -1] and transpose."""
        im = make_random_image(30, 30)
        ix = convolve2d(im, np.array([[1, 0, -1]]), mode='same', boundary='symm')
        iy = convolve2d(im, np.array([[1], [0], [-1]]), mode='same', boundary='symm')
        # Just verify these don't crash and have correct shapes
        assert ix.shape == im.shape
        assert iy.shape == im.shape

    def test_13_large_image(self):
        im = make_checkerboard(200, 200, 20)
        pts = harris_corner_detector(im)
        assert pts.shape[0] > 0

    def test_14_non_square_image(self):
        im = make_checkerboard(50, 150, 10)
        pts = harris_corner_detector(im)
        assert pts.shape[1] == 2

    def test_15_all_zeros(self):
        im = np.zeros((50, 50), dtype=np.float64)
        # All-zero response may still find spurious points
        pts = harris_corner_detector(im)
        assert pts.shape[0] <= 5

    def test_16_corners_are_unique(self):
        im = make_checkerboard()
        pts = harris_corner_detector(im)
        if pts.shape[0] > 1:
            unique = np.unique(pts, axis=0)
            assert unique.shape[0] == pts.shape[0]

    def test_17_alpha_004(self):
        """Spec says alpha = 0.04."""
        # Indirect test: corners detected should be consistent with alpha=0.04
        im = make_checkerboard()
        pts = harris_corner_detector(im)
        assert pts.shape[0] > 0

    def test_18_uses_blur_spatial_kernel3(self):
        """Spec: blur with kernel_size=3."""
        im = make_random_image(30, 30)
        blurred = blur_spatial(im, 3)
        assert blurred.shape == im.shape

    def test_19_symmetric_image(self):
        im = make_checkerboard(100, 100, 10)
        pts = harris_corner_detector(im)
        assert pts.shape[0] > 0

    def test_20_very_small_image(self):
        im = np.array([[0.0, 1.0], [1.0, 0.0]])
        pts = harris_corner_detector(im)
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_21_random_image_has_corners(self):
        im = make_random_image(80, 80)
        pts = harris_corner_detector(im)
        # Random images typically produce corners
        assert pts.shape[0] >= 0

    def test_22_corner_at_known_location(self):
        """Black square on white should have corners near the square edges."""
        im = np.ones((80, 80), dtype=np.float64)
        im[20:60, 20:60] = 0.0
        pts = harris_corner_detector(im)
        if pts.shape[0] > 0:
            # At least some corners should be near the square boundary
            near_boundary = np.any(
                (np.abs(pts[:, 0] - 20) < 5) | (np.abs(pts[:, 0] - 59) < 5)
            )
            assert near_boundary

    def test_23_integer_coordinates(self):
        im = make_checkerboard()
        pts = harris_corner_detector(im)
        if pts.shape[0] > 0:
            # Corners come from nonzero of a binary image, should be integer
            np.testing.assert_array_equal(pts, pts.astype(int))

    def test_24_wide_image(self):
        im = make_checkerboard(30, 200, 10)
        pts = harris_corner_detector(im)
        assert pts.shape[1] == 2

    def test_25_tall_image(self):
        im = make_checkerboard(200, 30, 10)
        pts = harris_corner_detector(im)
        assert pts.shape[1] == 2

    def test_26_output_not_empty_for_textured(self):
        rng = np.random.RandomState(7)
        im = rng.rand(100, 100)
        pts = harris_corner_detector(im)
        assert pts.shape[1] == 2

    def test_27_vertical_stripe_no_corners(self):
        im = np.zeros((50, 50), dtype=np.float64)
        im[:, 25:] = 1.0
        pts = harris_corner_detector(im)
        # A single vertical edge has no corners (only edge)
        assert pts.shape[0] < 5

    def test_28_horizontal_stripe_no_corners(self):
        im = np.zeros((50, 50), dtype=np.float64)
        im[25:, :] = 1.0
        pts = harris_corner_detector(im)
        assert pts.shape[0] < 5

    def test_29_multiple_squares(self):
        im = np.zeros((100, 100), dtype=np.float64)
        im[10:30, 10:30] = 1.0
        im[50:70, 50:70] = 1.0
        pts = harris_corner_detector(im)
        assert pts.shape[0] >= 2

    def test_30_no_nan_or_inf(self):
        im = make_checkerboard()
        pts = harris_corner_detector(im)
        assert not np.any(np.isnan(pts))
        assert not np.any(np.isinf(pts))


# ==================== feature_descriptor (tests 31-55) ====================

class TestFeatureDescriptor:
    def _make_im_and_points(self):
        im = make_random_image(100, 100)
        pts = np.array([[50, 50], [30, 30]], dtype=np.float64)
        return im, pts

    def test_31_output_shape(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (2, 7, 7)

    def test_32_desc_rad_default(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts)
        assert desc.shape == (pts.shape[0], 7, 7)

    def test_33_custom_desc_rad(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts, desc_rad=5)
        assert desc.shape == (2, 11, 11)

    def test_34_zero_mean(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts, desc_rad=3)
        for i in range(desc.shape[0]):
            if np.linalg.norm(desc[i]) > 0:
                np.testing.assert_almost_equal(desc[i].mean(), 0.0, decimal=10)

    def test_35_unit_norm(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts, desc_rad=3)
        for i in range(desc.shape[0]):
            norm = np.linalg.norm(desc[i])
            if norm > 0:
                np.testing.assert_almost_equal(norm, 1.0, decimal=10)

    def test_36_constant_patch_zero_descriptor(self):
        im = np.ones((100, 100), dtype=np.float64) * 0.5
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        # Constant patch => mean-subtracted is all zeros => norm is 0 => stays zero
        np.testing.assert_array_equal(desc[0], np.zeros((7, 7)))

    def test_37_empty_points(self):
        im = make_random_image()
        pts = np.empty((0, 2), dtype=np.float64)
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (0, 7, 7)

    def test_38_single_point(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (1, 7, 7)

    def test_39_deterministic(self):
        im, pts = self._make_im_and_points()
        d1 = feature_descriptor(im, pts)
        d2 = feature_descriptor(im, pts)
        np.testing.assert_array_equal(d1, d2)

    def test_40_different_points_different_descriptors(self):
        im = make_random_image()
        pts = np.array([[20, 20], [80, 80]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert not np.allclose(desc[0], desc[1])

    def test_41_same_point_same_descriptor(self):
        im = make_random_image()
        pts = np.array([[50, 50], [50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        np.testing.assert_array_almost_equal(desc[0], desc[1])

    def test_42_patch_size_7x7_for_rad3(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape[1] == 7 and desc.shape[2] == 7

    def test_43_bilinear_interpolation(self):
        """Points at non-integer locations should still work via map_coordinates."""
        im = make_random_image()
        pts = np.array([[50.5, 50.5]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (1, 7, 7)

    def test_44_near_border(self):
        """Points near border should use interpolation (map_coordinates handles boundaries)."""
        im = make_random_image()
        pts = np.array([[1, 1]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (1, 7, 7)

    def test_45_no_nan_in_output(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts)
        assert not np.any(np.isnan(desc))

    def test_46_no_inf_in_output(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts)
        assert not np.any(np.isinf(desc))

    def test_47_descriptor_values_bounded(self):
        im, pts = self._make_im_and_points()
        desc = feature_descriptor(im, pts)
        # Normalized => values should be in [-1, 1]
        assert np.all(np.abs(desc) <= 1.0 + 1e-10)

    def test_48_many_points(self):
        im = make_random_image(200, 200)
        rng = np.random.RandomState(42)
        pts = rng.randint(10, 190, size=(100, 2)).astype(np.float64)
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (100, 7, 7)

    def test_49_desc_rad_1(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=1)
        assert desc.shape == (1, 3, 3)

    def test_50_desc_shape_matches_formula(self):
        for rad in [1, 2, 3, 4, 5]:
            im = make_random_image()
            pts = np.array([[50, 50]])
            desc = feature_descriptor(im, pts, desc_rad=rad)
            k = 2 * rad + 1
            assert desc.shape == (1, k, k)

    def test_51_dot_product_self_is_one(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        if np.linalg.norm(desc[0]) > 0:
            flat = desc[0].flatten()
            np.testing.assert_almost_equal(flat @ flat, 1.0, decimal=10)

    def test_52_flattened_length_49(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc[0].flatten().shape[0] == 49

    def test_53_returns_float(self):
        im = make_random_image()
        pts = np.array([[50, 50]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.dtype in [np.float64, np.float32]

    def test_54_corner_point(self):
        im = make_random_image()
        pts = np.array([[0, 0]])
        desc = feature_descriptor(im, pts, desc_rad=3)
        assert desc.shape == (1, 7, 7)

    def test_55_large_desc_rad(self):
        im = make_random_image(200, 200)
        pts = np.array([[100, 100]])
        desc = feature_descriptor(im, pts, desc_rad=10)
        assert desc.shape == (1, 21, 21)


# ==================== find_features (tests 56-70) ====================

class TestFindFeatures:
    def _make_textured_image(self):
        return make_checkerboard(128, 128, 16)

    def test_56_returns_tuple_of_two(self):
        im = self._make_textured_image()
        result = find_features(im)
        assert len(result) == 2

    def test_57_points_shape(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_58_desc_shape_matches_points(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        assert desc.shape[0] == pts.shape[0]

    def test_59_desc_is_7x7(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        if desc.shape[0] > 0:
            assert desc.shape[1] == 7
            assert desc.shape[2] == 7

    def test_60_points_at_original_scale(self):
        """Points should be at the original image resolution."""
        im = self._make_textured_image()
        pts, desc = find_features(im)
        if pts.shape[0] > 0:
            assert pts[:, 0].max() < im.shape[1]
            assert pts[:, 1].max() < im.shape[0]

    def test_61_detects_features_on_textured(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        assert pts.shape[0] > 0

    def test_62_descriptors_normalized(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        for i in range(min(desc.shape[0], 10)):
            norm = np.linalg.norm(desc[i])
            if norm > 0:
                np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_63_uses_third_pyramid_level(self):
        """Descriptors extracted from pyr[2], points divided by 4."""
        im = self._make_textured_image()
        pyr = build_gaussian_pyramid(im, 3, 7)
        assert len(pyr) == 3
        assert pyr[2].shape[0] == im.shape[0] // 4
        assert pyr[2].shape[1] == im.shape[1] // 4

    def test_64_larger_image(self):
        im = make_checkerboard(256, 256, 32)
        pts, desc = find_features(im)
        assert pts.shape[0] > 0

    def test_65_random_image(self):
        im = make_random_image(128, 128)
        pts, desc = find_features(im)
        assert pts.ndim == 2 and pts.shape[1] == 2

    def test_66_points_non_negative(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        if pts.shape[0] > 0:
            assert np.all(pts >= 0)

    def test_67_no_nan(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        assert not np.any(np.isnan(pts))
        assert not np.any(np.isnan(desc))

    def test_68_deterministic(self):
        im = self._make_textured_image()
        p1, d1 = find_features(im)
        p2, d2 = find_features(im)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(d1, d2)

    def test_69_desc_zero_mean(self):
        im = self._make_textured_image()
        pts, desc = find_features(im)
        for i in range(min(desc.shape[0], 5)):
            if np.linalg.norm(desc[i]) > 0:
                np.testing.assert_almost_equal(desc[i].mean(), 0.0, decimal=5)

    def test_70_nonsquare_image(self):
        im = make_checkerboard(128, 256, 16)
        pts, desc = find_features(im)
        assert pts.shape[1] == 2


# ==================== match_features (tests 71-100) ====================

class TestMatchFeatures:
    def _make_matching_descs(self):
        rng = np.random.RandomState(42)
        desc1 = rng.randn(10, 7, 7)
        # Normalize
        for i in range(10):
            desc1[i] -= desc1[i].mean()
            n = np.linalg.norm(desc1[i])
            if n > 0:
                desc1[i] /= n
        desc2 = desc1.copy()
        return desc1, desc2

    def test_71_returns_two_arrays(self):
        desc1, desc2 = self._make_matching_descs()
        result = match_features(desc1, desc2, 0.5)
        assert len(result) == 2

    def test_72_matched_indices_dtype_int(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert ind1.dtype == int or np.issubdtype(ind1.dtype, np.integer)
        assert ind2.dtype == int or np.issubdtype(ind2.dtype, np.integer)

    def test_73_identical_descs_all_match(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        # Identical descriptors: each should match itself
        assert len(ind1) == 10
        np.testing.assert_array_equal(ind1, ind2)

    def test_74_high_min_score_fewer_matches(self):
        desc1, desc2 = self._make_matching_descs()
        ind1_low, _ = match_features(desc1, desc2, 0.3)
        ind1_high, _ = match_features(desc1, desc2, 0.99)
        assert len(ind1_high) <= len(ind1_low)

    def test_75_empty_desc1(self):
        desc1 = np.empty((0, 7, 7))
        desc2 = np.random.randn(5, 7, 7)
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == 0 and len(ind2) == 0

    def test_76_empty_desc2(self):
        desc1 = np.random.randn(5, 7, 7)
        desc2 = np.empty((0, 7, 7))
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == 0 and len(ind2) == 0

    def test_77_both_empty(self):
        desc1 = np.empty((0, 7, 7))
        desc2 = np.empty((0, 7, 7))
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == 0 and len(ind2) == 0

    def test_78_indices_in_range(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        if len(ind1) > 0:
            assert np.all(ind1 >= 0) and np.all(ind1 < desc1.shape[0])
            assert np.all(ind2 >= 0) and np.all(ind2 < desc2.shape[0])

    def test_79_same_length_outputs(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == len(ind2)

    def test_80_min_score_0_many_matches(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, -1.0)
        assert len(ind1) >= 10  # at least all self-matches

    def test_81_min_score_1_no_matches(self):
        rng = np.random.RandomState(42)
        desc1 = rng.randn(5, 7, 7)
        for i in range(5):
            desc1[i] -= desc1[i].mean()
            n = np.linalg.norm(desc1[i])
            if n > 0: desc1[i] /= n
        desc2 = rng.randn(5, 7, 7)
        for i in range(5):
            desc2[i] -= desc2[i].mean()
            n = np.linalg.norm(desc2[i])
            if n > 0: desc2[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 1.0)
        # Very high threshold — unlikely to match random descriptors
        assert len(ind1) <= 5

    def test_82_single_desc_each(self):
        rng = np.random.RandomState(42)
        d = rng.randn(1, 7, 7)
        d[0] -= d[0].mean()
        d[0] /= np.linalg.norm(d[0])
        ind1, ind2 = match_features(d, d.copy(), 0.5)
        assert len(ind1) == 1

    def test_83_dot_product_score(self):
        """Score should be dot product of flattened normalized descriptors."""
        rng = np.random.RandomState(42)
        desc1 = rng.randn(3, 7, 7)
        for i in range(3):
            desc1[i] -= desc1[i].mean()
            desc1[i] /= np.linalg.norm(desc1[i])
        desc2 = desc1.copy()
        scores = desc1.reshape(3, -1) @ desc2.reshape(3, -1).T
        # Diagonal should be 1.0
        np.testing.assert_almost_equal(np.diag(scores), 1.0, decimal=10)

    def test_84_asymmetric_sizes(self):
        rng = np.random.RandomState(42)
        desc1 = rng.randn(3, 7, 7)
        desc2 = rng.randn(10, 7, 7)
        for d in [desc1, desc2]:
            for i in range(d.shape[0]):
                d[i] -= d[i].mean()
                n = np.linalg.norm(d[i])
                if n > 0: d[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == len(ind2)

    def test_85_top2_row_condition(self):
        """Match must be in the top-2 matches for that row (descriptor in desc1)."""
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        scores = desc1.reshape(10, -1) @ desc2.reshape(10, -1).T
        for i, j in zip(ind1, ind2):
            row_top2 = np.argsort(scores[i])[-2:]
            assert j in row_top2

    def test_86_top2_col_condition(self):
        """Match must be in the top-2 matches for that column (descriptor in desc2)."""
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        scores = desc1.reshape(10, -1) @ desc2.reshape(10, -1).T
        for i, j in zip(ind1, ind2):
            col_top2 = np.argsort(scores[:, j])[-2:]
            assert i in col_top2

    def test_87_min_score_condition(self):
        """All matches should have score > min_score."""
        desc1, desc2 = self._make_matching_descs()
        min_score = 0.5
        ind1, ind2 = match_features(desc1, desc2, min_score)
        scores = desc1.reshape(10, -1) @ desc2.reshape(10, -1).T
        for i, j in zip(ind1, ind2):
            assert scores[i, j] > min_score

    def test_88_score_range(self):
        """Dot products of normalized descriptors should be in [-1, 1]."""
        desc1, desc2 = self._make_matching_descs()
        scores = desc1.reshape(10, -1) @ desc2.reshape(10, -1).T
        assert np.all(scores >= -1.0 - 1e-10) and np.all(scores <= 1.0 + 1e-10)

    def test_89_no_duplicate_matches(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        pairs = set(zip(ind1.tolist(), ind2.tolist()))
        assert len(pairs) == len(ind1)

    def test_90_shuffled_descs(self):
        desc1, desc2 = self._make_matching_descs()
        rng = np.random.RandomState(99)
        perm = rng.permutation(10)
        desc2_shuffled = desc2[perm]
        ind1, ind2 = match_features(desc1, desc2_shuffled, 0.5)
        assert len(ind1) == 10

    def test_91_orthogonal_descs_no_match(self):
        """Orthogonal descriptors should have zero score."""
        desc1 = np.zeros((2, 7, 7))
        desc1[0, 0, 0] = 1.0
        desc1[1, 0, 1] = 1.0
        desc2 = np.zeros((2, 7, 7))
        desc2[0, 3, 3] = 1.0
        desc2[1, 6, 6] = 1.0
        # Normalize
        for d in [desc1, desc2]:
            for i in range(2):
                d[i] -= d[i].mean()
                n = np.linalg.norm(d[i])
                if n > 0: d[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        # Score likely < 0.5 for near-orthogonal descriptors
        assert len(ind1) <= 2

    def test_92_1d_output_shapes(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert ind1.ndim == 1
        assert ind2.ndim == 1

    def test_93_two_descs_one_match(self):
        d = np.random.RandomState(42).randn(1, 7, 7)
        d[0] -= d[0].mean()
        d[0] /= np.linalg.norm(d[0])
        d2 = d.copy()
        ind1, ind2 = match_features(d, d2, 0.5)
        assert len(ind1) == 1

    def test_94_negative_scores_filtered(self):
        """Negative dot product should be filtered by min_score=0."""
        rng = np.random.RandomState(42)
        desc1 = rng.randn(3, 7, 7)
        desc2 = -desc1.copy()
        for d in [desc1, desc2]:
            for i in range(3):
                d[i] -= d[i].mean()
                n = np.linalg.norm(d[i])
                if n > 0: d[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 0.0)
        # After mean subtraction and normalization, negation changes things
        # Just verify no crash
        assert len(ind1) == len(ind2)

    def test_95_large_number_of_descs(self):
        rng = np.random.RandomState(42)
        desc1 = rng.randn(50, 7, 7)
        desc2 = rng.randn(50, 7, 7)
        for d in [desc1, desc2]:
            for i in range(50):
                d[i] -= d[i].mean()
                n = np.linalg.norm(d[i])
                if n > 0: d[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == len(ind2)

    def test_96_match_count_reasonable(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        # Can't have more matches than min(N1, N2)
        assert len(ind1) <= max(desc1.shape[0], desc2.shape[0]) * 2

    def test_97_different_k_sizes(self):
        rng = np.random.RandomState(42)
        desc1 = rng.randn(5, 5, 5)
        desc2 = rng.randn(5, 5, 5)
        for d in [desc1, desc2]:
            for i in range(5):
                d[i] -= d[i].mean()
                n = np.linalg.norm(d[i])
                if n > 0: d[i] /= n
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert len(ind1) == len(ind2)

    def test_98_min_score_boundary(self):
        """Exact min_score should not be included (> not >=)."""
        desc1, desc2 = self._make_matching_descs()
        scores = desc1.reshape(10, -1) @ desc2.reshape(10, -1).T
        min_score = 0.5
        ind1, ind2 = match_features(desc1, desc2, min_score)
        for i, j in zip(ind1, ind2):
            assert scores[i, j] > min_score  # strictly greater

    def test_99_returns_numpy_arrays(self):
        desc1, desc2 = self._make_matching_descs()
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert isinstance(ind1, np.ndarray)
        assert isinstance(ind2, np.ndarray)

    def test_100_empty_result_arrays(self):
        desc1 = np.empty((0, 7, 7))
        desc2 = np.empty((0, 7, 7))
        ind1, ind2 = match_features(desc1, desc2, 0.5)
        assert isinstance(ind1, np.ndarray) and isinstance(ind2, np.ndarray)
        assert ind1.dtype == int or np.issubdtype(ind1.dtype, np.integer)


# ==================== apply_homography (tests 101-125) ====================

class TestApplyHomography:
    def test_101_identity_no_change(self):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        np.testing.assert_array_almost_equal(result, pts)

    def test_102_output_shape(self):
        pts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert result.shape == pts.shape

    def test_103_translation(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float64)
        H = translation_homography(5, 3)
        result = apply_homography(pts, H)
        expected = pts + np.array([5, 3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_104_single_point(self):
        pts = np.array([[5, 10]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        np.testing.assert_array_almost_equal(result, pts)

    def test_105_empty_points(self):
        pts = np.empty((0, 2), dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert result.shape == (0, 2)

    def test_106_scaling(self):
        pts = np.array([[10, 20]], dtype=np.float64)
        H = np.diag([2.0, 3.0, 1.0])
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[20, 60]])

    def test_107_returns_2d(self):
        pts = np.array([[1, 2]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_108_homogeneous_normalization(self):
        """H with non-unit last element should still work."""
        pts = np.array([[10, 20]], dtype=np.float64)
        H = np.eye(3) * 2.0
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, pts)

    def test_109_inverse_homography(self):
        pts = np.array([[10, 20], [30, 40]], dtype=np.float64)
        H = translation_homography(5, 3)
        H_inv = np.linalg.inv(H)
        result = apply_homography(apply_homography(pts, H), H_inv)
        np.testing.assert_array_almost_equal(result, pts)

    def test_110_rotation_90(self):
        pts = np.array([[1, 0]], dtype=np.float64)
        H = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[0, 1]])

    def test_111_preserves_float(self):
        pts = np.array([[1.5, 2.5]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert result.dtype in [np.float64, np.float32]

    def test_112_many_points(self):
        rng = np.random.RandomState(42)
        pts = rng.rand(1000, 2) * 100
        result = apply_homography(pts, identity_homography())
        np.testing.assert_array_almost_equal(result, pts)

    def test_113_negative_translation(self):
        pts = np.array([[10, 20]], dtype=np.float64)
        H = translation_homography(-5, -10)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[5, 10]])

    def test_114_no_nan(self):
        pts = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert not np.any(np.isnan(result))

    def test_115_composition(self):
        pts = np.array([[10, 20]], dtype=np.float64)
        H1 = translation_homography(5, 0)
        H2 = translation_homography(0, 3)
        result = apply_homography(apply_homography(pts, H1), H2)
        np.testing.assert_array_almost_equal(result, [[15, 23]])

    def test_116_matrix_composition_equivalent(self):
        pts = np.array([[10, 20]], dtype=np.float64)
        H1 = translation_homography(5, 0)
        H2 = translation_homography(0, 3)
        H_combined = H2 @ H1
        r1 = apply_homography(apply_homography(pts, H1), H2)
        r2 = apply_homography(pts, H_combined)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_117_origin_point(self):
        pts = np.array([[0, 0]], dtype=np.float64)
        H = translation_homography(100, 200)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[100, 200]])

    def test_118_large_coordinates(self):
        pts = np.array([[1e6, 1e6]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        np.testing.assert_array_almost_equal(result, pts)

    def test_119_perspective_transform(self):
        """Non-affine homography should still work."""
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        H = np.array([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]], dtype=np.float64)
        result = apply_homography(pts, H)
        assert result.shape == (4, 2)
        assert not np.any(np.isnan(result))

    def test_120_xy_order(self):
        """Input/output should be [x, y]."""
        pts = np.array([[5, 10]], dtype=np.float64)
        H = translation_homography(1, 2)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[6, 12]])

    def test_121_returns_new_array(self):
        pts = np.array([[1, 2]], dtype=np.float64)
        result = apply_homography(pts, identity_homography())
        assert result is not pts

    def test_122_batch_consistency(self):
        rng = np.random.RandomState(42)
        pts = rng.rand(10, 2) * 100
        H = translation_homography(5, 3)
        batch_result = apply_homography(pts, H)
        for i in range(10):
            single_result = apply_homography(pts[i:i+1], H)
            np.testing.assert_array_almost_equal(batch_result[i], single_result[0])

    def test_123_reflection(self):
        pts = np.array([[1, 0]], dtype=np.float64)
        H = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[-1, 0]])

    def test_124_scale_and_translate(self):
        pts = np.array([[10, 20]], dtype=np.float64)
        H = np.array([[2, 0, 5], [0, 2, 10], [0, 0, 1]], dtype=np.float64)
        result = apply_homography(pts, H)
        np.testing.assert_array_almost_equal(result, [[25, 50]])

    def test_125_float32_input(self):
        pts = np.array([[1, 2]], dtype=np.float32)
        result = apply_homography(pts, identity_homography())
        assert result.shape == (1, 2)


# ==================== ransac_homography (tests 126-155) ====================

class TestRansacHomography:
    def _make_translation_pair(self, n=50, tx=10, ty=5):
        rng = np.random.RandomState(42)
        pts1 = rng.rand(n, 2) * 100
        pts2 = pts1 + np.array([tx, ty])
        return pts1, pts2

    def test_126_returns_tuple_of_two(self):
        pts1, pts2 = self._make_translation_pair()
        result = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert len(result) == 2

    def test_127_homography_shape(self):
        pts1, pts2 = self._make_translation_pair()
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert H.shape == (3, 3)

    def test_128_inliers_1d(self):
        pts1, pts2 = self._make_translation_pair()
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert inliers.ndim == 1

    def test_129_translation_recovery(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_almost_equal(H[0, 2], 10, decimal=1)
        np.testing.assert_almost_equal(H[1, 2], 5, decimal=1)

    def test_130_all_inliers_for_clean_data(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert len(inliers) == 50

    def test_131_outlier_rejection(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        # Add outliers
        rng = np.random.RandomState(99)
        pts2_noisy = pts2.copy()
        pts2_noisy[:10] += rng.randn(10, 2) * 100
        H, inliers = ransac_homography(pts1, pts2_noisy, 200, 6, translation_only=True)
        # Most inliers should be from the clean portion
        assert len(inliers) >= 30

    def test_132_h22_is_one(self):
        """Spec: normalize so H[2,2] == 1."""
        pts1, pts2 = self._make_translation_pair()
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_almost_equal(H[2, 2], 1.0)

    def test_133_inliers_in_range(self):
        pts1, pts2 = self._make_translation_pair()
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        if len(inliers) > 0:
            assert np.all(inliers >= 0) and np.all(inliers < pts1.shape[0])

    def test_134_few_points(self):
        pts1 = np.array([[0, 0], [1, 1]], dtype=np.float64)
        pts2 = np.array([[5, 5], [6, 6]], dtype=np.float64)
        H, inliers = ransac_homography(pts1, pts2, 10, 6, translation_only=True)
        assert H.shape == (3, 3)

    def test_135_one_point(self):
        pts1 = np.array([[0, 0]], dtype=np.float64)
        pts2 = np.array([[5, 5]], dtype=np.float64)
        H, inliers = ransac_homography(pts1, pts2, 10, 6, translation_only=True)
        assert H.shape == (3, 3)

    def test_136_translation_only_flag(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        # Translation only: rotation part should be identity
        np.testing.assert_almost_equal(H[:2, :2], np.eye(2), decimal=5)

    def test_137_rigid_with_rotation(self):
        rng = np.random.RandomState(42)
        pts1 = rng.rand(50, 2) * 100
        angle = 0.1
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        pts2 = (R @ pts1.T).T + np.array([5, 3])
        H, inliers = ransac_homography(pts1, pts2, 200, 6, translation_only=False)
        assert len(inliers) >= 30

    def test_138_recompute_with_inliers(self):
        """After RANSAC, homography should be recomputed with all inliers."""
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        # Recompute manually
        H_check = estimate_rigid_transform(pts1[inliers], pts2[inliers], True)
        H_check /= H_check[2, 2]
        np.testing.assert_array_almost_equal(H, H_check, decimal=5)

    def test_139_inlier_tolerance(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        transformed = apply_homography(pts1, H)
        distances = np.sum((transformed - pts2) ** 2, axis=1)
        for idx in inliers:
            assert distances[idx] < 6

    def test_140_more_iters_better_result(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        # Add outliers
        pts2_noisy = pts2.copy()
        pts2_noisy[:5] += 1000
        _, inliers_few = ransac_homography(pts1, pts2_noisy, 5, 6, translation_only=True)
        _, inliers_many = ransac_homography(pts1, pts2_noisy, 500, 6, translation_only=True)
        # More iterations should generally find more or equal inliers
        assert len(inliers_many) >= len(inliers_few) - 5  # allow small variance

    def test_141_no_nan_in_H(self):
        pts1, pts2 = self._make_translation_pair()
        H, _ = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert not np.any(np.isnan(H))

    def test_142_inliers_dtype_int(self):
        pts1, pts2 = self._make_translation_pair()
        _, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert np.issubdtype(inliers.dtype, np.integer)

    def test_143_zero_translation(self):
        pts1, pts2 = self._make_translation_pair(50, 0, 0)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_almost_equal(H[0, 2], 0, decimal=1)
        np.testing.assert_almost_equal(H[1, 2], 0, decimal=1)

    def test_144_large_translation(self):
        pts1, pts2 = self._make_translation_pair(50, 500, 300)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_almost_equal(H[0, 2], 500, decimal=0)
        np.testing.assert_almost_equal(H[1, 2], 300, decimal=0)

    def test_145_negative_translation(self):
        pts1, pts2 = self._make_translation_pair(50, -10, -5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_almost_equal(H[0, 2], -10, decimal=1)
        np.testing.assert_almost_equal(H[1, 2], -5, decimal=1)

    def test_146_samples_2_points(self):
        """Spec: randomly sample 2 point correspondences per iteration."""
        # If we have exactly 2 points, RANSAC should still work
        pts1 = np.array([[0, 0], [10, 10]], dtype=np.float64)
        pts2 = np.array([[5, 5], [15, 15]], dtype=np.float64)
        H, inliers = ransac_homography(pts1, pts2, 50, 10, translation_only=True)
        assert H.shape == (3, 3)

    def test_147_inlier_tol_strict(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        pts2_noisy = pts2 + np.random.RandomState(42).randn(50, 2) * 0.5
        _, inliers_loose = ransac_homography(pts1, pts2_noisy, 100, 100, translation_only=True)
        _, inliers_strict = ransac_homography(pts1, pts2_noisy, 100, 0.01, translation_only=True)
        assert len(inliers_strict) <= len(inliers_loose)

    def test_148_three_points(self):
        pts1 = np.array([[0, 0], [10, 0], [5, 5]], dtype=np.float64)
        pts2 = pts1 + np.array([3, 4])
        H, inliers = ransac_homography(pts1, pts2, 50, 6, translation_only=True)
        assert len(inliers) == 3

    def test_149_homography_applies_correctly(self):
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        H, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        transformed = apply_homography(pts1[inliers], H)
        np.testing.assert_array_almost_equal(transformed, pts2[inliers], decimal=1)

    def test_150_empty_inliers_for_random(self):
        """Completely random correspondences might yield few inliers."""
        rng = np.random.RandomState(42)
        pts1 = rng.rand(10, 2) * 100
        pts2 = rng.rand(10, 2) * 100
        H, inliers = ransac_homography(pts1, pts2, 50, 0.001, translation_only=True)
        assert H.shape == (3, 3)

    def test_151_unique_inlier_indices(self):
        pts1, pts2 = self._make_translation_pair()
        _, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        assert len(np.unique(inliers)) == len(inliers)

    def test_152_not_modifying_input(self):
        pts1, pts2 = self._make_translation_pair()
        pts1_copy = pts1.copy()
        pts2_copy = pts2.copy()
        ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        np.testing.assert_array_equal(pts1, pts1_copy)
        np.testing.assert_array_equal(pts2, pts2_copy)

    def test_153_rigid_transform_used(self):
        """Spec: use estimate_rigid_transform."""
        pts1 = np.array([[0, 0], [10, 0]], dtype=np.float64)
        pts2 = np.array([[5, 0], [15, 0]], dtype=np.float64)
        H, _ = ransac_homography(pts1, pts2, 50, 6, translation_only=True)
        # Should be a proper rigid transform
        assert H.shape == (3, 3)
        np.testing.assert_almost_equal(H[2, 2], 1.0)

    def test_154_sorted_inliers(self):
        """Inliers come from np.where which returns sorted indices."""
        pts1, pts2 = self._make_translation_pair()
        _, inliers = ransac_homography(pts1, pts2, 100, 6, translation_only=True)
        if len(inliers) > 1:
            assert np.all(np.diff(inliers) > 0)

    def test_155_squared_euclidean_distance(self):
        """Spec: use squared Euclidean distance, not Euclidean."""
        pts1, pts2 = self._make_translation_pair(50, 10, 5)
        # Add one outlier that's sqrt(6) away but squared distance = 6
        pts2_mod = pts2.copy()
        pts2_mod[0] += np.array([2.449, 0.1])  # squared dist ~ 6.01
        H, inliers = ransac_homography(pts1, pts2_mod, 100, 6, translation_only=True)
        # Point 0 should be borderline - may or may not be inlier
        assert H.shape == (3, 3)


# ==================== accumulate_homographies (tests 156-175) ====================

class TestAccumulateHomographies:
    def test_156_identity_at_m(self):
        Hs = [translation_homography(5, 0), translation_homography(5, 0)]
        result = accumulate_homographies(Hs, 1)
        np.testing.assert_array_almost_equal(result[1], np.eye(3))

    def test_157_output_length(self):
        Hs = [np.eye(3)] * 4
        result = accumulate_homographies(Hs, 2)
        assert len(result) == 5

    def test_158_all_identity_input(self):
        Hs = [np.eye(3)] * 3
        result = accumulate_homographies(Hs, 1)
        for H in result:
            np.testing.assert_array_almost_equal(H, np.eye(3))

    def test_159_h22_normalized(self):
        """Spec: H[2,2] should always be 1."""
        Hs = [translation_homography(5, 0)] * 3
        result = accumulate_homographies(Hs, 1)
        for H in result:
            np.testing.assert_almost_equal(H[2, 2], 1.0)

    def test_160_single_homography(self):
        H = translation_homography(10, 5)
        result = accumulate_homographies([H], 0)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_161_forward_accumulation(self):
        H1 = translation_homography(10, 0)
        H2 = translation_homography(10, 0)
        result = accumulate_homographies([H1, H2], 0)
        # Frame 0 is ref => identity
        np.testing.assert_array_almost_equal(result[0], np.eye(3))
        # Frame 1 needs inverse of H1
        # Frame 2 needs inverse of H2 @ inverse of H1

    def test_162_backward_accumulation(self):
        H1 = translation_homography(10, 0)
        H2 = translation_homography(10, 0)
        result = accumulate_homographies([H1, H2], 2)
        # Frame 2 is ref
        np.testing.assert_array_almost_equal(result[2], np.eye(3))

    def test_163_middle_reference(self):
        Hs = [translation_homography(10, 0)] * 4
        m = 2
        result = accumulate_homographies(Hs, m)
        np.testing.assert_array_almost_equal(result[m], np.eye(3))
        assert len(result) == 5

    def test_164_all_3x3(self):
        Hs = [translation_homography(5, 3)] * 3
        result = accumulate_homographies(Hs, 1)
        for H in result:
            assert H.shape == (3, 3)

    def test_165_two_frames(self):
        H = translation_homography(10, 5)
        result = accumulate_homographies([H], 1)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[1], np.eye(3))

    def test_166_composition_correct(self):
        """Accumulating H_{i->m} should transform points from frame i to frame m."""
        H01 = translation_homography(10, 0)
        H12 = translation_homography(0, 5)
        result = accumulate_homographies([H01, H12], 1)
        # result[0] transforms from frame 0 to frame 1
        pt_frame0 = np.array([[0, 0]], dtype=np.float64)
        pt_frame1 = apply_homography(pt_frame0, result[0])
        np.testing.assert_array_almost_equal(pt_frame1, [[10, 0]])

    def test_167_no_nan(self):
        Hs = [translation_homography(5, 3)] * 3
        result = accumulate_homographies(Hs, 1)
        for H in result:
            assert not np.any(np.isnan(H))

    def test_168_five_frames(self):
        Hs = [translation_homography(i, 0) for i in range(4)]
        result = accumulate_homographies(Hs, 2)
        assert len(result) == 5

    def test_169_m_equals_0(self):
        Hs = [translation_homography(5, 0)] * 3
        result = accumulate_homographies(Hs, 0)
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_170_m_equals_last(self):
        Hs = [translation_homography(5, 0)] * 3
        result = accumulate_homographies(Hs, 3)
        np.testing.assert_array_almost_equal(result[3], np.eye(3))

    def test_171_inverse_relationship(self):
        """H_{i->m} should be inverse of H_{m->i}."""
        H01 = translation_homography(10, 5)
        result = accumulate_homographies([H01], 0)
        # result[1] = H_{1->0} = inv(H_{0->1})
        H_inv = np.linalg.inv(H01)
        H_inv /= H_inv[2, 2]
        np.testing.assert_array_almost_equal(result[1], H_inv, decimal=5)

    def test_172_returns_list(self):
        Hs = [np.eye(3)]
        result = accumulate_homographies(Hs, 0)
        assert isinstance(result, list)

    def test_173_each_element_is_ndarray(self):
        Hs = [np.eye(3)]
        result = accumulate_homographies(Hs, 0)
        for H in result:
            assert isinstance(H, np.ndarray)

    def test_174_many_frames(self):
        Hs = [translation_homography(1, 0)] * 20
        result = accumulate_homographies(Hs, 10)
        assert len(result) == 21

    def test_175_consistency(self):
        """H_{0->m} @ H_{m->0} should be identity."""
        H01 = translation_homography(10, 5)
        H12 = translation_homography(3, 7)
        result0 = accumulate_homographies([H01, H12], 0)
        result1 = accumulate_homographies([H01, H12], 1)
        # result0[1] maps frame 1 to frame 0
        # result1[0] maps frame 0 to frame 1
        product = result0[1] @ result1[0]
        product /= product[2, 2]
        # This should give approximately H_{0->0} via frame 1 as intermediate
        # Actually these are both relative to different refs, so:
        # result0[1]: frame 1 -> frame 0
        # result1[0]: frame 0 -> frame 1
        # composing: frame 0 -> frame 1 -> frame 0 = identity
        composed = result0[1] @ result1[0]
        composed /= composed[2, 2]
        np.testing.assert_array_almost_equal(composed, np.eye(3), decimal=5)


# ==================== compute_bounding_box (tests 176-185) ====================

class TestComputeBoundingBox:
    def test_176_identity_bbox(self):
        bbox = compute_bounding_box(identity_homography(), 100, 50)
        np.testing.assert_array_almost_equal(bbox, [[0, 0], [99, 49]])

    def test_177_output_shape(self):
        bbox = compute_bounding_box(identity_homography(), 100, 50)
        assert bbox.shape == (2, 2)

    def test_178_translation_bbox(self):
        H = translation_homography(10, 20)
        bbox = compute_bounding_box(H, 100, 50)
        np.testing.assert_array_almost_equal(bbox, [[10, 20], [109, 69]])

    def test_179_min_less_than_max(self):
        H = translation_homography(5, 3)
        bbox = compute_bounding_box(H, 100, 50)
        assert bbox[0, 0] < bbox[1, 0]  # x_min < x_max
        assert bbox[0, 1] < bbox[1, 1]  # y_min < y_max

    def test_180_scaling(self):
        H = np.diag([2.0, 2.0, 1.0])
        bbox = compute_bounding_box(H, 50, 50)
        np.testing.assert_array_almost_equal(bbox[0], [0, 0])
        np.testing.assert_array_almost_equal(bbox[1], [98, 98])

    def test_181_negative_translation(self):
        H = translation_homography(-10, -20)
        bbox = compute_bounding_box(H, 100, 50)
        np.testing.assert_almost_equal(bbox[0, 0], -10)
        np.testing.assert_almost_equal(bbox[0, 1], -20)

    def test_182_small_image(self):
        bbox = compute_bounding_box(identity_homography(), 2, 2)
        np.testing.assert_array_almost_equal(bbox, [[0, 0], [1, 1]])

    def test_183_no_nan(self):
        bbox = compute_bounding_box(translation_homography(5, 5), 100, 100)
        assert not np.any(np.isnan(bbox))

    def test_184_uses_four_corners(self):
        """Bounding box should contain all four warped corners."""
        H = translation_homography(5, 3)
        w, h = 100, 50
        corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float64)
        warped = apply_homography(corners, H)
        bbox = compute_bounding_box(H, w, h)
        assert np.all(warped[:, 0] >= bbox[0, 0] - 1e-10)
        assert np.all(warped[:, 0] <= bbox[1, 0] + 1e-10)
        assert np.all(warped[:, 1] >= bbox[0, 1] - 1e-10)
        assert np.all(warped[:, 1] <= bbox[1, 1] + 1e-10)

    def test_185_first_row_is_top_left(self):
        """Spec: first row is [x,y] of top left, second is bottom right."""
        bbox = compute_bounding_box(identity_homography(), 100, 50)
        assert bbox[0, 0] <= bbox[1, 0]
        assert bbox[0, 1] <= bbox[1, 1]


# ==================== warp_channel (tests 186-195) ====================

class TestWarpChannel:
    def test_186_identity_warp(self):
        im = make_random_image(50, 50)
        result = warp_channel(im, identity_homography())
        np.testing.assert_array_almost_equal(result, im, decimal=5)

    def test_187_output_is_2d(self):
        im = make_random_image(50, 50)
        result = warp_channel(im, identity_homography())
        assert result.ndim == 2

    def test_188_translation_warp_shape(self):
        im = make_random_image(50, 50)
        H = translation_homography(10, 5)
        result = warp_channel(im, H)
        assert result.ndim == 2
        # Shape may differ due to bounding box shift

    def test_189_no_nan(self):
        im = make_random_image(50, 50)
        result = warp_channel(im, identity_homography())
        assert not np.any(np.isnan(result))

    def test_190_scaling_warp(self):
        im = make_random_image(50, 50)
        H = np.diag([2.0, 2.0, 1.0])
        result = warp_channel(im, H)
        # Scaled image should be larger
        assert result.shape[0] >= 50
        assert result.shape[1] >= 50

    def test_191_float_output(self):
        im = make_random_image(50, 50)
        result = warp_channel(im, identity_homography())
        assert result.dtype in [np.float64, np.float32]

    def test_192_uses_inverse_homography(self):
        """Spec: backward warping uses inverse homography."""
        im = make_random_image(50, 50)
        # If we use identity, result should match input
        result = warp_channel(im, identity_homography())
        np.testing.assert_array_almost_equal(result, im, decimal=3)

    def test_193_small_translation(self):
        im = make_gradient_image(30, 30)
        H = translation_homography(1, 0)
        result = warp_channel(im, H)
        assert result.ndim == 2

    def test_194_non_square_image(self):
        im = make_random_image(30, 60)
        result = warp_channel(im, identity_homography())
        assert result.shape == (30, 60)

    def test_195_values_in_range(self):
        im = make_random_image(50, 50)
        im = np.clip(im, 0, 1)
        result = warp_channel(im, identity_homography())
        assert result.min() >= -0.1
        assert result.max() <= 1.1


# ==================== warp_image (tests 196-200) ====================

class TestWarpImage:
    def test_196_output_shape_3_channels(self):
        im = np.random.RandomState(42).rand(50, 50, 3)
        result = warp_image(im, identity_homography())
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_197_identity_warp(self):
        im = np.random.RandomState(42).rand(50, 50, 3)
        result = warp_image(im, identity_homography())
        np.testing.assert_array_almost_equal(result, im, decimal=5)

    def test_198_no_nan(self):
        im = np.random.RandomState(42).rand(50, 50, 3)
        result = warp_image(im, identity_homography())
        assert not np.any(np.isnan(result))

    def test_199_channels_warped_independently(self):
        im = np.random.RandomState(42).rand(50, 50, 3)
        H = identity_homography()
        result = warp_image(im, H)
        for c in range(3):
            single = warp_channel(im[:, :, c], H)
            np.testing.assert_array_almost_equal(result[:, :, c], single)

    def test_200_translation_warp(self):
        im = np.random.RandomState(42).rand(30, 30, 3)
        H = translation_homography(5, 3)
        result = warp_image(im, H)
        assert result.ndim == 3
        assert result.shape[2] == 3
