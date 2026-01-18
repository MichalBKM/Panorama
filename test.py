from ex2_empty import *
import numpy as np

def test_apply_homography():
    """Test the apply_homography function with known transformations."""
    
    # Test 1: Identity transformation (should return same points)
    print("Test 1: Identity Matrix")
    points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    H_identity = np.eye(3)
    result = apply_homography(points, H_identity)
    print(f"Input points:\n{points}")
    print(f"Output points:\n{result}")
    print(f"Match: {np.allclose(points, result)}\n")
    
    # Test 2: Pure translation
    print("Test 2: Translation by (10, 20)")
    H_translation = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ])
    result = apply_homography(points, H_translation)
    expected = points + np.array([10, 20])
    print(f"Output points:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}\n")
    
    # Test 3: Scaling by factor of 2
    print("Test 3: Scaling by 2")
    H_scale = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ])
    result = apply_homography(points, H_scale)
    expected = points * 2
    print(f"Output points:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}\n")
    
    # Test 4: 90-degree rotation (counter-clockwise)
    print("Test 4: 90-degree rotation")
    H_rotate = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    result = apply_homography(points, H_rotate)
    expected = np.array([[-2.0, 1.0], [-4.0, 3.0], [-6.0, 5.0]])
    print(f"Output points:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Match: {np.allclose(result, expected)}\n")

def check_harris_detector():
    # ---- Load grayscale image ----
    im = read_image('photos/color.jpg', GRAY_REPRESENTATION)

    # ---- Detect corners ----
    corners = spread_out_corners(
        im,
        m=4,
        n=4,
        radius=8,
        harris_corner_detector=harris_corner_detector
    )

    print(f"Detected {len(corners)} corners")

    # ---- Compute descriptors ----
    desc_rad = 3
    descriptors = feature_descriptor(im, corners, desc_rad)

    # ---- 2. Overlay descriptor rectangles on image ----
    plt.imshow(im, cmap='gray')
    for x, y in corners:
        rect = plt.Rectangle(
            (x - desc_rad, y - desc_rad),
            2*desc_rad + 1,
            2*desc_rad + 1,
            edgecolor='magenta',
            facecolor='none',
            linewidth=1
        )
        plt.gca().add_patch(rect)
    plt.title("Descriptor sampling locations")
    plt.show()