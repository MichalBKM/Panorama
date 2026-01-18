import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

from utils import *

#########################################################
# TODO: Implement the following functions               #
#                                                       #
# PART 1: REGISTRATION                                  #
# 1. harris_corner_detector - DONE, Michal              #
# 2. feature_descriptor - DONE, Michal                  #
# 3. find_features - DONE, Michal                       #
# 4. match_features - DONE, Michal                      #
# 5. apply_homography - DONE                            #
# 6. ransac_homography                                  #
# 7. display_matches                                    #         
#                                                       #
# PART 2: STITCHING                                     #
# 8. accumulate_homographies                            #
# 9. compute_bounding_box                               #
# 10. warp_channel                                      #
# 11. warp_image                                        #
#########################################################


# DONE - This function works correctly
def harris_corner_detector(im):
    """
    Implements the harris corner detection algorithm.
    :param im: A 2D array representing a grayscale image.
    :return: An array with shape (N,2), where its ith entry is the [x,y] coordinates of the ith corner point.
    """
    # 1. Get the Ix and Iy image derivatives using the filters [-1 0 1] and its transpose.
    dx_filter = np.array([[-1, 0, 1]])
    dy_filter = dx_filter.T
    ix = convolve2d(im, dx_filter, mode='same', boundary='symm')
    iy = convolve2d(im, dy_filter, mode='same', boundary='symm')

    # 2. Blur the images: Ix^2, Iy^2, IxIy using blur_spatial with kernel_size=3.
    ix2 = ix * ix
    iy2 = iy * iy
    ixiy = ix * iy
    ix2_blurred = blur_spatial(ix2, kernel_size=3)
    iy2_blurred = blur_spatial(iy2, kernel_size=3)
    ixiy_blurred = blur_spatial(ixiy, kernel_size=3)

    # 3. This will yield the matrix M and we use the response function R = det(M) - alpha(trace(M))^2, with alpha=0.04,
    #    to measure how big the two eigenvalues of M are.
    #    M = [[ix2, ixiy], [ixiy, iy2]]
    det_M = (ix2_blurred * iy2_blurred) - (ixiy_blurred ** 2)
    trace_M = ix2_blurred + iy2_blurred
    alpha = 0.04
    R = det_M - alpha * (trace_M ** 2)

    # 4. Perform non-maximum suppression on R.
    nms_result = non_maximum_suppression(R)

    # 5. Return the (x,y) coordinates of the detected corners.
    corners = np.argwhere(nms_result)
    return corners[:, ::-1]  # Switch from (row, col) to (x, y)


# DONE - This function works correctly
def feature_descriptor(im, points, desc_rad=3):
    """
    Samples descriptors at the given feature points.
    :param im: A 2D array representing a grayscale image.
    :param points: An array with shape (N,2) representing feature points coordinates in the image.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: An array of 2D patches, each patch i representing the descriptor of point i.
    """
    patch_size = 2 * desc_rad + 1
    patches = []

    for point in points:
        # (x,y) image coordinates
        x, y = point

        # Create grid 
        x_pixels = np.arange(x - desc_rad, x + desc_rad + 1)
        y_pixels = np.arange(y - desc_rad, y + desc_rad + 1)
        x_grid, y_grid = np.meshgrid(x_pixels, y_pixels)

        # map_coordinates expects (row, col) = (y, x)
        coords = np.vstack((y_grid.ravel(), x_grid.ravel()))

        # Sample patch + reshape back into 2D
        patch = map_coordinates(im, coords, order=1, mode='reflect').reshape((patch_size, patch_size))

        # Normalize patch
        patch = normalize(patch)
        
        # add to list
        patches.append(patch)

    return np.array(patches)


# DONE - Need to check for correctness
def find_features(im):
    """
    Detects and extracts feature points from a specific pyramid level.
    :param im: A 2D array representing a grayscale image.
    :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                These coordinates are provided at the original image level.
            2) A feature descriptor array with shape (N,K,K)
    """
    # Build Gaussian pyramid with 3 levels: 0 (original image), 1, 2
    pyr = build_gaussian_pyramid(im, max_levels=3, filter_size=3) #what filter size?
    points_level_0 = spread_out_corners(im, m = 7,n = 7, radius = 12, harris_corner_detector=harris_corner_detector)
    # Convert the location of the points to level 3
    points_level_2 = points_level_0 // 4
    # Extract feature descriptors at level 3
    desc = feature_descriptor(pyr[2], points_level_2, desc_rad=3)
    # Return points at level 1 and descriptors
    return [points_level_0, desc]


# DONE - Need to check for correctness
def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K). K = 7
    :param desc2: A feature descriptor array with shape (N2,K,K). K = 7
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # Flatten each descriptor to a 1D array - now we have shape (N1, K*K) and (N2, K*K)
    desc1_flat = desc1.reshape(desc1.shape[0], -1)
    desc2_flat = desc2.reshape(desc2.shape[0], -1)

    # Compute the score matrix using dot product
    score_mat = np.dot(desc1_flat, desc2_flat.T)
    
    # Get indices of 2 largest values per row
    top2_in_desc2 = np.argsort(score_mat, axis=1)[:, -2:]
    
    # Get indices of 2 largest values per column
    top2_in_desc1 = np.argsort(score_mat, axis=0)[-2:, :]  
    
    # Find matches that satisfy all three conditions
    matches_idx1 = []
    matches_idx2 = []
    for i in range(desc1.shape[0]):
        for j in range(desc2.shape[0]):
            curr_score = score_mat[i, j]

            if curr_score >= min_score and \
               j in top2_in_desc2[i] and \
                i in top2_in_desc1[:, j]:
                 matches_idx1.append(i)
                 matches_idx2.append(j)

    return [np.array(matches_idx1, dtype=int), np.array(matches_idx2, dtype=int)]


def apply_homography(points, homographic_matrix):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """

    #Step1. Convert to homogenous coordinates
    n = points.shape[0]
    ones = np.ones((n, 1))
    points_padded = np.hstack((points, ones))

    #Step2. Linear Transformation: Apply the homograpjy matrix
    transformed_points = (np.dot(homographic_matrix, points_padded.T)).T

    #Step3. Convert back to inhomogenous coordinates
    w = transformed_points[:, 2:3]
    normalized_points = transformed_points[:, :2] / w

    return normalized_points

def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inliers = np.array([], dtype=int)
    for _ in range(num_iter):

        #Step1. Sample 2 point correspondences from the supplies N matches
        idx_i, idx_j = np.random.choice(len(points1), 2, replace=False)

        p1_i = points1[idx_i]
        p2_i = points2[idx_i]

        p1_j = points1[idx_j]
        p2_j = points2[idx_j]

        sample_p1 = np.array([p1_i, p1_j])
        sample_p2 = np.array([p2_i, p2_j])

        #Step2. Compute the homography
        H12 = estimate_rigid_transform(sample_p1, sample_p2, translation_only)

        #Step3.1  Apply H12 over all P1 points
        predicted_p2 = apply_homography(points1, H12)

        #Step3.2 Calculate the squared Euclidean distance between P1 and P2
        distances = np.sum((predicted_p2 - points2)**2, axis=1)

        #Step3.3 Mark points with distance < inlier_tol as inliers
        inliers = np.where(distances < inlier_tol)[0]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
        
        
    #4. recompute the homography using all the inliers
    if len(max_inliers) == 0:
        final_H12 = np.eye(3)
    else:
        final_H12 = estimate_rigid_transform(points1[max_inliers], 
                                         points2[max_inliers], 
                                         translation_only)
    return [final_H12, max_inliers]

def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    # 1. Concatenate images horizontally
    combined_im = np.hstack((im1, im2))
    plt.imshow(combined_im, cmap='gray')
    
    # Width of the first image to offset points in the second image
    width_offset = im1.shape[1]
    
    # 2. Iterate through all matches to draw lines
    for i in range(len(points1)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Offset x2 for the second image's position in the combined view
        x2_offset = x2 + width_offset
        
        # Determine color: Blue for inliers, Yellow for outliers
        color = 'b' if i in inliers else 'y'
        
        # Draw the line between matches
        plt.plot([x1, x2_offset], [y1, y2], color=color, lw=0.5)
        
        # Draw points as red dots
        plt.plot(x1, y1, 'ro', mfc='none', markersize=3)
        plt.plot(x2_offset, y2, 'ro', mfc='none', markersize=3)
    
    plt.title("Matches: Blue = Inliers, Yellow = Outliers")
    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of successive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    pass


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    pass

def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    pass

def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    pass



##################################################################################################


def align_images(files, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in files:
        image = read_image(file, 1)
        points_and_descriptors.append(find_features(image))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
        desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

        Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    homographies = np.stack(accumulated_homographies)
    frames_for_panoramas = filter_homographies_with_translation(homographies, minimum_right_translation=5)
    homographies = homographies[frames_for_panoramas]
    return frames_for_panoramas, homographies

def generate_panoramic_images(data_dir, file_prefix, num_images, out_dir, number_of_panoramas, translation_only=False):
    """
    combine slices from input images to panoramas.
    The naming convention for a sequence of images is file_prefixN.jpg, where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    :param out_dir: path to output panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """

    file_prefix = file_prefix
    files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    files = list(filter(os.path.exists, files))
    print('found %d images' % len(files))
    image = read_image(files[0], 1)
    h, w = image.shape

    frames_for_panoramas, homographies = align_images(files, translation_only)

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    bounding_boxes = np.zeros((frames_for_panoramas.size, 2, 2))
    for i in range(frames_for_panoramas.size):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset

    slice_centers = np.linspace(0, w, number_of_panoramas + 2, endpoint=True, dtype=np.int32)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
        slice_center_2d = np.array([slice_centers[i], h // 2])[None, :]
        # homography warps the slice center to the coordinate system of the middle image
        warped_centers = [apply_homography(slice_center_2d, h) for h in homographies]
        # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
        warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(np.int32) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int32)

    panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(frames_for_panoramas):
        # warp every input image once, and populate all panoramas
        image = read_image(files[frame_index], 2)
        warped_image = warp_image(image, homographies[i])
        x_offset, y_offset = bounding_boxes[i][0].astype(np.int32)
        y_bottom = y_offset + warped_image.shape[0]

        for panorama_index in range(number_of_panoramas):
            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[panorama_index, i:i + 2]
            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
            x_end = boundaries[0] + image_strip.shape[1]
            panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    os.makedirs(out_dir, exist_ok=True)
    for i, panorama in enumerate(panoramas):
        plt.imsave('%s/panorama%02d.png' % (out_dir, i + 1), panorama)


if __name__ == "__main__":
    import ffmpeg
    video_name = "mt_cook.mp4"
    video_name_base = video_name.split('.')[0]
    os.makedirs(f"dump/{video_name_base}", exist_ok=True)
    ffmpeg.input(f"videos/{video_name}").output(f"dump/{video_name_base}/{video_name_base}%03d.jpg").run()
    num_images = len(os.listdir(f"dump/{video_name_base}"))
    print(f"Generated {num_images} images")

    # Visualize feature points on two sample images
    print("Extracting and visualizing feature points...")
    image1 = read_image(f"dump/{video_name_base}/{video_name_base}200.jpg", 1)
    image2 = read_image(f"dump/{video_name_base}/{video_name_base}300.jpg", 1)

    # Extract feature points and descriptors
    points1, desc1 = find_features(image1)
    points2, desc2 = find_features(image2)

    # Visualize points on first image
    print(f"Found {len(points1)} feature points in image 1")
    visualize_points(image1, points1)

    # Visualize points on second image
    print(f"Found {len(points2)} feature points in image 2")
    visualize_points(image2, points2)

    # Match features between the two images
    print("Matching features between images...")
    ind1, ind2 = match_features(desc1, desc2, 0.6)
    matched_points1 = points1[ind1]
    matched_points2 = points2[ind2]
    print(f"Found {len(ind1)} matches")

    # Run RANSAC to find inliers
    H12, inliers = ransac_homography(matched_points1, matched_points2, 100, 6, translation_only=False)
    print(f"Found {len(inliers)} inliers out of {len(matched_points1)} matches")

    # Display matches with inliers and outliers
    display_matches(image1, image2, matched_points1, matched_points2, inliers)

    """
    # Generate panoramic images
    print("\nGenerating panoramic images...")
    generate_panoramic_images(f"dump/{video_name_base}/", video_name_base,
                              num_images=num_images, out_dir=f"out/{video_name_base}", number_of_panoramas=3)
    """

