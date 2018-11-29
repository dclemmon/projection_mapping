#! /usr/bin/env python3

import numpy as np
import cv2
import time
#from matplotlib import pyplot as plt
from imutils.video import VideoStream

MIN_MATCH_COUNT = 2
# FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6

def show_calibration_image(frame):
    """
    Given a calibration frame, display the image in full screen
    Usecase is a projector.  The camera module can find the projection region
    using the test pattern
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", frame)


def hide_calibration_image(window="Calibration"):
    """
    Kill a named window, the default is the window named "Calibration"
    """
    cv2.destroyWindow(window)


def get_camera_image(stream, preview=False):
    """
    Show what the camera sees, and when the spacebar is pressed
    capture an image
    """
    frame = None
    while True:
        frame = stream.read()
        if preview:
            cv2.imshow("Calibration Preview", frame)
        key = cv2.waitKey(1)
        if ' ' == chr(key & 255):
            break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def get_orb_info(frame):
    """
    Given a frame return the SIFT keypoints and descriptors
    """
    MAX_MATCHES = 500
    orb = cv2.ORB_create()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des


def get_homography_info(camera_image, test_image):
    kp_calib, des_calib = get_orb_info(camera_image)
    kp_training, des_training = get_orb_info(test_image)
    #matches = get_flann_matches(
    #        camera_image,
    #        test_image,
    #        des_calib,
    #        des_training)
    matches = get_matches(des_calib, des_training)
    if not matches:
        return

    # Let's look at our matches!
    imMatches = cv2.drawMatches(camera_image, kp_calib, test_image, kp_training, matches, None)
    cv2.imshow("Found!", imMatches)
    cv2.waitKey(0)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1 [i, :] = kp_calib[match.queryIdx].pt
        points2 [i, :] = kp_training[match.trainIdx].pt

    # Find Homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = test_image.shape
    im1Reg = cv2.warpPerspective(camera_image, h, (width, height))

    cv2.imwrite('out.jpg', im1Reg)
    return im1Reg, h

    src_pts = np.float32([ kp_calib[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp_training[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = calibration_frame.shape
    pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    training_frame = cv2.polylines(training_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    draw_params = {
            "matchColor": (0, 255, 0), # draw in green
            "singlePointColor": None,
            "matchesMask": matchesMask, # draw only inliers
            "flags": 2,
            }
    found_image = cv2.drawMatches(
            camera_image,
            kp_calib,
            test_image,
            kp_training,
            matches,
            None,
            **draw_params)
    cv2.imshow(found_image)
    cv2.waitKey(0)


def get_matches(des1, des2):
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    # Sort by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Take the top matches
    MATCH_PERCENT = 0.15
    cutPoint = int(len(matches) * MATCH_PERCENT)
    matches = matches[:cutPoint]
    return matches
    


def get_flann_matches(image1, image2, des1, des2):
    index_params = {
            "algorithm": FLANN_INDEX_LSH,
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Lets remove any matches that don't have two components
    filtered_matches = [x for x in matches if len(x) == 2]
    # store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in filtered_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    if len(good_matches) < MIN_MATCH_COUNT:
        print("Not enough matches found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
        return
    return good_matches


if __name__ == "__main__":
    # Start up the video stream, it takes a little while before we can get data
    #v_stream = VideoStream(usePiCamera=True, resolution=(1280, 800)).start()
    v_stream = VideoStream(usePiCamera=True).start()
    time.sleep(1)

    # Load the calibration image
    training_frame = cv2.imread("calibration_image.jpg", 0)

    # For setup, let's load a preview and make sure the camera can see the screen
    get_camera_image(v_stream, preview=True)

    # Show the calibration image full screen
    show_calibration_image(training_frame)
    calibration_frame = get_camera_image(v_stream)

    # We're done with the calibration image, so hide it
    hide_calibration_image()

    # Find our projectable area
    homography_info = get_homography_info(calibration_frame, training_frame)

