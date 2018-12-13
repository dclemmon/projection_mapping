#! /usr/bin/env python3
"""
    This program calculates the perspective transform of an aruco marker.
    The user should be able to provide appropriate camera calibration 
    information.
"""

import cv2
import imutils
import json
import numpy as np
import time

from imutils.video import VideoStream

from charuco import perspectiveBoard
from charuco import perspectiveDictionary
from charuco import detectorParams


def show_full_frame(frame):
    """
    Given a calibration frame, display the image in full screen
    Usecase is a projector.  The camera module can find the projection region
    using the test pattern
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", frame)


def hide_full_frame(window="Calibration"):
    """
    Kill a named window, the default is the window named "Calibration"
    """
    cv2.destroyWindow(window)


def get_reference_image(resolution=(1680, 1050)):
    """
    Build the image we will be searching for.  In this case, we just want a
    large white box (full screen)
    @param: resolution: this is our screen/projector resolution
    """
    width, height = resolution
    img = np.ones((height, width, 1), np.uint8) * 255
    return img


def get_reference_pixels(gray=None):
    """
    Given an grayscale image, find the charuco corner locations
    It is important to note that this function expects to only find
    four corners in the image.  This is due to our need of four 
    corners later in the perspective transform.
    """
    if gray is None:
        gray = get_reference_image((960, 720))
    charucoCorners = None

    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
        gray,
        perspectiveDictionary,
        parameters=detectorParams)
    if len(markerCorners) == 4:
        ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners,
            markerIds,
            gray,
            perspectiveBoard
        )
    print('Found %s/4 Corners' % ('0' if charucoCorners is None else len(charucoCorners)))

    return charucoCorners


def load_camera_props():
    """
    Load the camera properties from file.  To build this file you need
    to run the aruco_calibration.py file
    """
    with open('camera_config.json', 'r') as f:
        data = json.load(f)
    cameraMatrix = np.array(data.get('cameraMatrix'))
    distCoeffs = np.array(data.get('distCoeffs'))
    return cameraMatrix, distCoeffs


def undistort_image(image):
    """
    Given an image from the camera module, laod the camera properties and correct
    for camera distortion
    """
    resolution = image.shape
    if len(resolution) == 3:
        resolution = resolution[:2]
    resolution = resolution[::-1]  # Shape gives us (height, width) so reverse it
    cameraMatrix, distCoeffs = load_camera_props()
    newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        resolution,
        0
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix,
        distCoeffs,
        None,
        newCameraMatrix,
        resolution,  # What should this be?
        5  # TODO what is the 5?
    )
    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return image


def get_perspective_transform():
    """
    Determine the perspective transform for the current physical layout
    """
    resolution = (960, 720)  # Camera frame resolution

    stream = VideoStream(usePiCamera=True, resolution=resolution).start()
    time.sleep(2)  # Let the camera warm up

    reference_image = get_reference_image()

    # We're going to work with a smaller image, so we need to save the scale
    ratio = resolution[1] / 300.0
    # Display the reference image
    show_full_frame(reference_image)
    cv2.waitKey(250)

    #while True:  # We probably don't need a loop

    # Grab a photo of the frame
    frame = stream.read()
    # Undistort the camera image
    frame = undistort_image(frame)
    orig = frame.copy()
    frame = imutils.resize(frame, height = 300)  # Resize our image smaller
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Add some blur
    edged = cv2.Canny(gray, 30, 200)  # Find our edges
    # findContours is destructive, so send in a copy
    image, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort our contours by area, and keep the 10 largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our contour has four points, we probably found the screen
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        print('Did not find contour')
    # Uncomment these lines to see the contours on the image
    #cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)
    #cv2.imshow('Screen', frame)
    #cv2.waitKey(0)

    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = 'float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    rect *= ratio  # We shrank the image, so now we have to scale our points up

    print(rect)

    (tl, tr, br, bl) = rect
    print(tl, tr, br, bl)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    print(widthA, widthB, heightA, heightB)

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
        ], dtype = 'float32')

    # Remove the reference image from the display
    hide_full_frame()

    m = cv2.getPerspectiveTransform(rect, dst)
    wrap = cv2.warpPerspective(orig, m, (maxWidth, maxHeight))

    cv2.imshow('all better', wrap)
    cv2.waitKey(0)

    stream.stop()
    

if __name__ == '__main__':
    get_perspective_transform()
    cv2.destroyAllWindows()
