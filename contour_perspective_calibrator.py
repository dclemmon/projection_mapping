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
    Given a frame, display the image in full screen
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


def load_camera_props(props_file=None):
    """
    Load the camera properties from file.  To build this file you need
    to run the aruco_calibration.py file
    """
    if props_file is None:
        props_file = 'camera_config.json'
    with open(props_file, 'r') as f:
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


def find_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Add some blur
    edged = cv2.Canny(gray, 30, 200)  # Find our edges
    return edged


def get_region_corners(frame):
    """
    Find the four corners of our projected region and return them in
    the proper order
    """
    edged = find_edges(frame)
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
    rect = order_corners(pts)
    return rect


def order_corners(pts):
    """
    Given the four points found for our contour order them into
    Top Left, Top Right, Bottom Right, Bottom Left
    This order is important for perspective transforms
    """
    rect = np.zeros((4, 2), dtype = 'float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_description_array(rect):
    """
    Given a rectangle return the description array
    """
    (tl, tr, br, bl) = rect  # Unpack the values
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
        ], dtype = 'float32')
    return dst, maxWidth, maxHeight


def get_perspective_transform(stream):
    """
    Determine the perspective transform for the current physical layout
    return the perspective transform, maxWidth, and maxHeight for the 
    projected region
    """
    # Grab a photo of the frame
    frame = stream.read()

    reference_image = get_reference_image()

    # We're going to work with a smaller image, so we need to save the scale
    ratio = frame.shape[0] / 300.0
    # Display the reference image
    show_full_frame(reference_image)
    # Delay execution a quarter of a second to make sure the image is displayed 
    # Don't use time.sleep() here, we want the IO loop to run.  Sleep doesn't do that
    cv2.waitKey(250) 

    # Undistort the camera image
    frame = undistort_image(frame)
    orig = frame.copy()
    # Resize our image smaller, this will make things a lot faster
    frame = imutils.resize(frame, height = 300)

    rect = get_region_corners(frame)
    rect *= ratio  # We shrank the image, so now we have to scale our points up

    dst, maxWidth, maxHeight = get_description_array(rect)

    # Remove the reference image from the display
    hide_full_frame()

    m = cv2.getPerspectiveTransform(rect, dst)

    # Uncomment the lines below to see the transformed image
    #wrap = cv2.warpPerspective(orig, m, (maxWidth, maxHeight))

    #cv2.imshow('all better', wrap)
    #cv2.waitKey(0)
    return m, maxWidth, maxHeight


if __name__ == '__main__':
    resolution = (960, 720)  # Camera frame resolution

    stream = VideoStream(usePiCamera=True, resolution=resolution).start()

    time.sleep(2)  # Let the camera warm up

    get_perspective_transform(stream)
    stream.stop()
    cv2.destroyAllWindows()
