#! /usr/bin/env python3
"""
    This program calculates the perspective transform of an aruco marker.
    The user should be able to provide appropriate camera calibration 
    information.
"""

import cv2
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
    image = perspectiveBoard.draw(resolution)  # Our screen/projector resolution
    return image


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
    show_full_frame(reference_image)

    # Set up the ReferecePixels here... format [[x,y],[x,y],[x,y],[x,y]]
    # ReferencePixels = get_reference_pixels(reference_image)
    ReferencePixels = get_reference_pixels()
    # Display the reference image
    show_full_frame(reference_image)
    while True:
        # Grab a photo of the frame
        frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Undistort the camera image
        corrected_image = undistort_image(gray)
        # Find the key points from the corrected camera image
        projector_points = get_reference_pixels(corrected_image)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
        if projector_points is not None:
            break
    # Remove the reference image from the display
    hide_full_frame()
    print(projector_points)
    print(ReferencePixels)
    M = cv2.getPerspectiveTransform(projector_points, ReferencePixels)
    dst = cv2.warpPerspective(corrected_image, M, (2000, 2000))  # TODO Fix the scaling issue
    cv2.imshow('undistort', dst)
    cv2.imshow('corrected', corrected_image)
    cv2.imshow('original', gray)
    while True:
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    

if __name__ == '__main__':
    get_perspective_transform()
