#! /usr/bin/env python3

"""
    A script to calibrate the PiCam module using a Charuco board
"""

import time

import cv2

from cv2 import aruco
from imutils.video import VideoStream

from charuco import charucoBoard
from charuco import charucoDictionary
# from charuco import markerDictionary
from charuco import detectorParams


def show_calibration_frame(frame):
    """
    Given a calibration frame, display the image in full screen
    Usecase is a projector.  The camera module can find the projection region
    using the test pattern
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", frame)


def hide_calibration_frame(window="Calibration"):
    """
    Kill a named window, the default is the window named "Calibration"
    """
    cv2.destroyWindow(window)


def calibrate_camera():
    """

    """
    REQUIRED_COUNT = 50
    resolution = (960, 720)
    stream = VideoStream(usePiCamera=True, resolution=resolution).start()
    time.sleep(3)  # Warm up the camera

    allCorners = []
    allIds = []

    frameIdx = 0
    frameSpacing = 5
    success = False

    calibration_board = charucoBoard.draw((1680, 1050))
    show_calibration_frame(calibration_board)

    while True:
        frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        markerCorners, markerIds, _ = aruco.detectMarkers(
            gray,
            # markerDictionary,  # This may be a bug, could need to be charucoDictionary
            charucoDictionary,  # This may be a bug, could need to be charucoDictionary
            parameters=detectorParams)

        if len(markerCorners) > 0 and frameIdx % frameSpacing == 0:
            ret, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
                markerCorners,
                markerIds,
                gray,
                charucoBoard
                )
            if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
                allCorners.append(charucoCorners)
                allIds.append(charucoIds)

            aruco.drawDetectedMarkers(gray, markerCorners, markerIds)
        # cv2.imshow('frame', gray)  # If we're showing the calibration image, we can't preview

        if cv2.waitKey(1) & 255 == ord('q'):
            break

        frameIdx += 1
        print("Found: " + str(len(allIds)) + " / " + str(REQUIRED_COUNT))

        if len(allIds) >= REQUIRED_COUNT:
            success = True
            break
    hide_calibration_frame()
    if success:
        print('Finished collecting data, computing...')

        try:
            err, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                allCorners,
                allIds,
                charucoBoard,
                resolution,
                None,
                None)
            print('Calibrated with error: ', err)

            # TODO Save off the data, JSON could work, so could numpy

            print('...DONE')
        except Exception as e:
            print(e)
            success = False

        # Generate the corrections
        newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(
            cameraMatrix,
            distCoeffs,
            resolution,
            0)
        mapx, mapy = cv2.initUndistortRectifyMap(
            cameraMatrix,
            distCoeffs,
            None,
            newCameraMatrix,
            resolution,
            5)
        while True:
            frame = stream.read()
            if mapx is not None and mapy is not None:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_camera()
