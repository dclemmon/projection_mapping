#! /usr/bin/env python3

"""
    A script to calibrate the PiCam module using a Charuco board
"""

import time
import json

import cv2

from cv2 import aruco
from imutils.video import VideoStream

from charuco import charucoBoard
from charuco import charucoDictionary
from charuco import detectorParams


def show_calibration_frame(frame):
    """
    Given a calibration frame, display the image in full screen
    Use case is a projector.  The camera module can find the projection region
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


def save_json(data):
    """
    Save our data object as json to the camera_config file
    :param data: data to  write to file
    """
    filename = 'camera_config.json'
    print('Saving to file: ' + filename)
    json_data = json.dumps(data)
    with open(filename, 'w') as f:
        f.write(json_data)


def calibrate_camera():
    """
    Calibrate our camera
    """
    required_count = 50
    resolution = (960, 720)
    stream = VideoStream(usePiCamera=True, resolution=resolution).start()
    time.sleep(2)  # Warm up the camera

    all_corners = []
    all_ids = []

    frame_idx = 0
    frame_spacing = 5
    success = False

    calibration_board = charucoBoard.draw((1680, 1050))
    show_calibration_frame(calibration_board)

    while True:
        frame = stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = aruco.detectMarkers(
            gray,
            charucoDictionary,
            parameters=detectorParams)

        if len(marker_corners) > 0 and frame_idx % frame_spacing == 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                charucoBoard
                )
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

            aruco.drawDetectedMarkers(gray, marker_corners, marker_ids)
        # cv2.imshow('frame', gray)  # If we're showing the calibration image, we can't preview

        if cv2.waitKey(1) & 255 == ord('q'):
            break

        frame_idx += 1
        print("Found: " + str(len(all_ids)) + " / " + str(required_count))

        if len(all_ids) >= required_count:
            success = True
            break
    hide_calibration_frame()
    if success:
        print('Finished collecting data, computing...')

        try:
            err, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                all_corners,
                all_ids,
                charucoBoard,
                resolution,
                None,
                None)
            print('Calibrated with error: ', err)

            save_json({
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'err': err
            })

            print('...DONE')
        except Exception as e:
            print(e)
            success = False

        # Generate the corrections
        new_camera_matrix, valid_pix_roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            resolution,
            0)
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
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
