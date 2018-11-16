#!/bin/env python3

"""
Camera calibration code from OpenCV python tutorials
"""

import numpy as np
import cv2
import glob



if __name__ == "__main__":
    images = glob.glob('post/sample_images/*')
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cbrow = 7
    cbcol = 9

    #prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('img', img)
            # The user needs to filter the images to make sure the points align
            # pressing 'n' will skip adding the pic to the calibration
            key = cv2.waitKey(0)
            if 'n' == chr(key & 255):
                continue
            objpoints.append(objp)
            imgpoints.append(corners2)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    np.savez("picam_calibration_output", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Total Error: ", mean_error/len(objpoints))

