#!/bin/env python3

import cv2
import time
from datetime import datetime
from imutils.video import VideoStream

def getImage(stream):
    """
    Loop through the image stream and look for a pattern
    return: found frame
    """
    corners = None
    ret = False
    while not ret:
        frame = stream.read()
        cv2.imshow("grid", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = findCorners(gray)
        cv2.imshow('grid', frame)
        cv2.waitKey(1)
    cv2.waitKey(3000)
    return frame

def findCorners(frame):
    """
    Find out if the frame has the chessboard pattern
    """
    ret, corners = cv2.findChessboardCorners(frame, (7, 9), None)
    return ret, corners


if __name__ == "__main__":
    stream = VideoStream(usePiCamera=True, resolution=(720, 480)).start()
    time.sleep(2)

    for index in range(30):
        image = getImage(stream)

        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite("post/sample_images/" + filename, image)

    cv2.destroyAllWindows()
    stream.stop()
