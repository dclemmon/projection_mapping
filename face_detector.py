#! /bin/env python3
from imutils.video import VideoStream
from imutils import face_utils
from imutils import rotate_bound
from contour_perspective_calibrator import load_camera_props
from contour_perspective_calibrator import undistort_image
from contour_perspective_calibrator import get_perspective_transform
from contour_perspective_calibrator import show_full_frame
import datetime
import imutils
import math
import time
import dlib
import cv2
import numpy as np


def build_image(frame, display_resolution, markers, predictor):
    """
    Function to build our marker image
    We're building a black image and adding the proper markers to it, so that
    when it's projected, only the markers display on the target
    :param image: corrected and transformed image
    :param display_resolution (tuple): the current displayed or projected screen resolution
    :param markers (list): Found detector markers
    :param predictor: the loaded facial predictor
    :return: built image
    """
    d_width, d_height = display_resolution
    f_height, f_width = frame.shape
    img = np.zeros((d_height, d_width, 3), np.uint8)
    for mark in markers:
        shape = predictor(gray, mark)
        shape = face_utils.shape_to_np(shape)
        # Grab some info from the detected face.
        # The top and left give us the origin
        # The width and height give us scale/size
        # DONT FORGET we need to map the values back to screen resolution
        face_left = int(np.interp(mark.left(), [0, f_width], [0, d_width]))
        face_top = int(np.interp(mark.top(), [0, f_height], [0, d_height]))
        face_width = int(mark.width() * (d_width/f_width))
        face_height = int(mark.height() * (d_height/f_height))

        scaled_shape = np.copy(shape)
        for index, (x, y) in enumerate(shape):
            # We need to map our points to the new image from the origial
            new_x = int(np.interp(x, [0, f_width], [0, d_width]))
            new_y = int(np.interp(y, [0, f_height], [0, d_height]))
            scaled_shape[index] = [new_x, new_y]
            # Uncomment the line below to set the point projected on the target
            #cv2.circle(img, (new_x, new_y), 1, (255, 255, 255), -1)
        inclination = calc_incl(scaled_shape[17], scaled_shape[26])  # get the info from eyebrows
        apply_sprite(img, face_width, face_left, face_top, inclination)
    return img


def apply_sprite(image, width, x, y, angle):
    # sprite = cv2.imread('santa_hat.png', -1)
    sprite = cv2.imread('santa_hat.png', cv2.IMREAD_UNCHANGED)
    sprite = rotate_bound(sprite, angle)
    sprite, y_final = transform_sprite(sprite, width, y)
    sp_h, sp_w = sprite.shape[:2]
    img_h, img_w = image.shape[:2]

    if y_final + sp_h >= img_h:  # Off the page to the bottom
        sprite = sprite[0:img_h-y_final, :, :]
    if x + sp_w >= img_w:  # Off the page to the right
        sprite = sprite[:, 0:img_w-x, :]
    if x < 0:  # Off the page to the left
        sprite = sprite[:, abs(x)::, :]
        sp_w = sprite.shape[1]
        x = 0

    # loop through and combine the image and sprite based on their alpha vlues
    for chan in range(3):
        image[y_final:y_final+sp_h, x:x+sp_w, chan] = \
                sprite[:, :, chan] * (sprite[:, :, 3] / 255.0) + \
                image[y_final:y_final+sp_h, x:x+sp_w, chan] * \
                (1.0 - sprite[:, :, 3] / 255.0)
    return image


def transform_sprite(sprite, width, y):
    manual_adjust = 1.2  # Added this to account for the extra width of the sprite
    sp_h, sp_w = sprite.shape[:2]
    ratio = (float(width) * manual_adjust)/float(sp_w)
    sprite = cv2.resize(sprite, (0, 0), fx=ratio, fy=ratio)
    sp_h, sp_w = sprite.shape[:2]
    y_origin = y - sp_h
    if y_origin < 0:  # the sprite is off the page, so cut it off
        sprite = sprite[abs(y_origin)::,:,:]
        y_origin = 0
    return sprite, y_origin


def calc_incl(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    incl = 180/math.pi*math.atan((float(y2-y1)/(x2-x1)))
    return incl


print('Loading facial landmark predictor...')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print('Camera sensor warming up...')
vs = VideoStream(usePiCamera=True, resolution=(960, 720)).start()
time.sleep(2)

cameraMatrix, distCoeffs = load_camera_props()
m, maxWidth, maxHeight = get_perspective_transform(vs)

while True:
    frame = vs.read()
    frame = undistort_image(frame, cameraMatrix, distCoeffs)  # Remove camera distortion
    frame = cv2.warpPerspective(frame, m, (maxWidth, maxHeight))
    #frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    image = build_image(gray, (1824, 984), rects, predictor)

    show_full_frame(image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

