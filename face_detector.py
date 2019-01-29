#! /bin/env python3

from imutils.video import VideoStream
from imutils import face_utils
from imutils import rotate_bound
from contour_perspective_calibrator import load_camera_props
from contour_perspective_calibrator import undistort_image
from contour_perspective_calibrator import get_perspective_transform
from contour_perspective_calibrator import show_full_frame

import argparse
import math
import time
import dlib
import cv2
import numpy as np


def build_image(frame, display_resolution, markers, predictor, sprite_path):
    """
    Function to build our marker image
    We're building a black image and adding the proper markers to it, so that
    when it's projected, only the markers display on the target
    :param frame: corrected and transformed image (already b&w)
    :param display_resolution: the current displayed or projected screen resolution
    :param markers: Found detector markers
    :param predictor: the loaded facial predictor
    :param sprite_path: the location of the sprite
    :return: built image
    """
    d_width, d_height = display_resolution
    f_height, f_width = frame.shape
    img = np.zeros((d_height, d_width, 3), np.uint8)
    for mark in markers:
        shape = predictor(frame, mark)
        shape = face_utils.shape_to_np(shape)
        # Grab some info from the detected face.
        # The top and left give us the origin
        # The width and height give us scale/size
        # DON'T FORGET we need to map the values back to screen resolution
        face_left = int(np.interp(mark.left(), [0, f_width], [0, d_width]))
        face_top = int(np.interp(mark.top(), [0, f_height], [0, d_height]))
        face_width = int(mark.width() * (d_width/f_width))
        face_height = int(mark.height() * (d_height/f_height))

        scaled_shape = np.copy(shape)
        for index, (x, y) in enumerate(shape):
            # We need to map our points to the new image from the original
            new_x = int(np.interp(x, [0, f_width], [0, d_width]))
            new_y = int(np.interp(y, [0, f_height], [0, d_height]))
            scaled_shape[index] = [new_x, new_y]
            # Uncomment the line below to set the point projected on the target
            # cv2.circle(img, (new_x, new_y), 1, (255, 255, 255), -1)
        inclination = calc_incl(scaled_shape[17], scaled_shape[26])  # get the info from eyebrows
        apply_sprite(img, face_width, face_left, face_top, inclination, sprite_path)
    return img


def apply_sprite(image, width, x, y, angle, sprite_file):
    """
    Given an image, add our sprite
    :param image: our image to be projected
    :param width: Target face width
    :param x: Face location left
    :param y: Face location top
    :param angle: face tilt
    :param sprite_file: the filename of our sprite
    :return: projection image
    """
    sprite = cv2.imread(sprite_file, cv2.IMREAD_UNCHANGED)
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

    # loop through and combine the image and sprite based on the sprite alpha values
    for chan in range(3):
        image[y_final:y_final+sp_h, x:x+sp_w, chan] = \
                sprite[:, :, chan] * (sprite[:, :, 3] / 255.0) + \
                image[y_final:y_final+sp_h, x:x+sp_w, chan] * \
                (1.0 - sprite[:, :, 3] / 255.0)
    return image


def transform_sprite(sprite, width, y):
    """
    Match the size of our sprite to our detected face width
    :param sprite: the fun image
    :param width: the width we need to adjust to
    :param y: Vertical position of the sprite
    :return: the sprite (may be modified) and the new origin
    """
    manual_adjust = 1.2  # Added this to account for the extra width of the sprite
    sp_h, sp_w = sprite.shape[:2]
    ratio = (float(width) * manual_adjust)/float(sp_w)
    sprite = cv2.resize(sprite, (0, 0), fx=ratio, fy=ratio)
    sp_h, sp_w = sprite.shape[:2]
    y_origin = y - sp_h
    if y_origin < 0:  # the sprite is off the page, so cut it off
        sprite = sprite[abs(y_origin)::, :, :]
        y_origin = 0
    return sprite, y_origin


def calc_incl(point1, point2):
    """
    Calculate the angle of inclination between two points
    :param point1:
    :param point2:
    :return: the angle in question
    """
    x1, y1 = point1
    x2, y2 = point2
    incl = 180/math.pi*math.atan((float(y2-y1)/(x2-x1)))
    return incl


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--camera_props', default='camera_config.json',
                    help='Camera property file')
    ap.add_argument('-w', '--camera_width', type=int, default=960,
                    help='Camera image width')
    ap.add_argument('-h', '--camera_height', type=int, default=720,
                    help='Camera image height')
    ap.add_argument('-sw', '--screen_width', type=int, default=1824,
                    help='Projector or screen width')
    ap.add_argument('-sh', '--screen_height', type=int, default=984,
                    help='Projector or screen height')
    ap.add_argument('-s', '--sprite', default='santa_hat.png',
                    help='Our image sprite')
    ap.add_argument('-p', '--predictor',
                    default='shape_predictor_68_face_landmarks.dat',
                    help='Face landmark shape predictor')

    return vars(ap.parse_args())


if __name__ == '__main__':
    args = parse_args()
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.get('predictor'))
    print('Camera sensor warming up...')
    camera_resolution = (args.get('camera_width'), args.get('camera_height'))
    vs = VideoStream(usePiCamera=True, resolution=camera_resolution).start()
    time.sleep(2)

    prop_file = args.get('camera_props')
    cameraMatrix, distCoeffs = load_camera_props(prop_file)
    screen_resolution = (args.get('screen_width'), args.get('screen_height'))
    m, maxWidth, maxHeight = get_perspective_transform(
        vs,
        screen_resolution,
        prop_file
    )

    while True:
        frame = vs.read()
        frame = undistort_image(frame, cameraMatrix, distCoeffs)  # Remove camera distortion
        frame = cv2.warpPerspective(frame, m, (maxWidth, maxHeight))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        image = build_image(gray, screen_resolution, rects, predictor, args.get('sprite'))

        show_full_frame(image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

