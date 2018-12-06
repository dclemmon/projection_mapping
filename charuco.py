#! /usr/bin/env python3

from cv2 import aruco

inToM = 0.0254

maxWidthIn = 17
maxHeightIn = 23
maxWidthM = maxWidthIn * inToM
maxHeightM = maxHeightIn * inToM

charucoNSqVert = 10
charucoSqSizeM = float(maxHeightM) / float(charucoNSqVert)
charucoMarkerSizeM = charucoSqSizeM * 0.7
# charucoNSqHoriz = int(maxWidthM / charucoSqSizeM)
charucoNSqHoriz = 16

# charucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
charucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
charucoBoard = aruco.CharucoBoard_create(
        charucoNSqHoriz,
        charucoNSqVert,
        charucoSqSizeM,
        charucoMarkerSizeM,
        charucoDictionary)

markerDictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
markerSizeIn = 5
markerSizeM = markerSizeIn * inToM

detectorParams = aruco.DetectorParameters_create()
detectorParams.cornerRefinementMaxIterations = 500
detectorParams.cornerRefinementMinAccuracy = 0.001
detectorParams.adaptiveThreshWinSizeMin = 10
detectorParams.adaptiveThreshWinSizeMax = 10
