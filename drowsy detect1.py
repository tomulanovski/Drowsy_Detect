import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import winsound
import time


def mouth_aspect_ratio(mouth):  # gives the mouth aspect ratio
    top_lip = mouth[2:7]
    low_lip = mouth[8:13]
    A = distance.euclidean(top_lip[0], low_lip[-1])
    B = distance.euclidean(top_lip[1], low_lip[-2])
    C = distance.euclidean(top_lip[2], low_lip[-3])
    D = distance.euclidean(top_lip[3], low_lip[-4])
    MAR = (A + B + C + D) / 4
    return MAR


def calculate_EAR(eye):  # gives the eye aspect ratio based on the EAR equation
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # using the 68 face landmark file
(lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
cap = cv2.VideoCapture(0)
blinkFlag = 0
yawnFlag = 0
while True:
    ret, frame = cap.read()  # capturing the frames
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)  # giving the face landmarks by points based on the face detector
        face_landmarks = face_utils.shape_to_np(face_landmarks)  # making it an array
        leftEye = face_landmarks[lEyeStart:lEyeEnd]  # making it an array
        rightEye = face_landmarks[rEyeStart:rEyeEnd]  # making it an array
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = (left_ear + right_ear) / 2  ## average of left and right EAR
        mouth = face_landmarks[mouthStart:mouthEnd]
        MAR = mouth_aspect_ratio(mouth)  # calculating mouth aspect ratio
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # EAR = round(EAR, 2)
        if EAR < 0.25:  # 0.25 is the threshold for blinking
            blinkFlag += 1
            if blinkFlag > 20:
                winsound.Beep(600,100)
                cv2.putText(frame, "WAKE UP!!", (20, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 5)
                # to put alarm voice here
        else:  # eyes are open
            blinkFlag = 0

        if MAR > 35:  # need to improve it
            yawnFlag += 1
            if yawnFlag > 10:
                cv2.putText(frame, "You Yawn a lot, stop the car", (20, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

        # else:
        #     yawnFlag = 0

    cv2.imshow("img", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
