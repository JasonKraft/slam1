#!/usr/bin/env/python3

import cv2
import argparse
import numpy as np
import math
from enum import Enum


class InputType(Enum):
    CAMERA = "C"
    VIDEO = "V"


def comeOnAndSLAM(inputData, maxFeatures = 1000):
    cap = cv2.VideoCapture(inputData)
    orb = cv2.ORB_create(nfeatures=25)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # initialize state vector
    x_hat_t = np.array([
        0.0,0.0,0.0,                    # X,Y,Z position of the camera
        0.0,0.0,0.0,math.sin(1.0/2.0),  # Orientation quaternion of the camera
        0.0,0.0,0.0,                    # X,Y,Z linear velocity of the camera
        0.0,0.0,0.0                     # X,Y,Z angular velocity of the camera
        ])

    print(x_hat_t)

    old_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    old_kp, old_des = orb.detectAndCompute(old_frame, None)

    while(True):
        # Capture a frame
        new_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

        # Detect ORB features and descriptors
        # new_kp, new_des = orb.detectAndCompute(new_frame, None)

        #calculate optical flow
        new_kp, status, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, old_kp, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # filter out only the good points
        good_old = old_kp[status == 1]
        good_new = new_kp[status == 1]

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

        img = cv2.add(frame,mask)

        # frameWithKeypoints = cv2.drawKeypoints(
        #     new_frame, new_kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        cv2.imshow('Capture', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_frame = new_frame.copy()
        old_kp = good_new.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Run SLAM on either a pre-recorded video or live camera feed.")
    parser.add_argument("--inputtype", metavar="-T", type=InputType, choices=list(
        InputType), help="The input type (either C for camera or V for video).", default="C")

    args = parser.parse_args()
    print("Input type chosen: ", args.inputtype)

    if args.inputtype == InputType.CAMERA:
        comeOnAndSLAM(0)


if __name__ == "__main__":
    main()
