import cv2
import argparse
from enum import Enum


class InputType(Enum):
    CAMERA = "C"
    VIDEO = "V"


def comeOnAndSLAM(inputData):
    cap = cv2.VideoCapture(inputData)
    orb = cv2.ORB_create(nfeatures=25)

    while(True):
        # Capture a frame
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        # Detect ORB features and descriptors
        kp, des = orb.detectAndCompute(gray, None)

        if (len(kp) > 0):
            print("First keypoint:\n", kp[0])
            print("First descriptor:\n", des[0])

        frameWithKeypoints = cv2.drawKeypoints(
            frame, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        cv2.imshow('Capture', frameWithKeypoints)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
