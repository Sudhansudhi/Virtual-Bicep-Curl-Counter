import cv2
import time
import numpy as np
from PoseModule import poseDetector

def main():
    cap = cv2.VideoCapture(0)  # Use webcam, assuming it's at index 0
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = poseDetector()
    count = 0
    dir = 0
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture frame from webcam.")
            break
        
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            per = int(np.interp(angle, (210, 310), (0, 100)))
            bar = int(np.interp(angle, (220, 310), (650, 100)))

            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, bar), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{per} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        cv2.imshow("Image", img)

        # Key event handler to close the webcam feed
        key = cv2.waitKey(1)
        if key == 27:  # Press 'esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
