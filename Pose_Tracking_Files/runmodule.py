import cv2
import time
import mediapipe as mp
import posetrackinngmodule as PTM

detector = PTM.posedetection()
capture = cv2.VideoCapture('C:/Users/Rohan/Downloads/fooot2.mp4')
#capture  = cv2.VideoCapture(0)
pTime = 0
position = 0
while True:
    success, img = capture.read()
    img = detector.findingpose(img, draw= False)

    lmLisr = detector.getposition(img, position)
    print(success)
    cTime = time.time()
    fps   = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(70,58), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("IMAGE", img)
    cv2.waitKey(5)