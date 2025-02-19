import cv2
import mediapipe as mp
import time 
import handtrackingmodule as HTM


Capture = cv2.VideoCapture(0)
cTime   = 0
pTime   = 0
detector = HTM.handDetection()
while True:
    success, img = Capture.read()
    position = 3
    img = detector.findhands(img)
    lmlist = detector.findpositions(img,position)
    if len(lmlist) != 0:
        print(lmlist[position])
    cTime = time.time()
    fps   = 1/(cTime - pTime) 
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)               


    cv2.imshow("Image",img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
