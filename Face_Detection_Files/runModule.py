import cv2 
import time
import facetrackingmodule as FDM

capture = cv2.VideoCapture("C:/Users/Rohan/Downloads/face1.mp4")
detector = FDM.FaceDetector()
pTime = 0

while True:
    success, img = capture.read() 
    img,BBox = detector.findFaces(img)

    cTime = time.time()
    fps   = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)