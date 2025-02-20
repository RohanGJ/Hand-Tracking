import mediapipe as mp
import cv2 
import time


capture = cv2.VideoCapture("C:/Users/Rohan/Downloads/face1.mp4")

mpDraw          = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
mpFace          = mpFaceDetection.FaceDetection()
pTime = 0

while True:
    success, img = capture.read() 
    imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    imgresulting = mpFace.process(imgRGB)
    
    if imgresulting.detections:
        for id,detection in enumerate(imgresulting.detections):
            mpDraw.draw_detection(img,detection)

    cTime = time.time()
    fps   = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(5)