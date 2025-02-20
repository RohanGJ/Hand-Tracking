import mediapipe as mp
import cv2 
import time


capture = cv2.VideoCapture("C:/Users/Rohan/Downloads/face1.mp4")

mpDraw          = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
mpFace          = mpFaceDetection.FaceDetection(0.75)
pTime = 0

while True:
    success, img = capture.read() 
    imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    imgresulting = mpFace.process(imgRGB)
    
    if imgresulting.detections:
        for id,detection in enumerate(imgresulting.detections):
            #mpDraw.draw_detection(img,detection)
            BoundBoxMain = detection.location_data.relative_bounding_box
            ih, iw, ic   = img.shape
            BoundBox     = int(BoundBoxMain.xmin * iw), int(BoundBoxMain.ymin * ih),\
                           int(BoundBoxMain.width * iw), int(BoundBoxMain.height * ih)
            cv2.rectangle(img, BoundBox, (0,0,255), 2 )
            cv2.putText(img,f"{int(detection.score[0]*100)}",
                        (BoundBox[0],BoundBox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    cTime = time.time()
    fps   = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(5)