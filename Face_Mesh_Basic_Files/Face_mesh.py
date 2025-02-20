import cv2
import mediapipe as mp
import time 

pTime    = 0
capture  = cv2.VideoCapture("C:/Users/Rohan/Downloads/face2.mp4") 

mpDraw     = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh   = mpFaceMesh.FaceMesh(max_num_faces=1)
drawspecs  = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

while True:
    success, img = capture.read()
    imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultingimg = faceMesh.process(imgRGB)

    if resultingimg.multi_face_landmarks:
        for id,facelms in enumerate(resultingimg.multi_face_landmarks):
            mpDraw.draw_landmarks(img, facelms, mp.solutions.face_mesh.FACEMESH_TESSELATION,drawspecs,drawspecs)
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS : {int(fps)}",(70,30), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0),3)

    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)