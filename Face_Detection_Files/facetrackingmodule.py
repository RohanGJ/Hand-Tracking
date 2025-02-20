import cv2
import time
import mediapipe as mp

class FaceDetector():
    def __init__(self, minimumdetectionconf = 0.5):
        self.minconf         = minimumdetectionconf    

        self.mpDraw          = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpFace          = self.mpFaceDetection.FaceDetection(self.minconf)
    
    def findFaces(self, img, draw = True):
        imgRGB       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        self.imgresulting = self.mpFace.process(imgRGB)

        BoundingBoxes = []

        if self.imgresulting.detections:
            for id, detection in enumerate(self.imgresulting.detections):
                BoundBoxMain = detection.location_data.relative_bounding_box
                ih, iw, ic   = img.shape
                BoundBox     = int(BoundBoxMain.xmin * iw), int(BoundBoxMain.ymin * ih),\
                            int(BoundBoxMain.width * iw), int(BoundBoxMain.height * ih)
                BoundingBoxes.append([BoundBox, detection.score[0]])
                if draw:
                    img = self.BBOX(img,BoundBox)

                cv2.putText(img,f"{int(detection.score[0]*100)}",
                            (BoundBox[0],BoundBox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        return img, BoundingBoxes

    def BBOX(self,img,BoundingBox, l = 30, thickness = 5):
        x, y, w, h = BoundingBox
        x1, y1     = x+w, y+h
        cv2.rectangle(img, BoundingBox, (0,0,255), 2 )
        #Top Left Corner
        cv2.line(img,(x,y),(x+l,y),(0,0,255),thickness)
        cv2.line(img,(x,y),(x,y+l),(0,0,255),thickness)
        #Top Right Corner
        cv2.line(img,(x1,y),(x1-l,y),(0,0,255),thickness)
        cv2.line(img,(x1,y),(x1,y+l),(0,0,255),thickness)
        #Bottom Left
        cv2.line(img,(x,y1),(x+l,y1),(0,0,255),thickness)
        cv2.line(img,(x,y1),(x,y1-l),(0,0,255),thickness)
        #Top Right Corner
        cv2.line(img,(x1,y1),(x1-l,y1),(0,0,255),thickness)
        cv2.line(img,(x1,y1),(x1,y1-l),(0,0,255),thickness)
        return img


def main():

    detector = FaceDetector()
    capture  = cv2.VideoCapture("C:/Users/Rohan/Downloads/face2.mp4")
    pTime    = 0
    while True:
        success, img = capture.read() 
        img, BBox = detector.findFaces(img)
        cTime = time.time()
        fps   = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()