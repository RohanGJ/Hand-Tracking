import cv2
import time
import mediapipe as mp

class posedetection():
    def __init__(self, mode = False, upBody = False, smooth = True, detectconf = 0.5, trackingconf = 0.5):
        self.mode         = mode
        self.upBody       = upBody
        self.smooth       = smooth
        self.detectconf   = detectconf
        self.trackingconf = trackingconf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose 
        self.pose   = self.mpPose.Pose(static_image_mode=self.mode,
                                       smooth_landmarks=self.smooth,
                                       enable_segmentation=self.upBody,
                                       min_detection_confidence=self.detectconf,
                                       min_tracking_confidence=self.trackingconf,
                                       model_complexity=1,
                                       smooth_segmentation=True)
        
    
    def findingpose(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultingimage = self.pose.process(imgRGB)
    
        if self.resultingimage.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.resultingimage.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img 
    
    def getposition(self, img,position = 0, draw = True):
        lmList = []
        if self.resultingimage.pose_landmarks:
            for id, lm in enumerate(self.resultingimage.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h )
                lmList.append([id, cx, cy])
                for id,cx,cy in lmList:
                    if draw and position == id:
                        cv2.circle(img, (cx,cy), 4, (0,0,255), cv2.FILLED)



def main():
    
    detector = posedetection()
    
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

if __name__ == "__main__":
    main()