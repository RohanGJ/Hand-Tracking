import cv2 
import mediapipe as mp
import time 


class handDetection():
    def __init__(self, mode = False, maxHands = 2, Detectconf = 0.5, trackingconf = 0.5):
        self.mode                = mode
        self.maxhands            = maxHands
        self.detectconf          = Detectconf
        self.trackingconf        = trackingconf
        
        self.mphands = mp.solutions.hands
        self.hands   = self.mphands.Hands(static_image_mode =self.mode,
                                          max_num_hands = self.maxhands,
                                          min_detection_confidence = self.detectconf,
                                          min_tracking_confidence =self.trackingconf)
        self.mpDraw  = mp.solutions.drawing_utils

    def findhands(self,img,draw = True):
        imgRBG       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultingimg = self.hands.process(imgRBG)
        if self.resultingimg.multi_hand_landmarks:
            for handlms in self.resultingimg.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
        return img
   
    def findpositions(self,img,position = 0,handNo = 0, draw = True):
        lmList = []
        if self.resultingimg.multi_hand_landmarks:
            myHand = self.resultingimg.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                for id,cx,cy in lmList:
                    if draw and id == position:
                        cv2.circle(img, (cx,cy), 5, (0,255,255), cv2.FILLED)
        return lmList 

def main():
    Capture = cv2.VideoCapture(0)
    cTime   = 0
    pTime   = 0
    detector = handDetection()
    while True:
        success, img = Capture.read()
        position = 8
        img = detector.findhands(img,draw= False)
        lmlist = detector.findpositions(img,position)
        if len(lmlist) != 0:
            print(lmlist[position])
        cTime = time.time()
        fps   = 1/(cTime - pTime) 
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,200,255),2)               


        cv2.imshow("Image",img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()