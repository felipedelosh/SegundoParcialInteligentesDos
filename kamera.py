"""
felipedelosh
This is a class to capture a camera

"""
import cv2
import numpy as np

class Kamera():
    def __init__(self) -> None:
        self.default_kamera = 0
        self.host_kamera = ''
        # Cameras
        self.kameraWhiteAndBlack = None
        self.kameraHSV = None
        self.kameraIAVision = None
        self.kameraBorders = None
        #
        self.cap = cv2.VideoCapture(self.default_kamera)

    def nothing(self, x):
        """
        No have idea for this method.
        if delete it the program NOT RUN
        if try to return None inside createTraker the program NOT RUN
        """
        pass


    def generateSlidersPanel(self):
        nameWindow ="Controllers: ->"
        cv2.namedWindow(nameWindow)
        cv2.createTrackbar("min",nameWindow,0,255,self.nothing)
        cv2.createTrackbar("max",nameWindow,1,100,self.nothing)
        cv2.createTrackbar("kernel",nameWindow,0,255,self.nothing)
        cv2.createTrackbar("areaMin",nameWindow,500,10000,self.nothing)
       

    def launchKamera(self):
        self.generateSlidersPanel()
        self._initKamera()

    def filters(self, image, name_image='NULL'):
        self.showInBlackAndWhite(image)
        self.showInHSV(image)
        self.showIAVision(image)

    def showInBlackAndWhite(self, image):
        self.kameraWhiteAndBlack = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gris", self.kameraWhiteAndBlack)

    def showInHSV(self, image):
        self.kameraHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", self.kameraHSV)

    def showIAVision(self, image):
        nameWindow ="Controllers: ->"
        min=cv2.getTrackbarPos("min",nameWindow)
        max=cv2.getTrackbarPos("max",nameWindow)
        self.kameraBorders=cv2.Canny(self.kameraWhiteAndBlack,min,max)
        tamañokernel=cv2.getTrackbarPos("kernel",nameWindow)
        kernel=np.ones((tamañokernel,tamañokernel),np.uint8)
        self.kameraBorders=cv2.dilate(self.kameraBorders,kernel)
        cv2.imshow("Borders",self.kameraBorders)
        objetos,jerarquias=cv2.findContours(self.kameraBorders,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        self.kameraIAVision = np.zeros_like(self.kameraBorders)
        cv2.drawContours(self.kameraIAVision, objetos, -1, 255, 1)
        cv2.imshow('IA visión', self.kameraIAVision)





    def _initKamera(self):
        cont = 0
        while(True):
            ret,frame = self.cap.read()
            self.filters(frame) # Init another images
            cv2.imshow('Kamera del loko a todo color...',frame)
            if not ret:
                break
            k=cv2.waitKey(1)
            if k%256 == 99 :
                cont += 1
                img_name ="imagen_{}.jpg".format(img_counter)
                cv2.imwrite(img_name,frame)

                img_counter += 1

        self.cap.release()
        cv2.destroyAllWindows()
