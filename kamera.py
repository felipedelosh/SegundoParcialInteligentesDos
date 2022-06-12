"""
felipedelosh
This is a class to capture a camera

"""
import cv2
from cv2 import namedWindow
from ArtificialInteligence import *
import numpy as np

class Kamera():
    def __init__(self) -> None:
        self.default_kamera = 0
        self.host_kamera = 'http://192.168.20.62:4747/video' # I hav anndriod and install DroidCAM
        # Config kamera
        self.nameWindow = "Controllers: ->"
        cv2.namedWindow(self.nameWindow)
        cv2.createTrackbar("min",self.nameWindow,0,255,self.nothing)
        cv2.createTrackbar("max",self.nameWindow,1,100,self.nothing)
        cv2.createTrackbar("kernel",self.nameWindow,0,255,self.nothing)
        cv2.createTrackbar("areaMin",self.nameWindow,500,10000,self.nothing)
        # Cameras
        self.kameraWhiteAndBlack = None
        self.kameraHSV = None
        self.kameraIAVision = None
        self.kameraBorders = None
        # IA
        self.IA = ArtificialInteligence()
        #
        self.cap = cv2.VideoCapture(self.host_kamera)

    def nothing(self, x):
        """
        No have idea for this method.
        if delete it the program NOT RUN
        if try to return None inside createTraker the program NOT RUN
        """
        pass


    def generateSlidersPanel(self):
        pass
       

    def launchKamera(self):
        self.generateSlidersPanel()
        self._initKamera()

    def filters(self, image, name_image='NULL'):
        self.showInBlackAndWhite(image)
        #self.showInHSV(image)
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
        img_counter = 0
        while(True):
            ret,frame = self.cap.read()
            self.filters(frame) # Init another images
            cv2.imshow('Kamera del loko a todo color...',frame)
            if not ret:
                break
            k=cv2.waitKey(1)
            if k%256 == 99 :
                cont = cont + 1
                print("Take a picture: ")
                img_name ="imagen_{}.jpg".format(img_counter)
                print(img_name)
                cv2.imwrite(img_name,frame) # Save a img in a root project
                self.detectarForma(frame, frame, img_name)

                img_counter = img_counter + 1
                acum = self.probarModelo(img_name)

                self.mostrarAcumulado(acum, frame)


        self.cap.release()
        cv2.destroyAllWindows()

    def detectarForma(self,frame,imagen,img_name):
        nameWindow = "IA detection"
        imagenGris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grises",imagenGris)
        min=cv2.getTrackbarPos("min",self.nameWindow)
        max=cv2.getTrackbarPos("max",self.nameWindow)
        bordes=cv2.Canny(imagenGris,min,max)
        tamañokernel=cv2.getTrackbarPos("kernel",self.nameWindow)
        kernel=np.ones((tamañokernel,tamañokernel),np.uint8)
        bordes=cv2.dilate(bordes,kernel)
        cv2.imshow("Bordes",bordes)
        objetos,jerarquias=cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        nuevaImagen = np.zeros_like(bordes)
        cv2.drawContours(nuevaImagen, objetos, -1, 255, 1)
        cv2.imshow('contornos', nuevaImagen)
        areas=self.calcularAreas(objetos)
        i=0
        areaMin=cv2.getTrackbarPos("areaMin",self.nameWindow)
        for objetoActual in objetos:
            if areas[i]>=areaMin:

                vertices=cv2.approxPolyDP(objetoActual,0.025*cv2.arcLength(objetoActual, closed=True),True)

                if len(vertices) == 4 :
                    x, y, w, h = cv2.boundingRect(vertices)
                    new_img=frame[y:y+h,x:x+w]
                    cv2.imwrite(img_name,new_img)
            i = i+1


    def calcularAreas(self, objetos):
        areas=[]
        for objetoActual in objetos:
            areas.append(cv2.contourArea(objetoActual))
        return areas

    def probarModelo(self, imagen):
        categorias = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

        width = 128
        height = 128

        miModeloCNN = Prediction("models/model_c.h5", width, height)
        imagen_seleccionada = cv2.imread(imagen, 0)

        categodria_predicha = miModeloCNN.predecir(imagen_seleccionada)

        return categodria_predicha


    def mostrarAcumulado(self, acum, img):
        cv2.putText(img, 'Acumulado: + {}'.format(acum), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Imagen", img)