"""
FelipedelosH



"""

import base64
import numpy
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import keras
import numpy as np
import cv2

class Prediction():
    def __init__(self, ruta, width, height):
        self.ruta_modelo = ruta
        self.model = load_model(ruta)
        self.width = width
        self.heigth = height

    def predecir(self, imagen):
        #Imagen debe llega en blanco y negro
        imagen=cv2.resize(imagen,  (self.width, self.heigth))

        imagen = imagen.flatten()
        imagen = imagen / 255
        imagenesCargadas = []
        imagenesCargadas.append(imagen)
        imagenesCargadas=numpy.array(imagenesCargadas)

        resultados = self.model.predict(x=imagenesCargadas)

        print("", resultados)
        claseMayor = numpy.argmax(resultados, axis=1)

        return claseMayor[0]