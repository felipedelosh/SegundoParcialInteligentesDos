"""
FelipedelosH



"""

import base64
from typing_extensions import Self
import tensorflow as tf
import keras
import numpy as np
import cv2
from Prediction import Prediction
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import time
from tensorflow.python.keras.models import load_model
import os
from os import scandir

class ArtificialInteligence():
    def __init__(self) -> None:
        self.project_path = str(os.path.dirname(os.path.abspath(__file__))) # the folder 

    def model_a(self):
        width = 128
        height = 128
        pixels = width * height

        num_channels = 1    # RGB -> 3  // Black and white-> 1 
        img_shape = (width, height, num_channels)

        # Cant elementos a clasifica
        num_clases = 10
        cantidad_datos_entenamiento =  [8,8,8,8,8,8,8,8,8,8,8,8]
        cantidad_datos_pruebas = [5,5,5,5,5,5,5,5,5,5]


        path = self.project_path+"/DATASET/"
        images, probabilities = self.loadData(path, width, height)



    def loadData(self, path, width, height):
        """
        Enter a folder path

        The DATASET Contains 8 pic in 12 folders.
        Read al .JPEG images with inteligentID name.
        """
        imagenesCargadas=[]
        valorEsperado=[]

        for datafolderID in range(1, 13): # In folder DATASET exists 12 folders
            for idImage in range(1, 9): # Inside DATASET folder exist 8 pictures
                imagePATH = path + str(datafolderID) + "/" + str(datafolderID) + "_" + str(idImage) + ".JPEG"
                #print("Estoy en:", imagePATH)
                imagen=cv2.imread(imagePATH)
                #print("Lei la imagen", imagePATH)
                imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

                imagen = cv2.resize(imagen, (width, height))
                imagen=imagen.flatten()
                imagen=imagen/255
                imagenesCargadas.append(imagen)
                probabilidades=np.zeros(13)
                
                probabilidades[datafolderID]=1
                valorEsperado.append(probabilidades)
        imagenes_entrenamiento = np.array(imagenesCargadas)
        valores_esperados = np.array(valorEsperado)

        print("CANTIDAD DE IMAGINES", len(imagenes_entrenamiento))
        print("CANTIDAD DE VALORES", len(valores_esperados))

        return imagenes_entrenamiento, valores_esperados


a = ArtificialInteligence()
a.model_a()

    