"""
FelipedelosH

python -m pip install --ignore-installed --upgrade TensorFlow


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

    def cargarDatos(self, fase, numeroCategorias, limite, width, height):
        imagenesCargadas=[]
        valorEsperado=[]

        print("Este es el numero de categoias", numeroCategorias)
        for categoria in range(1, numeroCategorias+1): # Start #1 ID folder and end #10 id folder //DATASET
            for idImagen in range(1, limite[categoria-1]): # Start in img 1 and end in img 27>> example 3_27.JPG 
                if 'TESTDATA/' in fase:
                    ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
                else:
                    ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".JPG"
                
                imagen=cv2.imread(ruta) # Se lee la imagen desde disco duro
                imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # Se convierte a blanco y negro
                imagen = cv2.resize(imagen, (width, height)) # Se más pequeña
                imagen=imagen.flatten()
                imagen=imagen/255
                imagenesCargadas.append(imagen)
                probabilidades=np.zeros(numeroCategorias)
                probabilidades[categoria-1]=1
                valorEsperado.append(probabilidades)
                print("cargada img: "+ruta + " ... Probabilidades: " + str(probabilidades) + " ... categoria: " + str(categoria))

        imagenes_entrenamiento = np.array(imagenesCargadas)
        valores_esperados = np.array(valorEsperado)

        print("CANTIDAD DE IMAGINES", len(imagenes_entrenamiento))
        print("CANTIDAD DE VALORES", len(valores_esperados))

        return imagenes_entrenamiento, valores_esperados

    ####### Funciones requeridas
    def modelo1(self):
        width = 128
        height = 128
        pixeles = width * height

        # Si es a blanco y negro es -> 1 si es RGB es -> 3
        num_channels = 1
        img_shape = (width, height, num_channels)

        # Cant elementos a clasifica
        num_clases = 10
        cantidad_datos_entenamiento =  [28,28,28,28,28,28,28,28,28,28]
        cantidad_datos_pruebas = [6,6,6,6,6,6,6,6,6,6]
        

        ##Carga de los datos
        imagenes, probabilidades =  self.cargarDatos("DATASET/", num_clases, cantidad_datos_entenamiento, width, height)
        print(imagenes)


        model = Sequential()

        # Capa de entrada 128,128 a blanco y negro
        model.add(InputLayer(input_shape=(pixeles,)))

        # Re armar la imagen
        model.add(Reshape(img_shape))

        # Capas convolucionales
        model.add(Conv2D(kernel_size=8, strides=2, filters=8, padding="same", activation="relu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=8,strides=2, filters=26, padding="same", activation="relu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        
        # Capa de salida
        model.add(Dense(num_clases, activation="softmax"))

        # Traducir de keras a tensorflow
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=60)
        # Pruebas 
        imagenes_prueba, probabilidades_prueba = self.cargarDatos("TESTDATA/", num_clases, cantidad_datos_pruebas, width, height)
        resultados=model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
        print("METRIC NAMES", model.metrics_names)
        print("RESULTADOS", resultados)


        ## Guardar el modelo
        ruta="models/model_a.h5"
        model.save(ruta)

        #Estructura de la red

        model.summary()

        metricResult = model.evaluate(x=imagenes, y=probabilidades)

        scnn_pred = model.predict(imagenes_prueba, batch_size=60, verbose=1)
        scnn_predicted = np.argmax(scnn_pred, axis=1)

        # Creamos la matriz de confusión
        scnn_cm = confusion_matrix(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)

        # Visualiamos la matriz de confusión
        scnn_df_cm = pd.DataFrame(scnn_cm, range(10), range(10))
        plt.figure(figsize=(20, 14))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
        plt.show()

        scnn_report = classification_report(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)
        print("SCNN REPORT", scnn_report)
    def modelo2(self):
        width = 128
        height = 128
        pixeles = width * height

        # Si es a blanco y negro es -> 1 si es RGB es -> 3
        num_channels = 1
        img_shape = (width, height, num_channels)

        # Cant elementos a clasifica
        num_clases = 10
        cantidad_datos_entenamiento =  [28,28,28,28,28,28,28,28,28,28]
        cantidad_datos_pruebas = [5,5,5,5,5,5,5,5,5,5]

        ##Carga de los datos
        imagenes, probabilidades = self.cargarDatos("DATASET/", num_clases, cantidad_datos_entenamiento, width, height)
        print(imagenes)


        model = Sequential()

        # Capa de entrada
        model.add(InputLayer(input_shape=(pixeles,)))

        # Re armar la imagen
        model.add(Reshape(img_shape))

        # Capas convolucionales
        model.add(Conv2D(kernel_size=2, strides=2, filters=40, padding="same", activation="elu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(kernel_size=2, strides=2, filters=50, padding="same", activation="elu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))

        # Capa de salida
        model.add(Dense(num_clases, activation="softmax"))

        # Traducir de keras a tensorflow
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=50)
        # Pruebas
        imagenes_prueba, probabilidades_prueba = self.cargarDatos("TESTDATA/", num_clases, cantidad_datos_pruebas, width, height)
        resultados=model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
        print(model.metrics_names)
        print(resultados)


        ## Guardar el modelo
        ruta="models/model_b.h5"
        model.save(ruta)

        #Estructura de la red

        model.summary()

        metricResult = model.evaluate(x=imagenes, y=probabilidades)

        scnn_pred = model.predict(imagenes_prueba, batch_size=150, verbose=1)
        scnn_predicted = np.argmax(scnn_pred, axis=1)

        # Creamos la matriz de confusión
        scnn_cm = confusion_matrix(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)

        # Visualiamos la matriz de confusión
        scnn_df_cm = pd.DataFrame(scnn_cm, range(9), range(9))
        plt.figure(figsize=(20, 14))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
        plt.show()

        scnn_report = classification_report(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)
        print("SCNN REPORT", scnn_report)

    def modelo3(self):
        width = 128
        height = 128
        pixeles = width * height

        # Si es a blanco y negro es -> 1 si es RGB es -> 3
        num_channels = 1
        img_shape = (width, height, num_channels)

        # Cant elementos a clasifica
        num_clases = 10
        cantidad_datos_entenamiento =  [28,28,28,28,28,28,28,28,28,28]
        cantidad_datos_pruebas = [5,5,5,5,5,5,5,5,5,5]

        ##Carga de los datos
        imagenes, probabilidades = self.cargarDatos("DATASET/", num_clases, cantidad_datos_entenamiento, width, height)
        print(imagenes)


        model = Sequential()

        # Capa de entrada
        model.add(InputLayer(input_shape=(pixeles,)))

        # Re armar la imagen
        model.add(Reshape(img_shape))

        # Capas convolucionales
        model.add(Conv2D(kernel_size=8, strides=2, filters=30, padding="same", activation="selu", name="capa_1"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        model.add(Conv2D(kernel_size=8, strides=2, filters=36, padding="same", activation="selu", name="capa_2"))
        model.add(MaxPool2D(pool_size=2, strides=2))

        # Aplanamiento
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))

        # Capa de salida
        model.add(Dense(num_clases, activation="softmax"))

        # Traducir de keras a tensorflow
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x=imagenes, y=probabilidades, epochs=48, batch_size=150)
        # Pruebas
        imagenes_prueba, probabilidades_prueba = self.cargarDatos("TESTDATA/", num_clases, cantidad_datos_pruebas, width, height)
        resultados=model.evaluate(x=imagenes_prueba, y=probabilidades_prueba)
        print(model.metrics_names)
        print(resultados)


        ## Guardar el modelo
        ruta="models/model_c.h5"
        model.save(ruta)

        #Estructura de la red

        model.summary()

        metricResult = model.evaluate(x=imagenes, y=probabilidades)

        scnn_pred = model.predict(imagenes_prueba, batch_size=150, verbose=1)
        scnn_predicted = np.argmax(scnn_pred, axis=1)

        # Creamos la matriz de confusión
        scnn_cm = confusion_matrix(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)

        # Visualiamos la matriz de confusión
        scnn_df_cm = pd.DataFrame(scnn_cm, range(9), range(9)) # Por que son 12 categorias
        plt.figure(figsize=(20, 14))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
        plt.show()

        scnn_report = classification_report(np.argmax(probabilidades_prueba, axis=1), scnn_predicted)
        print("SCNN REPORT", scnn_report)



    def imageToText(self):
        imagen_seleccionada=cv2.imread("dataset/test/3/3_4.jpg")
        retral, buffer = cv2.imencode('.jpg', imagen_seleccionada)
        jpg_as_test = base64.encode(buffer)
        while True:
            cv2.imshow('imagen', imagen_seleccionada)
            k=cv2.waitKey(30) & 0xff
            if k==27:
                break
        cv2.destroyAllWindows()


#FelipedelosH

i = ArtificialInteligence()
i.modelo1()
print("===========Fin modelo 1==========")
#i.modelo2()
print("===========Fin modelo 2==========")
#i.modelo3()
print("===========Fin modelo 3==========")
"""
"""
