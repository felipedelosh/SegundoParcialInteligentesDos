"""
FelipedelosH&SantiguzmaN
This is a secord partial of AI.2




"""
from kamera import *
import os
from os import scandir

class Software():
    def __init__(self) -> None:
        # Verify if exists Neuronal archives
        self.existsModelsFitNames = False
        self.existsModelsFitNames = self.verifyIFExistsModelFitFile("model_a.h5")
        print("El modelo A esta cargado: ", self.existsModelsFitNames)
        self.existsModelsFitNames = self.verifyIFExistsModelFitFile("model_b.h5")
        print("El modelo B esta cargado: ", self.existsModelsFitNames)
        self.existsModelsFitNames = self.verifyIFExistsModelFitFile("model_b.h5")
        print("El modelo C esta cargado: ", self.existsModelsFitNames)
        # Init kamera
        self.kamera = Kamera()
        print("Launching kamera Press C key to take a picture.")
        self.kamera.launchKamera()
        
        

    def verifyIFExistsModelFitFile(self, model_fit_name):
        try:
            path = os.getcwd() + "/models/"
            
            filesNames = []
            
            for i in scandir(path):
                if i.is_file():
                    if ".h5" in i.name:

                        filesNames.append(i.name)

            return model_fit_name in filesNames
        except:
            return False

s = Software()