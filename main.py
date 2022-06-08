"""
This is a secord partial of AI.2




"""
from kamera import *


class Software():
    def __init__(self) -> None:
        self.kamera = Kamera()
        self.kamera.launchKamera()


s = Software()