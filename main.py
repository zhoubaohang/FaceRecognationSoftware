# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:57:19 2018

@author: 周宝航
"""

from main_window import MainWindow
from face_recognation import FaceRecognition

fr = FaceRecognition()

wm = MainWindow(fr)
wm.launch()