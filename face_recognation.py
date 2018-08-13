# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:28:58 2018

@author: 周宝航
"""

import cv2
import scipy.io as sio
import numpy as np
import numpy.linalg as la
from PIL import Image
import os


class FaceRecognition(object):
    
    def __init__(self):
        
        # Face Cascade Classifier Path
        self.CascadeClassifierPath = 'haarcascade_frontalface_default.xml'
        # Face Classfier
        self.faceCascade = cv2.CascadeClassifier(self.CascadeClassifierPath)
        # pca model path
        self.pca_model_path = "data\\pca_model.mat"
        # user face image path
        self.user_image_path = "data\\user\\"
        # user names
        self.user_names = []
        # face image size
        self.img_size = (92, 112)
        # load pca model
        self.__load_model()
        # user image matrix
        self.padding = (23, 28)
        # load user images
        self.load_user_images()
        
        
    def __load_model(self):
        """
        load pca model from the relative path
        contains : mean_face vector , pca space matrix
        """
        data = sio.loadmat(self.pca_model_path)
        # mean, W, V
        self.mean_face, self.V = data['mean_face'], data['V']
        
        
    def load_user_images(self):
        """
        load users' face images from relative path
        overoll image matrix to column vector
        store users' image vector to self.user_matrix
        """
        imgs = os.listdir(self.user_image_path)
        if len(imgs) > 0:
            self.user_names.clear()
            # load user images
            user_image_faces_matrix = None
            for i, img_file in enumerate(imgs):
                self.user_names.append(img_file.split('.')[0])
                img = np.array(Image.open(self.user_image_path + img_file).convert('L')).reshape([-1, 1])
#                img = np.array(cv2.imread(self.user_image_path + img_file, cv2.IMREAD_GRAYSCALE)).reshape([-1, 1])
                if i == 0:
                    user_image_faces_matrix = img
                else:
                    user_image_faces_matrix = np.c_[user_image_faces_matrix, img]
            self.user_matrix = self.V.T.dot(user_image_faces_matrix - self.mean_face)

    def __recognize(self, image, face):
        """
        the system approves the user's identity according to his face
        """
        padW, padH = self.padding
        name = ''
        try:
            (x, y, w, h) = face
            image = Image.fromarray(image).crop((x-padW, y-padH, x+padW+w, y+padH+h)).resize(self.img_size)
            img_vec = self.V.T.dot(np.array(image).reshape([-1, 1]) - self.mean_face)
            distances = np.array([la.norm(img_vec - self.user_matrix[:, j].reshape([-1, 1])) \
                         for j in range(self.user_matrix.shape[1])])
            min_dis = np.min(distances)

            index = np.where(distances == min_dis)[0][0]
            # print(min_dis, index)
            name = self.user_names[index]
        except Exception:
            print("识别异常")
        
        return name
        
    def findFaces(self, image):
        """
        the system find the faces in the camera's photo
        tag names and face regions 
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5,5),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        names = []
        for face in faces:
            (x,y,w,h) = face
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            names.append(self.__recognize(gray, face))  
        return (image, gray, faces, names)
