import cv2
import numpy as np
import face_recognition

#! step 1, hanya bisa detect 1 face
imgAdi = face_recognition.load_image_file('ImageBasic/Adi-1.jpeg')
imgAdi = cv2.cvtColor(imgAdi, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/nindya2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)




cv2.imshow('Adipati M.A.', imgAdi)
cv2.imshow('Adi Test', imgTest)
cv2.waitKey(0)
