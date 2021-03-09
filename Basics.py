import cv2
import numpy as np
import face_recognition

#! step 1, hanya bisa detect 1 face
imgAdi = face_recognition.load_image_file('ImageBasic/Adi-1.jpeg')
imgAdi = cv2.cvtColor(imgAdi, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/nindya2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#! step 2, locate the proper face location
faceLoc = face_recognition.face_locations(imgAdi)[0]
encodeAdi = face_recognition.face_encodings(imgAdi)[0] # encode the image
cv2.rectangle(imgAdi, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 255, 0), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeAdiTest = face_recognition.face_encodings(imgTest)[0] # encode the image
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 255, 0), 2)




cv2.imshow('Adipati M.A.', imgAdi)
cv2.imshow('Adi Test', imgTest)
cv2.waitKey(0)
