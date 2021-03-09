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

#! step 3, compare the encoding (128 measurements of both the faces)
# at the backend using linear SVM to find out match or not
results = face_recognition.compare_faces([encodeAdi], encodeAdiTest)
# can see the distance of similarity
faceDis = face_recognition.face_distance([encodeAdi], encodeAdiTest) # bisa digunakan saat ingin menggunakan train foto yg paling mirip
print(results, faceDis)
cv2.putText(imgTest, f'{results[0]} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Adipati M.A.', imgAdi)
cv2.imshow('Adi Test', imgTest)
cv2.waitKey(0)
