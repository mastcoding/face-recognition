import cv2
import pickle
import numpy as np
import os
import shutil
from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import pkg_resources
from mtcnn.exceptions import InvalidImage
from mtcnn.network.factory import NetworkFactory
video=cv2.VideoCapture(1)
##facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0
detector = MTCNN()
name=input("Enter Your Name: ")
output_directory = 'Train_Images/'+name
import os
os.makedirs(output_directory, exist_ok=True)
fg=0
while True:
    ret,frame=video.read()
    Print(ret, type(frame))
    faces = detector.detect_faces(frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for result in faces:
        x,y,w,h = result['box']
##    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (224,224))
        resized_img1=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=20 and i%4==0:
            fg=fg+1
            faces_data.append(resized_img)
            fname='/' + str(fg)+'.bmp'
            cv2.imwrite(output_directory + fname, resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==20:
        break
video.release()
cv2.destroyAllWindows()
