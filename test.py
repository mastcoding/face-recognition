from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import shutil
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import pkg_resources
from mtcnn.exceptions import InvalidImage
from mtcnn.network.factory import NetworkFactory
import pandas as pd
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.xception import Xception, preprocess_input
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import warnings
from keras.callbacks import ModelCheckpoint 
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
# Suppress general Python warnings (optional)
warnings.filterwarnings('ignore')
def get_files(directory):   
  if not os.path.exists(directory):
    return 0
  count=0
  # crawls inside folders
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count
train_data_dir = 'Train_Images'
train_samples =get_files(train_data_dir)
num_classes=len(glob.glob(train_data_dir+"/*")) 
print(num_classes,"Classes")
print(train_samples,"Train images")
IMG_SIZE= (224, 224)
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen=ImageDataGenerator(rescale=1./255)
input_shape=(224,224,3)
train_generator =test_datagen.flow_from_directory(train_data_dir,target_size=(224,224),batch_size=12,
                                              class_mode="categorical")
categories = list(train_generator.class_indices.keys())
print(categories)
model_final = tf.keras.models.load_model('Myface_Model.h5')
##path='E:/New folder/Youtube_2023/Face_Detection_Recognition/face_recognition_project-main/Train_Images/Suresh/1.bmp'
##img = plt.imread(path)
##plt.imshow(img)
###Prediction
##test_image = tf.keras.preprocessing.image.load_img(path, target_size = (224,224))
##test_image = tf.keras.preprocessing.image.img_to_array(test_image)
##test_image = np.expand_dims(test_image, axis = 0)
##test_image = test_image/255.0
##prediction = model_final.predict(test_image)
##print(categories[np.argmax(prediction)])

##from win32com.client import Dispatch
##
##def speak(str1):
##    speak=Dispatch(("SAPI.SpVoice"))
##    speak.Speak(str1)

video=cv2.VideoCapture(1)
##facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
detector = MTCNN()
##with open('data/names.pkl', 'rb') as w:
##    LABELS=pickle.load(w)
##with open('data/faces_data.pkl', 'rb') as f:
##    FACES=pickle.load(f)
##
##print('Shape of Faces matrix --> ', FACES.shape)
##knn=KNeighborsClassifier(n_neighbors=2)
##knn.fit(FACES, LABELS)

imgBackground=cv2.imread("background.png")

COL_NAMES = ['NAME', 'DATE', 'TIME']
attendance=[]
while True:
##    source_df = pd.read_csv("Attendance_Rep.csv")
##    result_df = source_df.drop_duplicates(inplace=True)
    ret,frame=video.read()
##    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_faces(frame)
##    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for result in faces:
        x,y,w,h = result['box']
##    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (224,224))
        resized_img = resized_img / 255.0
        test_image = np.expand_dims(resized_img, axis=0)
##        test_image = tf.keras.preprocessing.image.load_img(resized_img, target_size = (224,224))
        test_image = tf.keras.preprocessing.image.img_to_array(resized_img)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image/255.0
        prediction = model_final.predict(test_image)
        output=categories[np.argmax(prediction)]
        print(output)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance_Rep.csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance.append(str(output))
        ftime=str(timestamp)
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame",imgBackground)
    k=cv2.waitKey(1)
    if k==ord('o'):
        test_list1 = list(set(attendance))
        print(test_list1)
        if exist:
            with open("Attendance_Rep.csv", "+a") as csvfile:
                for ii in range(0,len(test_list1)):
                    test_list=[str(test_list1[ii]), str(date), str(ftime)]
                    writer=csv.writer(csvfile)
                    writer.writerow(test_list)
            test_list.clear()
            csvfile.close()
        else:
            with open("Attendance_Rep.csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            test_list.clear()
            csvfile.close()
        time.sleep(5)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

