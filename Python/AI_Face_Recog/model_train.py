import cv2 #importing modules
import numpy as np
import os
from PIL import Image

path='samples' # path of 'samples' directory 
recogniser = cv2.face.LBPHFaceRecognizer_create() # using face recognizer
detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml') # getting xml data to locate face on camera.

# defining function to assign labels to Images.
def Image_and_Label(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] # accessing all images in 'samples' directory
    faceSamples=[]
    ids = []
    for imagePath in imagePaths: # getting all images and assinging labels

        gray_img = Image.open(imagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')
        id = int((os.path.split(imagePath)[-1]).split(".")[1])
        faces = detector.detectMultiScale(img_arr)
        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

os.mkdir("model")
print("Training model........")

faces, ids = Image_and_Label(path)
recogniser.train(faces, np.array(ids)) # training model for face_recognition.py

recogniser.write('model/model.yml') # output as model -- 'trainer.yml'
print("Model Trained.....")
