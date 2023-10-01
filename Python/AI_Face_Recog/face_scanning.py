from datetime import datetime  # importing modules
import cv2
import os
import numpy as np


# os.mkdir("output_data")
filename = 'video.mp4'  # video filename
frames_per_second = 28.0  # fps
res = '480p'  # resoultion

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='720p'):
    width, height = STD_DIMENSIONS["720p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current caputre device
    # to the resulting resolution
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


# def get_video_type(filename):
#     os.chdir('output_data/')
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#         return VIDEO_TYPE[ext]
#     return VIDEO_TYPE['mp4']


# using cv2 module for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/model.yml')  # importing trained model
# getting xml data to locate face on camera.
cascadePath = 'haarcascade_frontalface_default.xml'
# getting xml data to locate face on camera.
faceCascade = cv2.CascadeClassifier(cascadePath)


id = 2
names = ['dhyey', 'unknown']  # give these as same as Face ID:

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# out = cv2.VideoWriter(filename, get_video_type(
#     filename), 25, get_dims(cam, res))
width = cam.set(3, 640)
height = cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


while True:

    ret, img = cam.read()
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        converted_img, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

        id, accuracy = recognizer.predict(converted_img[y:y+h, x:x+w])

        if accuracy < 100:
            id = names[id]
            accuracy = '{0}%'.format(round(accuracy))

            date_time = datetime.now()
            # with open('authorise_access.log', 'a') as f:
            #     data2write = f'{id} has access on {date_time}\n'
            #     f.write(data2write)
            #     f.close()

        # else:
        #     id = 'unknown'
        #     accuracy = '{0}%'.format(round(100-accuracy))
        #     date_time = datetime.now()
        #     with open('unauthorise_access.log', 'a') as f:
        #         data2write = f'{id} has access on {date_time}\n'
        #         f.write(data2write)
        #         f.close()
            # os.startfile('alert.wav')

        cv2.putText(img, str(id), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(accuracy), (x+5, y+h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(img, str(datetime.now().date()), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, str(f'{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}'),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # out.write(img)
    # print(f"Recording Saved.... {filename}")
    cv2.imshow("Face Detection", img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break


print("Tnx")
cam.release()
# out.release()
cv2.destroyAllWindows()
