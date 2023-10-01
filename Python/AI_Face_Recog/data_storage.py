import cv2
import os


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # accessing pc webcam

cam.set(3, 600)  # set width of camera
cam.set(4, 480)  # set height of camera

detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # getting xml data to locate face on camera.

face_id = input("Face ID: ")  # it is required for face_recognition.py

print("Taking pictures.......")
count = 0
os.mkdir("samples")
# taking and saving pictures in "samples" folder
while True:
    ret, img = cam.read()  # reading images from camera
    # converts all images in Grayscale format
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detects face with help of .xml file
    faces = detector.detectMultiScale(converted_image, 1.3, 5)

    for (x, y, w, h) in faces:  # creates square-border around face and saves the square imgage

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite("samples/face."+str(face_id) + str(".") +
                    str(count)+".jpg", converted_image[y:y+h, x:x+w])

        cv2.imshow("picture", img)  # displays camera output
        count += 1  # increases count

    if cv2.waitKey(100) & 0xff == 27:  # exits when 'Escape' is pressed
        break
    elif count >= 100:
        break

print("Samples taken......")
cam.release()
cv2.destroyAllWindows()
