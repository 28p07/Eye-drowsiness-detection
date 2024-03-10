import cv2
import os
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

eye_labels = ['Closed', 'Open']

model = load_model('eye_model.h5')

count = 0
score = 0
rpred = [99]
lpred = [99]

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred_x = model.predict(r_eye)
        rpred = np.argmax(rpred_x, axis=1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred_x = model.predict(l_eye)
        lpred = np.argmax(lpred_x, axis=1)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, frame.shape[0] - 20), font, 1, (255, 0, 0), 2, 0)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, frame.shape[0] - 20), font, 1, (255, 0, 0), 2, 0)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()