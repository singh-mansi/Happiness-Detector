# -*- coding: utf-8 -*-
import cv2

#Loading cascades
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(frame, (fx,fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        region_gray = gray[fy:fy+fh, fx:fx+fw]
        region_frame = frame[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(region_gray, 1.1, 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(region_frame, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(region_gray, 1.7, 22)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(region_frame, (sx,sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _,frame= video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()