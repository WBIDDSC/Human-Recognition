#this program displays two video streams. "Motion Frame" places a red box around a pixel difference between frames, while
#the "Faces Frame" places a green box around things it recognizes as faces. Press 'p' to take a photo after entering 'y'.
#The video stream will then quit and show the saved photo. Enter 'y' again when prompted to resume video stream and photo taking.

#import necessary libraries
import cv2 #make sure to have opencv-python package installed (If not this can be done via "pip install opencv-python")
import pandas #make sure to have the panda package installed (IF not this can be done via "pip install pandas")
import sys
import os.path
import logging as log
import datetime as dt
from time import sleep
from os import path

#Configure webcam
first_frame = None
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

photoNum=1 #will hold the photo number so that program creates a unique picture file

#allows user to set what camera they have connected that they wish to use
camNum = "0"
camNum = input("Enter the camera index for camera that will take photos (0 is builtin camera usually): ") 
while(camNum.isdigit() == False):
      camNum=input("Enter a digit for camera index: ")

#Initial prompt to user to enter video stream and instructions on taking photo
photoYorN = input("Do you want to take a photo? (enter y for yes or n for no):")
print("Enter p to take a photo, s to switch camera index, or q to quit: ")

while(photoYorN == 'y'):
    #create live stream video feed object

    video_capture = cv2.VideoCapture(int(camNum))
    anterior = 0

    while (True):
        #check to see if camera is succesfully opened by program for photos
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        #color frame for motion detection
        check,color_frame = video_capture.read()
        status = 0
        gray = cv2.cvtColor(color_frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)

        if first_frame is None:
            first_frame=gray
            continue

        #frame which holds the difference between the two frames 
        delta_frame = cv2.absdiff(first_frame,gray)
        thresh_frame = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

        thresh_frame = cv2.dilate(thresh_frame,None,iterations=3)
        (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #create rectangle which higlights pixel differences between frames due to motion
        for contour in cnts:
            if cv2.contourArea(contour)<10000:
                continue
            status=1
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(color_frame,(x,y),(x+w,y+h),(0,0,255),2)

        #display motion frame
        cv2.imshow("Motion Frame", color_frame)


        # Capture frame-by-frame for face
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


        # Display the resulting frame
        cv2.imshow("Faces Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('p'): 
        #take photo, make sure to have "Faces Frame" stream selected
            while(path.isfile("mission_img" + str(photoNum) + ".jpg")):
                photoNum += 1
            check, frame = video_capture.read()
            cv2.imshow("Capturing", frame)
            cv2.imwrite(filename="mission_img" + str(photoNum) + ".jpg", img=frame)
            video_capture.release()
            img_new = cv2.imread("mission_img" + str(photoNum) + ".jpg", cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            print("Image Saved")
            cv2.destroyAllWindows()
            photoNum += 1
            break

        elif cv2.waitKey(1) & 0xFF == ord('q'):
        #quit video video stream
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
        #quit video after getting new camera index
            print("Turning off currentcamera to switch to another.")
            video_capture.release()
            print("Current camera off.")
            cv2.destroyAllWindows()
            camNum = input("Enter the camera index for camera that will take photos (0 is builtin camera usually): ")
            while(camNum.isdigit() == False):
                camNum = input("Enter a digit for camera index: ")
            break

    #reprompt user
    photoYorN = input("Do you want to take a photo? (enter y for yes or n for no):")
    print("Enter p to take a photo or q to quit: ")

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

