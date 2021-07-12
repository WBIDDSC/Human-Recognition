import cv2
import pandas
import sys
import logging as log
import datetime as dt
from time import sleep

first_frame=None

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

photoYorN = input("Do you want to take a photo? (enter y for yes or n for no):")
print("Enter p to take a photo or q to quit: ")
photoNum = 1

while(photoYorN == 'y'):
    video_capture = cv2.VideoCapture(0)
    anterior = 0

    while (True):
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        #color frame for motion detection
        check,color_frame=video_capture.read()
        status=0
        gray=cv2.cvtColor(color_frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(21,21),0)

        if first_frame is None:
            first_frame=gray
            continue

        delta_frame=cv2.absdiff(first_frame,gray)
        thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

        thresh_frame=cv2.dilate(thresh_frame,None,iterations=3)
        (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour)<10000:
                continue
            status=1
            (x,y,w,h)=cv2.boundingRect(contour)
            cv2.rectangle(color_frame,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.imshow("Motion Frame", color_frame)


        # Capture frame-by-frame
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
        
            check, frame = video_capture.read()
            cv2.imshow("Capturing", frame)
            cv2.imwrite(filename="mission_img" + str(photoNum) + ".jpg", img=frame)
            video_capture.release()
            img_new = cv2.imread("mission_img" + str(photoNum) + ".jpg", cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            print("Image Saved")
            print("Program End")
            cv2.destroyAllWindows()
            photoNum += 1
            break

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    #reprompt user
    photoYorN = input("Do you want to take a photo? (enter y for yes or n for no):")
    print("Enter p to take a photo or q to quit: ")

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

