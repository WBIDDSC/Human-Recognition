import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
josh_image = face_recognition.load_image_file("Josh.jpg")
josh_face_encoding = face_recognition.face_encodings(josh_image)[0]

# Load a sample picture and learn how to recognize it.
willow_image = face_recognition.load_image_file("Willow Bragg.jpg")
willow_face_encoding = face_recognition.face_encodings(willow_image)[0]

# Load a sample picture and learn how to recognize it.
hunter_image = face_recognition.load_image_file("Hunter Bendele.jpg")
hunter_face_encoding = face_recognition.face_encodings(hunter_image)[0]

# Load a sample picture and learn how to recognize it.
dalton_image = face_recognition.load_image_file("Dalton.jpg")
dalton_face_encoding = face_recognition.face_encodings(dalton_image)[0]

# Load a sample picture and learn how to recognize it.
chris_image = face_recognition.load_image_file("Chris.jpg")
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

# Load a sample picture and learn how to recognize it.
jason_image = face_recognition.load_image_file("Jason.jpg")
jason_face_encoding = face_recognition.face_encodings(jason_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    josh_face_encoding,
    willow_face_encoding,
    hunter_face_encoding,
    dalton_face_encoding,
    chris_face_encoding,
    jason_face_encoding
]
known_face_names = [
    "Josh",
    "Willow",
    "Hunter",
    "Dalton",
    "Chris",
    "Jason"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, Rframe = video.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(Rframe, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(Rframe, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(Rframe, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(Rframe, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Recognized', Rframe)

    photoNum = 1
    if cv2.waitKey(1) & 0xFF == ord('p'): 
        #take photo, make sure to have "Faces Frame" stream selected with mouse
            #save photo file and display it. If a photo exists with that photoNum increment until a new unique file can be made
            #Also quit video stream to reset it
        while(path.isfile("mission_img" + str(photoNum) + ".jpg")):
            photoNum += 1
        check, frame = video_capture.read()
        cv2.imwrite(filename="mission_img" + str(photoNum) + ".jpg", img=frame)
        img_new = cv2.imread("mission_img" + str(photoNum) + ".jpg", cv2.IMREAD_GRAYSCALE)
        img_new = cv2.imshow("mission_img" + str(photoNum) + ".jpg", img_new)
        cv2.waitKey(1000)
        print("Image Saved")
        cv2.destroyWindow("mission_img" + str(photoNum) + ".jpg")
        photoNum += 1

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()
