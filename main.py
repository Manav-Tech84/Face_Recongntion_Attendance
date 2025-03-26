import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


video_cap = cv2.VideoCapture(0)

# Load Know faces

hari_image = face_recognition.load_image_file("faces/hari.jpg")
hari_encoding = face_recognition.face_encodings(hari_image)[0]

johnny_image = face_recognition.load_image_file("faces/johnny.jpg")
johnny_encoding = face_recognition.face_encodings(johnny_image)[0]

shiva_02_image = face_recognition.load_image_file("faces/shiva_02.jpg")
shiva_02_encoding = face_recognition.face_encodings(shiva_02_image)[0]

rajpal_image = face_recognition.load_image_file("faces/rajpal.jpg")
rajpal_encoding = face_recognition.face_encodings(rajpal_image)[0]

manav_image = face_recognition.load_image_file("faces/manav.jpg")
manav_encoding = face_recognition.face_encodings(manav_image)[0]

know_faces_encodings = [hari_encoding,johnny_encoding,shiva_02_encoding,rajpal_encoding,manav_encoding]
know_faces_names = ["hari","johnny","shiva_02","rajpal","manav"]

# make a list of expected students
Students = know_faces_names.copy()
face_locations = []
face_encodings = []

# get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv","w+",newline="")
lwrite = csv.writer(f)

while True:
        _, frame = video_cap.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

        # recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_faces_encodings, face_encoding)
            face_distance = face_recognition.face_distance(know_faces_encodings, face_encoding)
            best_matches_index = np.argmin(face_distance)

            if matches[best_matches_index]:
                name = know_faces_names[best_matches_index]

                # add a text if a person is present
                if name in know_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 100)
                    fontScale = 1.5
                    fontColor = (255, 0, 0)
                    thickness = 3
                    lineType = 2
                    cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in Students:
                    Students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lwrite.writerow([name, current_time])

        cv2.imshow("Attend",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_cap.release()
cv2.destroyAllWindows()





