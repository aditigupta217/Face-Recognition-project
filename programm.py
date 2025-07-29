import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)


aditi_image = face_recognition.load_image_file("/Users/aditi/Desktop/face_recog/WhatsApp Image 2025-07-28 at 6.12.48 PM.jpeg")
aditi_encoding = face_recognition.face_encodings(aditi_image)[0]

varad_image = face_recognition.load_image_file("/Users/aditi/Desktop/face_recog/WhatsApp Image 2025-07-28 at 6.15.02 PM-2.jpeg")
varad_encoding = face_recognition.face_encodings(varad_image)[0]

sharvari_image = face_recognition.load_image_file("/Users/aditi/Desktop/face_recog/WhatsApp Image 2025-07-28 at 6.15.02 PM.jpeg")
sharvari_encoding = face_recognition.face_encodings(sharvari_image)[0]

known_face_encoding = [aditi_encoding, varad_encoding, sharvari_encoding]
known_faces_name = ["Aditi", "Varad", "Sharvari"]
student = known_faces_name.copy()

current_date = datetime.now().strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w', newline='')
lnwriter = csv.writer(f)


while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    cv2.imshow("Face Recognition Attendance", frame)

    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_endco in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_endco)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_endco)

        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_faces_name[best_match_index]

        face_names.append(name)

        if name in known_faces_name:
            if name in student:
                student.remove(name)
                print(student)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
