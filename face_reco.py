from deepface import DeepFace
import cv2
from keras_facenet import FaceNet
from numpy import asarray, expand_dims
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
import pickle
from tkinter import *
import csv
import shutil

MyFaceNet = FaceNet()
database_file = "data.pkl"
database = {}

def load_database():
    global database
    if os.path.exists(database_file):
        with open(database_file, "rb") as myfile:
            database = pickle.load(myfile)
            if not database:
                trainer('dataset1/')
    else:
        trainer('dataset1/')
    print("Database loaded from file.")

def trainer(folder):
    for filename in os.listdir(f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}'):
        for file in os.listdir(f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}//{filename}'):
            img_path = f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}//{filename}//{file}'
            img = cv2.imread(img_path)

            # Use DeepFace RetinaFace for detection
            faces = DeepFace.extract_faces(img, detector_backend='retinaface', enforce_detection=False)

            if faces:
                for face in faces:
                    facial_area = face['facial_area']
                    print(face['facial_area'])
                    x = int(facial_area['x'])
                    y = int(facial_area['y'])
                    w = int(facial_area['w'])
                    h = int(facial_area['h'])

                    # Crop the face from the image
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (160, 160))

                    face_img = asarray(face_img)
                    face_img = expand_dims(face_img, axis=0)

                    signature = MyFaceNet.embeddings(face_img)
                    database[os.path.splitext(filename)[0]] = signature

    save_database()

def save_database():
    with open(database_file, "wb") as myfile:
        pickle.dump(database, myfile)

load_database()
cap=cv2.VideoCapture(0)
s = set()

def video_capture():
    back = cv2.imread('C:\\Users\\ASUS\\Desktop\\jvw\\Group 1.png')
    

    while True:
        vid = cv2.imread('C:\\Users\\ASUS\\Desktop\\jvw\\WhatsApp Image 2024-08-18 at 22.44.00_c51ec9e2.jpg')
        # j,vid=cap.read()
        vid = cv2.resize(vid, (800, 500))

        # Use DeepFace RetinaFace for detection
        faces = DeepFace.extract_faces(vid, detector_backend='retinaface', enforce_detection=False)

        if faces:
            for face in faces:
                facial_area = face['facial_area']
                print(face['facial_area'])
                x = int(facial_area['x'])
                y = int(facial_area['y'])
                w = int(facial_area['w'])
                h = int(facial_area['h'])

                # Crop the face from the image
                face_img = vid[y:y+h, x:x+w]
                if face_img.size == 0:
                        print("Empty face image detected")
                        continue
                face_img = cv2.resize(face_img, (160, 160))

                face_img = asarray(face_img)
                face_img = expand_dims(face_img, axis=0)

                signature = MyFaceNet.embeddings(face_img)

                max_similarity = -1
                min_euclidean_distance = float('inf')
                identity = 'Unknown'

                for key, value in database.items():
                    similarity = cosine_similarity(value, signature)[0][0]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        identity = key

                    euclidean_distance = euclidean_distances(value, signature)[0][0]
                    if euclidean_distance < min_euclidean_distance:
                        min_euclidean_distance = euclidean_distance
                        identity = key

                if max_similarity > 0.8 and min_euclidean_distance < 0.8:
                    print(f'name  {identity}')
                    print(f'Cosine similarity  {max_similarity}')
                    print(f'Euclidean distance  {min_euclidean_distance}')
                    cv2.putText(vid, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(vid, (x, y), (w+x, h+y), (255, 0, 0), 2)
                    s.add(identity)
                else:
                    cv2.putText(vid, '.', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(vid, (x, y), (x+w,y+h), (0, 255, 0), 2)

        back[174:174 + 500, 121:121 + 800] = vid
        cv2.imshow('Video Feed', back)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    show_attendance()

root = Tk()
selected_names = {}

def close():
    root.destroy()

def save():
    with open('attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name"])
        for name, var in selected_names.items():
            if var.get() == 1:
                writer.writerow([name])
    print("Attendance saved.")

def show_attendance():
    root.geometry('400x500')
    root.title("Recognized Students")

    for i in s:
        var = IntVar()
        selected_names[i] = var
        checkbox = Checkbutton(root, text=i, variable=var, font=('Arial', 15, 'bold'), padx=15, pady=15)
        checkbox.pack(pady=2)

    save_button = Button(root, text="Save", command=save)
    save_button.pack(pady=10)

    close_button = Button(root, text="Close", command=close)
    close_button.pack(pady=10)

    root.mainloop()

video_capture()
