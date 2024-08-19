import os
import cv2
import pickle
import numpy as np
from os import listdir
from numpy import expand_dims
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tkinter import *
import csv
from PIL import ImageTk, Image
import shutil

# Load DNN face detector (SSD with ResNet backbone)
face_detector = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 
    'res10_300x300_ssd_iter_140000.caffemodel'
)

# Load pre-trained DNN model for face embedding (OpenFace or any other)
face_recognizer = cv2.dnn.readNetFromTorch('C:\\Users\\ASUS\\Desktop\\jvw\\openface.nn4.small2.v1.t7')

# Dataset path
folder = 'dataset1/'
database_file = "data.pkl"
database = {}

# Load or initialize the database
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

# Save the database
def save_database():
    with open(database_file, "wb") as myfile:
        pickle.dump(database, myfile)

# Train the model on a ne
def trainer(folder):
    for filename in listdir(f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}'):
        for file in listdir(f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}//{filename}'):
            image_path = f'C:\\Users\\ASUS\\Desktop\\jvw\\{folder}//{filename}//{file}'
            image = cv2.imread(image_path)
            faces = detect_faces_dnn(image)

            for face_data in faces:
                (x1, y1, x2, y2) = face_data['box']
                face = image[y1:y2, x1:x2]
                face_embedding = get_face_embedding(face)
                database[os.path.splitext(filename)[0]] = face_embedding

    if folder == 'others/':
        print("Model trained on the ne")
        destination = 'C:\\Users\\ASUS\\Desktop\\jvw\\dataset1\\'
        main = 'C:\\Users\\ASUS\\Desktop\\jvw\\others\\'
        for folders in os.listdir(main):
            shutil.move(f'{main}//{folders}', f'{destination}//{folders}')
            print("Moved :", folders)
    else:
        print("Model trained on the bi")

    save_database()

# Detect faces using DNN
def detect_faces_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append({'box': (x1, y1, x2, y2)})

    return faces

# Get face embedding using DNN
def get_face_embedding(face):
    if face is None or face.size == 0:
        raise ValueError("The face image is empty or not loaded correctly.")
    
    # Proceed with image processing
    face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    face_recognizer.setInput(face_blob)
    embedding = face_recognizer.forward()
    return embedding
    

load_database()

cap = cv2.VideoCapture(0)
s = set()

def video_capture():
    back = cv2.imread('C:\\Users\\ASUS\\Desktop\\jvw\\Group 1.png')
    cv2.namedWindow('Video Feed')

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (800, 500))
        faces = detect_faces_dnn(frame)
        if frame is None:
            print("Failed to capture image from camera.")
            break

        for face_data in faces:
            (x1, y1, x2, y2) = face_data['box']
            face = frame[y1:y2, x1:x2]
            face_embedding = get_face_embedding(face)

            max_similarity = -1  
            min_euclidean_distance = float('inf')
            identity = 'Unknown'

            for key, value in database.items():
                similarity = cosine_similarity(value, face_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    identity = key

                euclidean_distance = euclidean_distances(value, face_embedding)[0][0]
                if euclidean_distance < min_euclidean_distance:
                    min_euclidean_distance = euclidean_distance
                    identity = key

            if max_similarity > 0.65 and min_euclidean_distance < 0.75:
                print(f'Name: {identity}')
                print(f'Cosine Similarity: {max_similarity}')
                print(f'Euclidean Distance: {min_euclidean_distance}') 
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                s.add(identity)
            else:
                cv2.putText(frame, '.', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        back[174:174 + 500, 121:121 + 800] = frame
        cv2.imshow('Video Feed', back)
        if cv2.waitKey(20) & 0xFF == ord('q'):
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
    root.title("Recognised Students")

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
