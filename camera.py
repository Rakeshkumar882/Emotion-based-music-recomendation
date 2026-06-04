import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from threading import Thread, Lock
from collections import Counter
import pandas as pd

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation="softmax"))
emotion_model.load_weights("model.h5")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist = {
    0:"songs/angry.csv", 1:"songs/disgusted.csv", 2:"songs/fearful.csv",
    3:"songs/happy.csv", 4:"songs/neutral.csv",   5:"songs/sad.csv",
    6:"songs/surprised.csv"
}

show_text = [0]
prediction_buffer = []
BUFFER_SIZE = 3
CONFIDENCE_THRESHOLD = 0.40
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# CSV cache
csv_cache = {}
def load_csv_cached(idx):
    if idx not in csv_cache:
        csv_cache[idx] = pd.read_csv(music_dist[idx], encoding="latin-1")[["Name","Album","Artist"]].head(100)
    return csv_cache[idx]

def preprocess_face(roi_gray):
    roi = cv2.equalizeHist(roi_gray)
    roi = clahe.apply(roi)
    roi = cv2.resize(roi, (48,48))
    roi = roi.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(roi, -1), 0)

def smooth_prediction(new_index):
    prediction_buffer.append(new_index)
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)
    return Counter(prediction_buffer).most_common(1)[0][0]

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class VideoCamera(object):
    def __init__(self):
        self.stream     = WebcamVideoStream(src=0).start()
        self.frame_count = 0
        self.lock        = Lock()

        # last known face box and label - updated by bg thread
        self.last_face   = None   # (x,y,w,h)
        self.last_label  = ""
        self.pending_roi = None   # roi waiting for prediction
        self.predicting  = False

        # start background prediction thread
        Thread(target=self._predict_loop, daemon=True).start()

    def _predict_loop(self):
        """Runs emotion prediction in background - never blocks camera stream"""
        while True:
            with self.lock:
                roi = self.pending_roi
                self.pending_roi = None

            if roi is None:
                cv2.waitKey(1)
                continue

            prediction = emotion_model.predict(roi, verbose=0)
            confidence  = float(np.max(prediction))
            raw_index   = int(np.argmax(prediction))

            if confidence >= CONFIDENCE_THRESHOLD:
                smoothed = smooth_prediction(raw_index)
                show_text[0] = smoothed
                conf_pct = int(confidence * 100)
                with self.lock:
                    self.last_label = f"{emotion_dict[smoothed]} ({conf_pct}%)"

    def get_frame(self):
        self.frame_count += 1
        img  = self.stream.read()
        img  = cv2.resize(img, (600, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect face every 2 frames
        if self.frame_count % 2 == 0:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                with self.lock:
                    self.last_face = faces[0]   # track biggest face

        # draw last known face box every frame (smooth visuals)
        with self.lock:
            face  = self.last_face
            label = self.last_label

        if face is not None:
            x, y, w, h = face
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0,255,0), 2)
            cv2.putText(img, label, (x+5, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (203,192,255), 2, cv2.LINE_AA)

            # queue roi for prediction every 2 frames
            if self.frame_count % 2 == 0:
                roi_gray  = gray[y:y+h, x:x+w]
                processed = preprocess_face(roi_gray)
                with self.lock:
                    self.pending_roi = processed

        ret, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes(), load_csv_cached(show_text[0])

def music_rec():
    return load_csv_cached(show_text[0])
