import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import time

# Load the trained model
model = load_model("./model2-010.keras")

# Initialize mixer and load alarm sound
mixer.init()
try:
    mixer.music.load("alarm.mp3")
except:
    print("Alarm sound file not found. Ensure 'alarm.mp3' is available.")

# Define label map and drawing colors
results = {0: 'mask', 1: 'without mask'}
GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Haarcascade path
haarcascade_path = 'C:\\Users\\vikas\\OneDrive\\Desktop\\elevate AI ML\\haarcascade_frontalface_default.xml'
haarcascade = cv2.CascadeClassifier(haarcascade_path)

if haarcascade.empty():
    raise IOError("Failed to load Haarcascade XML.")

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam.")

# Control alarm frequency
last_alarm_time = 0
cooldown = 5  # seconds

rect_size = 4

while True:
    rval, im = cap.read()
    if not rval:
        print("Failed to grab frame.")
        break

    im = cv2.flip(im, 1)
    resized_frame = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(resized_frame)

    mask_detected = True  # Assume all are wearing masks initially

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y+h, x:x+w]
        try:
            resized_face = cv2.resize(face_img, (150, 150))
        except:
            continue

        normalized = resized_face / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))

        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result, axis=1)[0]

        if label == 1:
            mask_detected = False  # At least one person without mask

        # Draw bounding box and label
        cv2.rectangle(im, (x, y), (x+w, y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y-40), (x+w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Trigger alarm only if at least one person is without mask
    current_time = time.time()
    if not mask_detected and (current_time - last_alarm_time > cooldown):
        if not mixer.music.get_busy():  # Play only if not already playing
            mixer.music.play()
        last_alarm_time = current_time

    cv2.imshow("LIVE", im)
    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
