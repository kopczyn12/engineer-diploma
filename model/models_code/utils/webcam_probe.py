import cv2
import tensorflow as tf
import numpy as np
import time

model = tf.keras.models.load_model('/model/trained_models/fer_model.h5')

# size for face
face_width = 48
face_height = 48

def classify_emotion(face):
    # change the size for our model
    face = cv2.resize(face, (face_width, face_height))
    face = np.expand_dims(face, axis=0)
    # make prediction, check accuracy
    acc = np.amax(model.predict(face, verbose=0))
    pred = np.argmax(model.predict(face, verbose=0))
    return pred, acc

# webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # for each found face
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # classify emotion and make time check
        start_time = time.time()
        emotion, accuracy = classify_emotion(face)
        end_time = time.time()
        processing_time = end_time - start_time

        # bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # add emotion label and convertion time per frame
        emotion_labels = ['angry', 'disgust', 'happy', 'sad', 'surprise']
        if accuracy > 0.95:
            label_text = f"{emotion_labels[emotion]} ({processing_time:.2f}s) {str(100*accuracy)[:4]}%"
        else:
            label_text = "no emotion detected"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # show frame with emotion label
    cv2.imshow('Emotion Detection', frame)

    # break after presing "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# close window
cap.release()
cv2.destroyAllWindows()
