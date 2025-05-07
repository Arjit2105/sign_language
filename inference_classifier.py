import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start video capture
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels
labels_dict = {0: "hello", 1: "no", 2: "yes"}

# Sentence construction
sentence = ""
last_prediction = None
cooldown_counter = 0
cooldown_limit = 20

while True:
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            current_prediction = labels_dict[int(prediction[0])]

            if current_prediction != last_prediction and cooldown_counter == 0:
                sentence += current_prediction + " "
                last_prediction = current_prediction
                cooldown_counter = cooldown_limit

    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Display sentence
    cv2.putText(frame, sentence.strip(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Show instructions
    cv2.putText(frame, 'Press "C" to clear sentence, "Q" to quit', (20, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Sign Language Translator', frame)
    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        last_prediction = None  # reset last prediction

cap.release()
cv2.destroyAllWindows()
