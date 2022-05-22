import cv2
import pickle
import numpy as np
from scipy.stats import mode
from utils.Distance import *
from utils.GlobalVar import *

#create opencv object to capture webcap feed 
cap = cv2.VideoCapture(DEVICE_ID)
cv2.namedWindow(WINDOW)

#load model file
clf = pickle.load(open(KEY_POINTS_CLASSIFIER_PATH, 'rb'))

frame_count = 0  
predictions = []  
prev_predictions = []  
similar_words = None  
isRecording = False
prev_length = 0  

#display predictions 
def show_predict(predict: np.ndarray, frame: np.ndarray, corner_coo: tuple, predicted_str: str):

    cv2.rectangle(frame, corner_coo, (corner_coo[0] + 70, corner_coo[1] - 70),
                  (255, 255, 255) if not isRecording else (0, 0, 200), thickness=-1)
    cv2.rectangle(frame, (frame.shape[1] - 90, frame.shape[0] - 25), (frame.shape[1], frame.shape[0]),
                  (255, 255, 255), thickness=-1)
    cv2.putText(frame, f'{np.max(predict):.3f}', (frame.shape[1] - 90, frame.shape[0] - 3),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'{classes[np.argmax(predict)]}' if np.max(predict) > 0.99 else '-',
                (corner_coo[0] + 18, corner_coo[1] - 18), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)

    if isRecording:
        cv2.circle(frame, (int(frame.shape[1] - 25), 15), 8, (0, 0, 255), -1)
        cv2.rectangle(frame, (0, frame.shape[0]), (frame.shape[1] - 150, frame.shape[0] - 25),
                      (255, 255, 255), thickness=-1)
        cv2.putText(frame, f' >{predicted_str}', (0, frame.shape[0] - 3),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)



def draw_bb(points: np.ndarray):

    global prev_length

    x_max, x_min = int(np.max(points[0][::2]) * image.shape[1]), int(np.min(points[0][::2]) * image.shape[1])
    y_max, y_min = int(np.max(points[0][1::2]) * image.shape[0]), int(np.min(points[0][1::2]) * image.shape[0])
    right_up_corner = (x_max, y_min)
    line_length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    diff = np.abs(line_length - prev_length)
    prev_length = line_length
    percent_diff = diff * 100 / line_length

    cv2.line(image, (x_max, y_min), (x_min, y_min), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_min, y_min), (x_max, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_max, y_min), (x_max, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_max, y_max), (x_min, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.line(image, (x_min, y_min), (x_min, y_max), (200, 10, 10), 2, cv2.LINE_AA)
    cv2.circle(image, (x_max, y_max), 2, (10, 10, 220), 3)
    cv2.circle(image, (x_min, y_min), 2, (10, 10, 220), 3)

    return percent_diff, right_up_corner


while cap.isOpened():
	#read images from webcam
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    #find hand in the image
    results = hand.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        frame_count += 1
		#get the 21 lanmarks of the hand if detected
        hand_landmarks = results.multi_hand_landmarks[0]
        #draw the landmarks on the image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        points_xyz = []
        for mark in hand_landmarks.landmark:
            points_xyz.extend([mark.x, mark.y])

        points_xyz = np.array(points_xyz).reshape(1, -1)
        difference, right_up_corner = draw_bb(points_xyz)
		#make predictions on the vector using the classifier 
        pred = clf.predict_proba(points_xyz)[0]
        show_predict(pred, image, right_up_corner, ''.join(predictions))

        if isRecording:
            if np.max(pred) > 0.985 and difference < 0.35:
                if len(prev_predictions) > 5:
                    letter = mode(prev_predictions)[0][0]
                    letter_count = mode(prev_predictions)[1][0]
                    if (len(predictions) == 0 or predictions[-1] != letter) and letter_count / 5 > 0.6:
                        predictions.append(letter)
                    prev_predictions = []
                else:
                    prev_predictions.append(classes[np.argmax(pred)])

    cv2.imshow(WINDOW, image)
    key = cv2.waitKey(5)

    if key & 0xFF == 27 or key == ord('q'):
        break

#record letters to create words
    if key == ord('s'):
        if isRecording:
            isRecording = False
            predictions = ''.join(predictions)
            #find similar words to the typed word using distance 
            similar_words = min_distance(predictions)
            print(f'Input: {predictions}, Predict: {similar_words}')
        else:
            predictions = []
            isRecording = True
            similar_words = None



hand.close()
cap.release()
