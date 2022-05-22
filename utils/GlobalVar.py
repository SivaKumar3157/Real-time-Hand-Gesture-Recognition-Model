import mediapipe as mp


DEVICE_ID = 0
WINDOW = "Gesture to letter"


classes = ['А', 'D', 'S', 'L', 'F', 'O', '-', 'I', 'V', 'K', 'N', 'M',
           'Н', 'W', 'Q', 'R', 'C', '-', 'Y', 'T', 'G', 'P', 'Z', 'E',
           'U', 'B', 'J', 'H', 'X']



KEY_POINTS_CLASSIFIER_PATH = f'models/lr_{len(classes)}_aug.sav'

#create objects of mediapipe for hand detection and hand pose estimation 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
