import cv2
import mediapipe as mp
import pygame
import time

# Initialize Pygame Mixer for music
pygame.mixer.init()

# Load a song (use your local file path)
pygame.mixer.music.load("Perfect(Mr-Jatt1.com).mp3")
pygame.mixer.music.play(-1)  # Play in a loop

# Hand detection using Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Helper function to detect heart shape gesture
def detect_heart_shape(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_1 = landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_tip_2 = landmarks2.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip_2 = landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    dist_thumb = ((thumb_tip_1.x - thumb_tip_2.x) ** 2 + (thumb_tip_1.y - thumb_tip_2.y) ** 2) ** 0.5
    dist_index = ((index_tip_1.x - index_tip_2.x) ** 2 + (index_tip_1.y - index_tip_2.y) ** 2) ** 0.5
    
    if dist_thumb < 0.05 and dist_index < 0.05:
        return True
    return False

# Helper function to detect "X" gesture for removing favorites
def detect_x_shape(landmarks1, landmarks2):
    wrist1 = landmarks1.landmark[mp_hands.HandLandmark.WRIST]
    wrist2 = landmarks2.landmark[mp_hands.HandLandmark.WRIST]
    index1 = landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index2 = landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Check if wrists are crossing and indices are crossing
    if wrist1.x > wrist2.x and abs(wrist1.y - wrist2.y) < 0.1 and abs(index1.x - index2.x) < 0.1:
        return True
    return False

# Helper function to detect gestures (thumbs up/down and index finger direction)
def detect_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Thumbs up gesture
    if thumb_tip.y < wrist.y and abs(thumb_tip.x - index_tip.x) > 0.1:
        return "thumb_up"
    
    # Thumbs down gesture
    if thumb_tip.y > wrist.y and abs(thumb_tip.x - index_tip.x) > 0.1:
        return "thumb_down"
    
    # Index finger pointing up
    if index_tip.y < wrist.y and abs(index_tip.x - wrist.x) < 0.1:
        return "volume_up"
    
    # Index finger pointing down
    if index_tip.y > wrist.y and abs(index_tip.x - wrist.x) < 0.1:
        return "volume_down"
    
    return None

# Helper function to detect "V" gesture
def detect_v_gesture(landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Detect "V" gesture based on the position of index and middle fingers
    if index_tip.y < wrist.y and middle_tip.y < wrist.y and abs(index_tip.x - middle_tip.x) > 0.1:
        return True
    return False

# Webcam capture
cap = cv2.VideoCapture(0)

volume = 0.5
pygame.mixer.music.set_volume(volume)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the image
    if not ret:
        break

    # Convert image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(rgb_frame)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 1:
            # Single hand gestures
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            print("Single hand detected")

            # Gesture detection logic
            gesture = detect_gesture(hand_landmarks)

            if gesture == "thumb_up":
                print("Thumb up detected - Resuming music")
                pygame.mixer.music.unpause()
            elif gesture == "thumb_down":
                print("Thumb down detected - Pausing music")
                pygame.mixer.music.pause()
            elif gesture == "volume_up":
                volume = min(volume + 0.1, 1.0)
                pygame.mixer.music.set_volume(volume)
                print(f"Volume increased to {volume}")
            elif gesture == "volume_down":
                volume = max(volume - 0.1, 0.0)
                pygame.mixer.music.set_volume(volume)
                print(f"Volume decreased to {volume}")
            
            # "V" gesture to stop the music
            if detect_v_gesture(hand_landmarks):
                print("V gesture detected - Stopping music and exiting application")
                pygame.mixer.music.stop()
                cap.release()
                cv2.destroyAllWindows()
                break

        elif len(result.multi_hand_landmarks) == 2:
            # Multi-hand gestures
            hand_landmarks1 = result.multi_hand_landmarks[0]
            hand_landmarks2 = result.multi_hand_landmarks[1]
            mp_drawing.draw_landmarks(frame, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, hand_landmarks2, mp_hands.HAND_CONNECTIONS)

            print("Two hands detected")

            if detect_heart_shape(hand_landmarks1, hand_landmarks2):
                print("Heart shape detected - Added to Favorites!")
            if detect_x_shape(hand_landmarks1, hand_landmarks2):
                print("X shape detected - Removed from Favorites!")

    # Display the frame with hand landmarks
    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


