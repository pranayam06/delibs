import cv2
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

point_queue = deque(maxlen=10)
last_gesture_time = 0
gesture_cooldown = 2  

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tips = [4, 8, 12, 16, 20]  # Thumb to Pinky
            extended = []
            
            for tip in tips[1:]:  # Skip thumb
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                    extended.append(True)
                else:
                    extended.append(False)
            
            extended_count = sum(extended)
            
            if (current_time - last_gesture_time) > gesture_cooldown:
                if extended_count == 1:  # Add point
                    point_queue.append(f"Point {len(point_queue)+1}")
                    last_gesture_time = current_time
                    
                elif extended_count == 2:  # Second point
                    if point_queue:
                        point_queue.append(f"Seconded: {point_queue[-1]}")
                        last_gesture_time = current_time
                        
                elif extended_count == 3:  # Remove last (pop)
                    if point_queue:
                        removed = point_queue.pop()
                        print(f"Removed: {removed}")
                        last_gesture_time = current_time
                        
                elif extended_count == 4:  # Dequeue first
                    if point_queue:
                        processed = point_queue.popleft()
                        print(f"Processed: {processed}")
                        last_gesture_time = current_time
    
    cv2.putText(frame, f"Queue: {len(point_queue)} items", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for i, item in enumerate(list(point_queue)[-3:]):
        cv2.putText(frame, item, (10, 70 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "1:Add 2:Second 3:Remove 4:Dequeue", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.imshow('Gesture Queue', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()