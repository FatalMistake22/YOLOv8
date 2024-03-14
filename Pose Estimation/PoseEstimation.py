#Yasmin Tarek, Nadeen El-Sayed, Rama Selim, Zahra Hassan

import cv2
from ultralytics import YOLO
import numpy as np
import cvzone  


model = YOLO("yolov8x-pose-p6.pt") 

cap = cv2.VideoCapture('cc.mp4') 


def calculate_angle(a, b, c):  #keep the keypoints(vertices of person) in an array
    a = np.array(a)  # First 
    b = np.array(b)  # Mid 
    c = np.array(c)  # End 

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]) #to get the angle between BA BC, then using artcan to get the angle between them
    angle = np.abs(radians * 180.0 / np.pi) #convert from radians to degrees

    if angle > 180.0: 
        angle = 360 - angle #to make the range of the angle [0,180]

    return angle

while True:  
    ret, frame = cap.read()
    
    try: 
        frame = cv2.resize(frame, (1200,600)) 
    except: 
        pass

    if not ret:
        break

    results = model.predict(frame, save=True) #to save results of bounding box and save=True to make results accessible later

    
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) #xyxy:top left w el bottom right,The boxes are saved as tensor(multidimensional array) on gpu so to convert it to numpy we have to send it to cpu

    statuses = [] #empty list to save the labels  (sitting,standing)

    keypoints_data = results[0].keypoints.data 

    for i, keypoints in enumerate(keypoints_data): 
        if keypoints.shape[0] > 0: 
            angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]) 
            print(f"Person {i + 1} is {'Sitting' if angle is not None and angle < 110 else 'Standing'} (Angle: {angle:.2f} degrees)") #.2f to approximate it to 2 decimal points
            statuses.append('Sitting' if angle is not None and angle < 110 else 'Standing') 

    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cvzone.putTextRect(
            frame, f"{statuses[i]}", (x1, y2 - 10),  
            scale=3, thickness=3,  
            colorT=(255, 255, 255), colorR=(255, 0, 255), 
            font=cv2.FONT_HERSHEY_PLAIN,
            offset=10, 
            border=0, colorB=(0, 255, 0) 
        )



    detection = results[0].plot()

    cv2.imshow('YOLOv8 Pose Detection', detection)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
