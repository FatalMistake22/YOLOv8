

from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0) # 0 for webcam
cap.set(3, 640) #width
cap.set(4, 480) #height


model = YOLO("yolov8x.pt") #weights from pretrained model


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read() #success=> boolean checks if it read the frames correctly, img contains the image data for each frame
    results = model(img, stream=True)

   
    for r in results: #loop on each result in one frame

        boxes = r.boxes #extract bounding boxes of object detected
        for box in boxes: #loop on Bounding boxes of object detected
            x1, y1, x2, y2 = box.xyxy[0] #top left and bottom right
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values for more accurate results

            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) #draw the bounding box on the frame

            
            confidence = box.conf[0]  #box.conf is an array that stores all confidences  
            print("Confidence --->",confidence)

            cls = int(box.cls[0]) #box.cls is an array that stores all class indeces, and then convert it to integer
            print("Class name -->", classNames[cls]) #then maps the integer to the classname array 

            #To draw the text(name of class) on top of the bounding box
            org = [x1, y1] 
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img) #window title with processed image
    if cv2.waitKey(1) == ord('q'): #waits for a key event with 1 ms delay
        break

cap.release() #brelease webcam resources
cv2.destroyAllWindows() #close all opencv windows
