import cv2
import time
from ultralytics import YOLO

#Egg Detector
model = YOLO("runs/detect/train-4/weights/best.pt")

#Camera (If USB cam change 0 to 1 or 2)
cap = cv2.VideoCapture(0)

print("Starting to capture, press q to quit")

prev_time = 0

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        print("Failed to grab camera frame")
        break


    #Runs detector on the given frame
    results = model.predict(source = image, conf = 0.65, iou = 0.45, verbose = False)

    #Bounding Boxes
    annotated_image = results[0].plot()

    #Gets current time
    curr_time = time.time()

    if prev_time > 0:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0

    cv2.putText(
        annotated_image,
        f"FPS: {fps: .1f}",
        (15, 35),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Egg Detection - Live  Feed", annotated_image)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.releaste()
cv2.destroyAllWindows()