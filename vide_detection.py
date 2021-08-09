import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

#cap = cv2.VideoCapture('pettah.mp4')
img = cv2.imread("img.jpeg")

# _, img = cap.read()
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

class_ids = []
confidences = []
boxes = []

for output in layerOutputs:
    for detection in output:  # each detection have 85 parameters
        scores = detection[5:]  # first four elements are locations of bounding boxes,
        # next is the confidence, all other 80 are class probabs
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)  # detections are normalised, therefore we multiply
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# if len(indexes) > 0: #use this for tiny
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = colors[class_ids[i]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

imS = cv2.resize(img, (550, 800))
cv2.imshow("Output", imS)
cv2.waitKey(0)



#cap.release()
cv2.destroyAllWindows()
