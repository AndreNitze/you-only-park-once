import argparse
import datetime
import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import natsort
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# initialize maximum number of vehicles
max_count = 0

# initialize number of vehicles sequence
cars_count = np.array([])

total_time_start = time.time()
img_array = []
print(str(len(glob.glob('F:/Crowd Counter/images/*.jpg'))) + " images found")
for filename in natsort.natsorted(glob.glob('F:/Crowd Counter/images/*.jpg')):
    if os.stat(filename).st_size == 0:
        cars_count = np.append(cars_count, 0)
        break

    # load our input image and grab its spatial dimensions
    image = cv2.imread(filename)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds for image {}".format(end - start, filename))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > max_count:
        max_count = len(idxs)

    # Append amount of vehicles to plot
    cars_count = np.append(cars_count, len(idxs))

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if LABELS[classIDs[i]] in ("car", "truck", "person"):
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                car_count_text = "Number of recognized vehicles: {}".format(len(idxs))
                cv2.putText(image, car_count_text, (5, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    max_count_text = "Maximum number of vehicles: {}".format(max_count)
    cv2.putText(image, max_count_text, (5, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # store image for creating video
    cv2.imshow("image", image)
    cv2.waitKey(0)
    img_array.append(image)

# save output images to video
out = cv2.VideoWriter('F:/Crowd Counter/output/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# Data for plotting
print(str(len(cars_count)) + " images processed")
total_time_end = time.time()
print("{:.1f} minutes passed".format((total_time_end - total_time_start)/60))

customdate = datetime.datetime(2019, 6, 2, 19, 10)
x = [customdate + datetime.timedelta(minutes=i*5) for i in range(len(cars_count))]

# Write CSV
cars_array = np.asarray(cars_count, dtype=int)
dates_array = np.asarray(x, dtype=np.unicode)
np.savetxt("car-data.csv", np.c_[dates_array, cars_array], delimiter=';', fmt='%s')

# Create plot
fig, ax = plt.subplots()
ax.plot(x, cars_count)
ax.set(xlabel='Time', ylabel='Number of vehicles',
       title='Parking lot utilization over time')
ax.grid()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a, %d.%m.'))
fig.autofmt_xdate()
fig.savefig("car-plot.png")
plt.show()
