import os

import cv2

# modelFile = 'faster_rcnn_resnet50/frozen_inference_graph.pb '
# configFile = 'faster_rcnn_resnet50/faster_rcnn_resnet50_coco_2018_01_28.pbtxt'
modelFile = 'ssd_mobelNet_v2/frozen_inference_graph.pb'
configFile = 'ssd_mobelNet_v2/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
classFile = 'object_detection_classes_coco.txt'

if not os.path.exists(modelFile):
    print('Cannot find model file in location')

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
with open(classFile) as fp:
    labels = fp.read().split('\n')


def detect_objects(net, img):
    """Run object detection over the input image."""
    # Blob dimension (dim x dim)
    dim = 300

    mean = (0, 0, 0)

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects


def draw_text(im, text, x, y):
    """Draws text label at a given x-y position with a black background."""
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    # Get text size
    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle.
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def draw_objects(im, objects, threshold=0.25):
    """Displays a box and text for each detected object exceeding the confidence threshold."""
    rows = im.shape[0]
    cols = im.shape[1]

    # For every detected object.
    for i in range(objects.shape[2]):
        # Find the class and confidence.
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        # Check if the detection is of good quality
        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Convert Image to RGB since we are using Matplotlib for displaying image.
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return mp_img

path ='C:/Users/Thunder/PycharmProjects/kikkoman_demo/demo/20230816/102341.mp4'
vcap = cv2.VideoCapture(path)
while True:
    ret, frame = vcap.read()
    if not ret:
        break
    img = frame.copy()
    results = detect_objects(net, img)
    frame = draw_objects(frame, results, 0.8)
    frame = frame[:, :, ::-1]

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#
vcap.release()
cv2.destroyAllWindows()
