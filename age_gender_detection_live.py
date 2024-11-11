import cv2
import time
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(0)
padding = 20

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        cv2.waitKey()
        break

    # Creating a smaller frame for better optimization
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue
    
    for bbox in bboxes:
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Label for the detected face
        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Resize the frame to make it larger
    large_frame = cv2.resize(frameFace, (0, 0), fx=2, fy=2)
    
    # Create a black canvas
    canvas_height = 1080
    canvas_width = 1920
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Get coordinates to place the large frame in the center
    x_offset = (canvas_width - large_frame.shape[1]) // 2
    y_offset = (canvas_height - large_frame.shape[0]) // 2

    # Place the enlarged frame in the center of the canvas
    canvas[y_offset:y_offset + large_frame.shape[0], x_offset:x_offset + large_frame.shape[1]] = large_frame

    # Display the centered canvas
    cv2.imshow("Age Gender Demo", canvas)

    print("Time: {:.3f}".format(time.time() - t))
