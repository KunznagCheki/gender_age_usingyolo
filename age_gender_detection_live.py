import cv2
import time
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.75):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
    return frame, bboxes

# Load models
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

# Find available camera index
for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Using camera index {index}")
        break
else:
    print("No camera found.")
    exit()

padding = 20

while cv2.waitKey(1) < 0:
    start_time = time.time()
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        print("No frame captured. Exiting...")
        break

    # Resize for optimization
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    
    if not bboxes:
        print("No face detected.")
        continue

    for bbox in bboxes:
        # Adjust bounding box to original frame
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, small_frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, small_frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Label detected face
        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Scale frame for better visibility
    large_frame = cv2.resize(frameFace, (0, 0), fx=2, fy=2)
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Center the large frame on the canvas
    x_offset = (canvas.shape[1] - large_frame.shape[1]) // 2
    y_offset = (canvas.shape[0] - large_frame.shape[0]) // 2
    canvas[y_offset:y_offset + large_frame.shape[0], x_offset:x_offset + large_frame.shape[1]] = large_frame

    # Display the centered canvas
    cv2.imshow("Age Gender Demo", canvas)
    print(f"Processing Time: {time.time() - start_time:.3f} seconds")

cap.release()
cv2.destroyAllWindows()
