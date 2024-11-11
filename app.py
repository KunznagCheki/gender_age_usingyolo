import cv2
import time
import os

def getFaceBox(net, frame, conf_threshold=0.75):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1, y1 = int(detections[0, 0, i, 3] * frameWidth), int(detections[0, 0, i, 4] * frameHeight)
            x2, y2 = int(detections[0, 0, i, 5] * frameWidth), int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxes

# Load models
faceProto, faceModel = "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"
ageProto, ageModel = "age_deploy.prototxt", "age_net.caffemodel"
genderProto, genderModel = "gender_deploy.prototxt", "gender_net.caffemodel"
ageNet, genderNet, faceNet = cv2.dnn.readNet(ageModel, ageProto), cv2.dnn.readNet(genderModel, genderProto), cv2.dnn.readNet(faceModel, faceProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList, genderList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'], ['Male', 'Female']

cap = cv2.VideoCapture(0)
padding = 20

# Adjust window display size (only if GUI is available)
window_scale = 0.6

# Check if DISPLAY environment variable is set (which indicates a GUI environment)
if 'DISPLAY' in os.environ:
    cv2.namedWindow("Age Gender Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Age Gender Demo", int(640 * window_scale), int(480 * window_scale))
else:
    print("Running in a headless environment, display disabled.")

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face detected.")
        continue
    
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # If GUI is available, display the image
    if 'DISPLAY' in os.environ:
        # Display with centered and adjusted frame size
        frame_display = cv2.resize(frameFace, (int(frame.shape[1] * window_scale), int(frame.shape[0] * window_scale)))
        cv2.imshow("Age Gender Demo", frame_display)
    else:
        # Save the frame to a file instead of displaying it
        cv2.imwrite("output_frame.jpg", frameFace)  # Save the image to a file

cap.release()

# Close all OpenCV windows if running in GUI mode
if 'DISPLAY' in os.environ:
    cv2.destroyAllWindows()
