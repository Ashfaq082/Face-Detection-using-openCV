import cv2
import numpy as np

s=0 # use web camera
video_cap=cv2.VideoCapture(s)

win_name=' Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Create a network object
net=cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel' )

# Model Parameters used to train model
mean=[104,117,123]
scale=1.0
in_width=300
in_height=300

# set the detection threshold for face detections
detection_threshold=0.5

# Annotating settings
font_style=cv2.FONT_HERSHEY_SIMPLEX
font_scale=0.5
font_thickness=1

while True:
    has_frame,frame=video_cap.read()
    if not has_frame:
        break
    h=frame.shape[0]
    w=frame.shape[1]
    # flip the video frame horizontally ( not required, just for convenience
    frame=cv2.flip(frame,1)

    # convert the image in to a blob format
    blob=cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width,in_height), mean=mean, swapRB=False, crop=False)
    # pass the blob to the DNN model
    net.setInput(blob)
    # Retrieve detections from DNN model
    detections=net.forward()

    # Process Each detection.
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > detection_threshold:

            # Extra bounding box coordinates from detection.
            box=detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2)= box.astype('int')

            # Annotate video frame with the detection result
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
            label= 'confidence: %4f' % confidence
            label_size,base_line=cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1,y1-label_size[1]), (x1 + label_size[0], y1+base_line), (255,255,255), cv2.FILLED)
            cv2.putText(frame, label, (x1,y1), font_style, font_scale,(0,0,0))

    cv2.imshow(win_name, frame)
    key=cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
video_cap.release()
cv2.destroyWindow(win_name)






