import cv2
import numpy as np
import time

# Install pygame for a better experience using pip install pygame
try:
    from pygame import mixer
except ModuleNotFoundError:
    mixer = None
    pass

# class to create a real time plot
class Plotter:
    def __init__(self, plot_width, plot_height, sample_buffer=None, scale_value=1):
        self.scale_value = scale_value
        self.width = plot_width
        self.height = plot_height
        self.color = (0, 255, 0)
        self.plot_canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 50
        self.margin_r = 50
        self.margin_u = 50
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer

    def plot(self, val, label="plot", t1=1, t2=1):
        self.t1 = t1
        self.t2 = t2
        if label not in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0

        self.plots[label].append(int(val * self.scale_value) / self.scale_value)
        if len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)

        self.show_plot(label)

    def show_plot(self, label):
        self.plot_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        scale_h = 8 * self.scale_value * (self.height - self.margin_d - self.margin_u) / self.height

        x_vals = np.linspace(0, self.sample_buffer - 2, self.width - self.margin_l - self.margin_r).astype(int)
        for j, i in enumerate(x_vals):
            if i + 1 >= len(self.plots[label]):
                break
            y1 = int((self.height - self.margin_d - self.margin_u) * (1 - self.plots[label][i]) + self.margin_u)
            y2 = int((self.height - self.margin_d - self.margin_u) * (1 - self.plots[label][i + 1]) + self.margin_u)
            cv2.line(self.plot_canvas, (j + self.margin_l, y1), (j + self.margin_l + 1, y2), self.color, 1)

        # Draw border and grid
        cv2.rectangle(self.plot_canvas, (self.margin_l, self.margin_u),
                      (self.width - self.margin_r, self.height - self.margin_d), (255, 255, 255), 1)

        for y_frac, label_text in zip([0.25, 0.5, 0.75], ["0.50", "0.25", "0.0"]):
            y = int((self.height - self.margin_d - self.margin_u) * y_frac + self.margin_u)
            cv2.line(self.plot_canvas, (self.margin_l, y), (self.width - self.margin_r, y), (50, 50, 50), 1)
            cv2.putText(self.plot_canvas, label_text, (10, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(self.plot_canvas, f"{label}: {self.plots[label][-1]:.2f}",
                    (int(self.width / 2 - 50), self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(1)
#----------------------------------
#1. Initalizations
#----------------------------------

# initalize the counter for the number of blinks detected
BLINK=0

# model file paths.
#MODEL_PATH =
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_PATH="deploy.prototxt"
LBF_MODEL='lbfmodel.yaml'

# Create a face detector network instance
net=cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

# create a landmark detector instance
LandmarkDetector=cv2.face.createFacemarkLBF()
LandmarkDetector.loadModel(LBF_MODEL)

# initalize the video capture object
cap=cv2.VideoCapture('input-video.mp4')
state_prev=state_curr = 'open'


# -------------------------------------
# 2. Function Definitions
#--------------------------------------



def detect_faces(image, detection_threshold=0.70):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    img_h, img_w = image.shape[:2]

    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left = int(detection[3] * img_w)
            top = int(detection[4] * img_h)
            right = int(detection[5] * img_w)
            bottom = int(detection[6] * img_h)
            face_w = right - left
            face_h = bottom - top
            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)

def get_primaryface(faces, frame_h, frame_w):

    primary_face_index=None
    face_height_max=0
    for idx in range(len(faces)):
        face=faces[idx]
        # confirm bounding box of primary face does not exceed from size.
        x1=face[0]
        y1=face[1]
        x2=x1+face[2]
        y2=y1+face[3]
        if x1> frame_w or y1> frame_h or x2>frame_w or y2> frame_h:
            continue
        if x1<0 or y1<0 or x2<0 or y2<0:
            continue
        if face[3]>face_height_max:
            primary_face_index=idx

    if primary_face_index is not None:
        primary_face=faces[primary_face_index]
    else:
        primary_face=None
    return primary_face

def visualize_eyes(frame, landmarks):
    for i in range(36, 48):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 255, 0), -1)

# Using this paper : http://vision.fe.uni-lj.si/cvwm2016/proceedings/papers/05.pdf
def get_eye_aspect_ratio(landmarks):
    # compute the euclidean distances between the two sets of vertical eye landmarks
    vert_dist_1right = calculate_distance(landmarks[37], landmarks[41])
    vert_dist_2right = calculate_distance(landmarks[38], landmarks[40])
    vert_dist_1left = calculate_distance(landmarks[43], landmarks[47])
    vert_dist_2left = calculate_distance(landmarks[44], landmarks[46])

    # compute the euclidean distances between the two sets of horizontal eye landmarks
    horz_dist_right = calculate_distance(landmarks[36], landmarks[39])
    horz_dist_left = calculate_distance(landmarks[42], landmarks[45])
    #compute the eye aspect ratio
    EAR_left= (vert_dist_1left + vert_dist_2left)/ (2.0 * horz_dist_left)
    EAR_right= (vert_dist_1right + vert_dist_2right)/ (2.0 * horz_dist_right)

    ear= (EAR_left + EAR_right)/2
    # return the eye aspect ratio
    return ear

def calculate_distance(A,B):
    distance= ((A[0] - B[0])**2+(A[1] - B[1])**2)**0.5
    return distance

def play(file):
    mixer.init()
    sound=mixer.Sound(file)
    sound.play()


# -------------------------------------
# 3. Main Processing Loop
# -------------------------------------

# EAR threshold and consecutive frame limit
EAR_THRESH = 0.25
CONSEC_FRAMES = 3
frame_counter = 0

# Plotter setup
plotter = Plotter(plot_width=600, plot_height=300, scale_value=100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect faces
    faces = detect_faces(frame)

    # Step 2: Choose the primary face
    face = get_primaryface(faces, frame.shape[0], frame.shape[1])
    if face is None:
        continue

    (x, y, w, h) = face
    face_rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Step 3: Detect landmarks
    _, landmarks = LandmarkDetector.fit(gray, np.array([face]))

    if len(landmarks) > 0:
        shape = landmarks[0][0]  # (68, 2) landmarks

        # Step 4: Calculate EAR
        ear = get_eye_aspect_ratio(shape)

        # Step 5: Visualize eyes
        visualize_eyes(frame, shape)

        # Step 6: Blink Detection
        if ear < EAR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                BLINK += 1
                if mixer:
                    play("click.wav")  # Make sure alert.wav exists
            frame_counter = 0

        # Step 7: Display EAR and Blink Count
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {BLINK}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Step 8: Plot EAR in real-time
        plotter.plot(ear, label="EAR")

    # Show video frame
    cv2.imshow("Frame", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()










