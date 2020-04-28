from tracking import CentroidTracker, TrackableObject
from scipy.spatial import distance as dist
from imutils.video import FPS
import numpy as np
import cvlib as cv
import imutils
import cv2

ONE_METER = 50  # px
ALLOWED_DISTANCE = 2  # m

MAX_PEOPLE_IN_AREA = 15

x = 1500
y = 620
z = 330
q = 500

IMAGE_SIZE = (900, 506)
ORIGINAL_SIZE = (1920, 1080)
BORDER_SIZE = 5

DST = np.float32([[x, y], [x, z], [q, y], [q, z]])
SRC = np.float32([[582, 562], [1085, 700], [1322, 105], [1647, 143]])

M = cv2.getPerspectiveTransform(SRC, DST)
print(M)


def frame_to_brid_view(img, transform_matrix):
    return cv2.warpPerspective(img, transform_matrix, ORIGINAL_SIZE)


def point_to_bird_view(p):
    px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
    return int(px), int(py)


VIDEO_NAME = 'TownCentreXVID.avi'  # "People.mp4"
MINIMUM_CONFIDENCE = 0.3
SKIP_FRAMES = 10
MAX_DISAPPEARED = 10

camera = cv2.VideoCapture(VIDEO_NAME)
cv2.namedWindow("People Tracker")

WIDTH, HEIGHT = None, None
ct = CentroidTracker(maxDisappeared=MAX_DISAPPEARED)
trackers = []
trackableObjects = {}
totalFrames = 0
fps = FPS().start()

while True:
    success, frame = camera.read()
    if frame is None:
        break

    #  frame = imutils.resize(frame, width=900)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if WIDTH is None or HEIGHT is None:
        (HEIGHT, WIDTH) = frame.shape[:2]

    status = "waiting"
    rects = []
    rects_transformed = []
    centroids_transformed = []

    if totalFrames % SKIP_FRAMES == 0: # running detector
        status = "detecting"
        trackers = []

        bbox, label, conf = cv.detect_common_objects(rgb, confidence=MINIMUM_CONFIDENCE, model='yolov3')

        for i in range(len(label)): # loop over the detections
            if label[i] != "person":
                continue
            startX, startY, endX, endY = bbox[i]
            tracker = cv2.TrackerMedianFlow_create()
            rect = (startX, startY, endX - startX, endY - startY)
            tracker.init(frame, rect)
            trackers.append(tracker)
    else:  # updating trackers
        for tracker in trackers:
            status = "tracking"
            (success, box) = tracker.update(frame)  # update the tracker and grab the updated position
            if success:
                (x, y, w, h) = [int(v) for v in box]
                rects.append((x, y, x + w, y + h))

    if rects:
        centroids = [(int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)) for (x1, y1, x2, y2) in rects]
        centroids_transformed = [point_to_bird_view(cnt) for cnt in centroids]

        D = dist.cdist(np.array(centroids), np.array(centroids))
        np.fill_diagonal(D, np.nan)
        D = np.nanmin(D, axis=0)

        D_transformed = dist.cdist(np.array(centroids_transformed), np.array(centroids_transformed))
        np.fill_diagonal(D_transformed, np.nan)
        D_argmin_arr = np.nanargmin(D_transformed, axis=0)
        D_transformed = np.nanmin(D_transformed, axis=0)

        for i in range(D.shape[0]):
            x1, y1, x2, y2 = rects[i]

            if D_transformed[i] <= ONE_METER * ALLOWED_DISTANCE:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

                D_argmin = D_argmin_arr[i]
                pt_1 = (centroids[i][0], centroids[i][1])
                pt_2 = (centroids[D_argmin][0], centroids[D_argmin][1])
                cv2.line(frame, pt_1, pt_2, (0, 0, 200), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # if D[i] < 50:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 180), 2)
            # else:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Coordinate label
            cnt = centroids[i]
            coords = f'{cnt[0]}, {cnt[1]}'
            # cv2.putText(frame, coords, (cnt[0], cnt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 1)

    # use the centroid tracker to associate the old object centroids with the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current object ID
        trackable_object = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if trackable_object is None:
            to = TrackableObject(objectID, centroid)
        trackableObjects[objectID] = trackable_object

    text = "People: {}".format(len(rects))
    cv2.putText(frame, text, (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 2)

    frame_bird = frame_to_brid_view(frame, M)

    frame = imutils.resize(frame, width=IMAGE_SIZE[0])
    frame_bird = imutils.resize(frame_bird, width=IMAGE_SIZE[0])

    # Coordinate labels
    for cntr in centroids_transformed:
        coords = f'{cntr[0]}, {cntr[1]}'
        # cv2.putText(frame_bird, coords, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 1)

    border_opacity = 255 if len(rects) > MAX_PEOPLE_IN_AREA else 0
    frame_alert_color = [0, 0, 250, border_opacity] if len(rects) > MAX_PEOPLE_IN_AREA else [0, 0, 0, 0]

    frame = cv2.copyMakeBorder(frame, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                               cv2.BORDER_CONSTANT, value=frame_alert_color)
    frame_bird = cv2.copyMakeBorder(frame_bird, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    full_frame = np.concatenate((frame, frame_bird), axis=1)
    cv2.imshow("People Tracker", full_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # quit
        break

    totalFrames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

camera.release()
cv2.destroyAllWindows()