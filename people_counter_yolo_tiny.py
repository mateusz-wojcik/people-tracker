from tracking import CentroidTracker, TrackableObject
from scipy.spatial import distance as dist
from imutils.video import FPS
import numpy as np
import cvlib as cv
import imutils
import cv2



#VIDEO_NAME = "TownCentreXVID.avi"
VIDEO_NAME = "People.mp4"
MINIMUM_CONFIDENCE = 0.3
SKIP_FRAMES = 50
MAX_DISAPPEARED = 30

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

    frame = imutils.resize(frame, width=900)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if WIDTH is None or HEIGHT is None:
        (HEIGHT, WIDTH) = frame.shape[:2]

    status = "waiting"
    rects = []

    if totalFrames % SKIP_FRAMES == 0: # running detector
        status = "detecting"
        trackers = []

        bbox, label, conf = cv.detect_common_objects(rgb, confidence=MINIMUM_CONFIDENCE, model='yolov3-tiny')

        for i in range(len(label)): # loop over the detections
            if label[i] != "person":
                continue
            startX, startY, endX, endY = bbox[i]
            tracker = cv2.TrackerMedianFlow_create()
            rect = (startX, startY, endX-startX, endY-startY)
            tracker.init(frame, rect)
            trackers.append(tracker)
    else:  # updating trackers
        for tracker in trackers:
            status = "tracking"
            (success, box) = tracker.update(frame)  # update the tracker and grab the updated position
            if success:
                (x, y, w, h) = [int(v) for v in box]
                rects.append((x, y, x+w, y+h))

    if rects:
        centroids = [(int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)) for (x1, y1, x2, y2) in rects]
        if len(rects) > 1:
            D = dist.cdist(np.array(centroids), np.array(centroids))
            np.fill_diagonal(D, np.nan)
            D = np.nanmin(D, axis=0)
            for i in range(D.shape[0]):
                x1, y1, x2, y2 = rects[i]
                if D[i] < 50:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 180), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            x1, y1, x2, y2 = rects[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
    cv2.imshow("People Tracker", frame)

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