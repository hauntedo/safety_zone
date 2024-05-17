import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import math
import cvzone
import lane

yolo = YOLO('../../yolov/weights/yolov8n.pt')
video_capture = cv.VideoCapture("../../data/cars_2.mp4")

classNames = yolo.model.names

if not video_capture.isOpened():
    print("Could not open video stream")

cv.waitKey(1)

cv.namedWindow("Video", cv.WINDOW_NORMAL)
cv.resizeWindow("Video", 1300, 800)

counter = object_counter.ObjectCounter()

while video_capture.isOpened():
    _, frame = video_capture.read()

    copy_frame = np.copy(frame)

    try:
        frame = lane.canny(frame)
        frame = lane.roi(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi / 180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        average_lines = lane.average_slope_intercept(frame, lines)
        line_image = lane.display_lines_2(copy_frame, average_lines)
        combo = cv.addWeighted(copy_frame, 0.8, line_image, 0.5, 1)

        # region_points = [reg_p(0), reg_p(1), reg_p(0), reg_p(1)]
        # counter.set_args(view_img=True,
        #                  reg_pts=region_points,
        #                  classes_names=yolo.names,
        #                  draw_tracks=True,
        #                  line_thickness=2)

        # tracks = yolo.track(combo, persist=True, show=False)
        # combo = counter.start_counting(combo, tracks)

        # results = yolo(combo, stream=True)
        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         # x1, y1, x2, y2 = box.xyxy[0]
        #         # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
        #         # Bounding Box
        #         x1, y1, x2, y2 = box.xyxy[0]
        #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        #         # Confidence
        #         conf = math.ceil((box.conf[0] * 100)) / 100
        #
        #         # Class Names
        #         obj = classNames[int(box.cls[0])]
        #
        #         if (obj == 'car' or obj == 'truck') and conf >= 0.3:
        #             w, h = x2 - x1, y2 - y1
        #             cvzone.cornerRect(combo, (x1, y1, w, h))
        #             cvzone.putTextRect(combo, f'{obj} {conf}', (x1, y1 - 20))

        cv.imshow("Video", combo)
    except:
        pass

    if cv.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        cv.destroyAllWindows()

video_capture.release()
cv.destroyAllWindows()
