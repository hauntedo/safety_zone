from ultralytics import YOLO
from object_counter import ObjectCounter
import cv2

def define_region_points(w, h):
    tl = 500  # top left
    tr = w - tl  # top right
    bl = 200  # bottom left
    br = w - bl  # bottom right
    bh = h  # bottom height
    return [(tl, h//1.7), (tr, h//1.7), (br, bh), (bl, bh)]

model = YOLO("../yolov/weights/yolov8n.pt")
cap = cv2.VideoCapture("../data/cars_5.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = define_region_points(w, h)

counter = ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)
    im0, is_inside = counter.start_counting(im0, tracks)
    # возвращаем ответ, если is_inside = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()